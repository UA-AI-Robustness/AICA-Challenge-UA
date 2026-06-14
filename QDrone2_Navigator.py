# =============================================================================
# QDrone2 Navigator — restored to exact 5675 known-good configuration
# -----------------------------------------------------------------------------
# Reverted the continuous-intention-resend "fix" - it introduced a regression
# (D2 and D3 pickups/drops failed). Returning to the proven send-once-per-
# state-transition pattern that achieved 5648/5670/5675.
# =============================================================================

import numpy as np
import cv2
from pathlib import Path

try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout

from pal.utilities.stream import BasicStream
from pal.utilities.timing import QTimer
from pal.utilities.vision import Camera2D, Camera3D

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List


class DroneState(Enum):
    IDLE = auto()
    APPROACHING_WAYPOINT = auto()
    AT_WAYPOINT_HOLDING = auto()
    ACTION_COMPLETE = auto()
    MISSION_COMPLETE = auto()


DRONE_INTENTION_NOTHING = 0
DRONE_INTENTION_PICKUP_SMALL = 1
DRONE_INTENTION_DROPOFF = 2
DRONE_INTENTION_TRANSFER_FROM_CAR = 3
DRONE_INTENTION_TRANSFER_TO_CAR = 4

DRONE_HORIZONTAL_TOLERANCE_M = 2.0
DRONE_VERTICAL_TOLERANCE_BELOW_M = 2.0
DRONE_VERTICAL_TOLERANCE_ABOVE_M = 4.0

DRONE_HOLD_DURATION_SEC = 3.05

DRONE_CRUISE_ALTITUDE_M = 3.0

WINDOW_Z_BUFFER_M = 0.35
WINDOW_Z_THRESHOLD_M = 4.0


@dataclass
class DroneMissionAction:
    target_xyz: np.ndarray
    target_yaw: float = 0.0
    intention: int = DRONE_INTENTION_NOTHING
    description: str = ""
    hold_duration: float = DRONE_HOLD_DURATION_SEC
    via_points: Optional[List[np.ndarray]] = None
    flight_time_override: Optional[float] = None


@dataclass
class DroneMissionState:
    actions: List[DroneMissionAction] = field(default_factory=list)
    current_action_idx: int = 0
    cargo_small_count: int = 0

    @property
    def current_action(self) -> Optional[DroneMissionAction]:
        if self.current_action_idx < len(self.actions):
            return self.actions[self.current_action_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_action_idx >= len(self.actions)

    def advance(self):
        self.current_action_idx += 1


def has_arrived_drone(current_pose: np.ndarray, target_xyz: np.ndarray,
                      horizontal_tol: float = DRONE_HORIZONTAL_TOLERANCE_M,
                      vertical_tol_below: float = DRONE_VERTICAL_TOLERANCE_BELOW_M,
                      vertical_tol_above: float = DRONE_VERTICAL_TOLERANCE_ABOVE_M) -> bool:
    horizontal_dist = np.linalg.norm(current_pose[:2] - target_xyz[:2])
    if horizontal_dist > horizontal_tol:
        return False
    vertical_offset = current_pose[2] - target_xyz[2]
    if not (-vertical_tol_below <= vertical_offset <= vertical_tol_above):
        return False
    return True


def hold_completed_drone(hold_start_time: Optional[float], current_time: float,
                         duration: float = DRONE_HOLD_DURATION_SEC) -> bool:
    if hold_start_time is None:
        return False
    return (current_time - hold_start_time) >= duration


D2_SAFE_ALTITUDE = 13.0
D2_APPROACH_1 = np.array([0.0, 29.0, D2_SAFE_ALTITUDE])


def apply_window_z_buffer(target_xyz: np.ndarray) -> np.ndarray:
    target = target_xyz.copy()
    if target[2] >= WINDOW_Z_THRESHOLD_M:
        target[2] += WINDOW_Z_BUFFER_M
    return target


D3_WINDOW_FLIGHT_SEC = 10.0
D3_RETURN_FLIGHT_SEC = 9.0
D1_WINDOW_FLIGHT_SEC = 13.0
D1_RETURN_FLIGHT_SEC = 10.0
D2_WINDOW_FLIGHT_SEC = 22.0


def build_drone_mission_three_windows() -> DroneMissionState:
    pickup_xyz = np.array([-2.50305, 29.6703, DRONE_CRUISE_ALTITUDE_M])
    d1_window_xyz = apply_window_z_buffer(np.array([15.1739, -18.04655, 9.65]))
    d2_window_xyz = apply_window_z_buffer(np.array([26.0478, 16.7703, 9.65]))
    d3_window_xyz = apply_window_z_buffer(np.array([1.3, 46.9735, 4.85]))

    mission = DroneMissionState()
    mission.actions = [
        DroneMissionAction(target_xyz=pickup_xyz,
                           intention=DRONE_INTENTION_PICKUP_SMALL,
                           description="Pickup small #1 at central pickup (immediate, spawned here)"),
        DroneMissionAction(target_xyz=d3_window_xyz,
                           intention=DRONE_INTENTION_DROPOFF,
                           description="Drop off at Delivery 3 window (floor 2, +200)",
                           flight_time_override=D3_WINDOW_FLIGHT_SEC),
        DroneMissionAction(target_xyz=pickup_xyz,
                           intention=DRONE_INTENTION_PICKUP_SMALL,
                           description="Pickup small #2 at central pickup",
                           flight_time_override=D3_RETURN_FLIGHT_SEC),
        DroneMissionAction(target_xyz=d1_window_xyz,
                           intention=DRONE_INTENTION_DROPOFF,
                           description="Drop off at Delivery 1 window (floor 4, +400)",
                           flight_time_override=D1_WINDOW_FLIGHT_SEC),
        DroneMissionAction(target_xyz=pickup_xyz,
                           intention=DRONE_INTENTION_PICKUP_SMALL,
                           description="Pickup small #3 at central pickup",
                           flight_time_override=D1_RETURN_FLIGHT_SEC),
        DroneMissionAction(target_xyz=d2_window_xyz,
                           intention=DRONE_INTENTION_DROPOFF,
                           description="Drop off at Delivery 2 window (floor 3, +300) via climb-out",
                           via_points=[D2_APPROACH_1],
                           flight_time_override=D2_WINDOW_FLIGHT_SEC),
    ]
    return mission


DEFAULT_FLIGHT_TIME_SEC = 12.0
WINDOW_FLIGHT_TIME_SEC = 16.0


@dataclass
class DroneTrajectory:
    waypoints: List[np.ndarray]
    start_time: float
    duration: float

    def waypoint_at(self, current_time: float) -> np.ndarray:
        elapsed = current_time - self.start_time
        if elapsed <= 0.0:
            return self.waypoints[0].copy()
        if elapsed >= self.duration:
            return self.waypoints[-1].copy()
        n_segments = len(self.waypoints) - 1
        if n_segments < 1:
            return self.waypoints[-1].copy()
        segment_duration = self.duration / n_segments
        segment_index = int(min(elapsed / segment_duration, n_segments - 1))
        local_elapsed = elapsed - segment_index * segment_duration
        alpha = local_elapsed / segment_duration
        wp0 = self.waypoints[segment_index]
        wp1 = self.waypoints[segment_index + 1]
        return ((1.0 - alpha) * wp0 + alpha * wp1).astype(np.float64)


def make_trajectory(start_xyz: np.ndarray, end_xyz: np.ndarray,
                    current_time: float,
                    duration: float = DEFAULT_FLIGHT_TIME_SEC,
                    via_points: Optional[List[np.ndarray]] = None) -> DroneTrajectory:
    waypoints = [start_xyz.copy()]
    if via_points:
        waypoints.extend(np.array(p, dtype=np.float64) for p in via_points)
    waypoints.append(end_xyz.copy())
    return DroneTrajectory(waypoints=waypoints, start_time=current_time, duration=duration)


def get_drone_pose_safe(received_data: np.ndarray, fallback_pose: np.ndarray) -> np.ndarray:
    pose_from_telemetry = np.asarray(received_data[16:20], dtype=np.float64)
    if np.allclose(pose_from_telemetry, 0.0):
        return fallback_pose.astype(np.float64)
    return pose_from_telemetry


def read_initial_positions(filepath: Path) -> np.ndarray:
    if not filepath.exists():
        raise FileNotFoundError(f"Spawn file not found: {filepath}")
    values: list[float] = []
    with filepath.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parts = [float(value.strip()) for value in line.split(",")]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in {filepath} on line {line_number}: {raw_line.strip()}"
                ) from exc
            values.extend(parts)
    if len(values) < 8:
        raise ValueError(
            f"{filepath} must contain at least 8 numeric values, but found {len(values)}."
        )
    return np.array(values[4:8], dtype=np.float64)


def load_plan_file(plan_path: Path):
    plan_data = np.load(plan_path, allow_pickle=True)
    qdrone2_wp1 = np.asarray(plan_data["qdrone2_wp1"])
    qdrone2_t1 = np.asarray(plan_data["qdrone2_t1"]).flatten()
    return qdrone2_wp1, qdrone2_t1


simulationTime = 10000
frequency = 200
frameRate = 30
CameraCounts = int(round(frequency / frameRate))
useCameras = False

counter = 0
receiveCounter = 0
receivedData = np.zeros(16)

initial_position = read_initial_positions(Path("spawn_locations.txt"))
plan_path = Path(r"tools\QDrone2_PathPlanning\qdrone2_plans.npz")
qdrone2_wp1, qdrone2_t1 = load_plan_file(plan_path)

if useCameras:
    realsense = Camera3D(deviceId="0@tcpip://localhost:18986", mode='RGB&DEPTH',
        frameWidthRGB=640, frameHeightRGB=480, frameRateRGB=frameRate,
        frameWidthDepth=640, frameHeightDepth=480, frameRateDepth=frameRate, readMode=0)
    camRight = Camera2D(cameraId="0@tcpip://localhost:18982", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camBack  = Camera2D(cameraId="1@tcpip://localhost:18983", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camLeft  = Camera2D(cameraId="2@tcpip://localhost:18984", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camDown  = Camera2D(cameraId="3@tcpip://localhost:18985", frameWidth=640, frameHeight=480, frameRate=frameRate)


dataStream = BasicStream(
    'tcpip://localhost:18373',
    agent='C',
    sendBufferSize=1460,
    receiveBuffer=np.zeros((1, 20), dtype=np.float64),
    recvBufferSize=1460,
    nonBlocking=False
)

client_drone = BasicStream(
    'tcpip://localhost:19001',
    agent='C',
    sendBufferSize=8,
    receiveBuffer=np.zeros((1, 3), dtype=np.float64),
    recvBufferSize=24,
    nonBlocking=False
)

timeout = Timeout(seconds=0, nanoseconds=10)


timer = QTimer(frequency, simulationTime)

flag_send_intention = True
intention = DRONE_INTENTION_NOTHING

if initial_position[2] >= 1.5:
    hover_command = initial_position.copy()
    print(f"[INIT] Drone spawn at altitude (z={initial_position[2]:.2f}); hovering in place.")
else:
    hover_command = initial_position + np.array([0.0, 0.0, DRONE_CRUISE_ALTITUDE_M, 0.0], dtype=np.float64)
    print(f"[INIT] Drone spawn on ground (z={initial_position[2]:.2f}); will take off to {DRONE_CRUISE_ALTITUDE_M}m.")
send_commands = hover_command.copy()

mission = build_drone_mission_three_windows()
print(f"Drone mission loaded with {len(mission.actions)} actions:")
for i, action in enumerate(mission.actions):
    ft = action.flight_time_override if action.flight_time_override is not None else "default"
    print(f"  [{i}] {action.description} (flight_time={ft})")
print(f"Settings: hold={DRONE_HOLD_DURATION_SEC}s")

drone_state = DroneState.IDLE
hold_start_time: Optional[float] = None
current_trajectory: Optional[DroneTrajectory] = None

pose = np.array([
    initial_position[0],
    initial_position[1],
    initial_position[2],
    initial_position[3],
])


try:
    while timer.check():
        current_time = timer.get_current_time()

        if not np.all(np.isfinite(pose)):
            print(f"[WARN] Invalid pose at counter={counter}: {pose} - using fallback")
            pose = np.array([
                initial_position[0], initial_position[1],
                max(initial_position[2], DRONE_CRUISE_ALTITUDE_M),
                initial_position[3]
            ])

        if drone_state == DroneState.IDLE:
            if not mission.is_complete:
                action = mission.current_action
                is_window = action.target_xyz[2] >= WINDOW_Z_THRESHOLD_M
                flight_time = WINDOW_FLIGHT_TIME_SEC if is_window else DEFAULT_FLIGHT_TIME_SEC
                if action.flight_time_override is not None:
                    flight_time = action.flight_time_override

                start = pose[:3].copy()
                via = []
                if start[2] < 1.5:
                    takeoff_xyz = np.array([start[0], start[1], DRONE_CRUISE_ALTITUDE_M])
                    via.append(takeoff_xyz)
                    flight_time += 4.0
                    print(f"[TRAJ] Adding takeoff via-point at {takeoff_xyz}")
                if action.via_points is not None:
                    via.extend(action.via_points)

                current_trajectory = make_trajectory(
                    start_xyz=start, end_xyz=action.target_xyz,
                    current_time=current_time, duration=flight_time,
                    via_points=via if via else None,
                )
                intention = DRONE_INTENTION_NOTHING
                flag_send_intention = True
                drone_state = DroneState.APPROACHING_WAYPOINT
                print(f"[STATE] IDLE -> APPROACHING_WAYPOINT "
                      f"(target=({action.target_xyz[0]:.2f}, "
                      f"{action.target_xyz[1]:.2f}, {action.target_xyz[2]:.2f}), "
                      f"flight_time={flight_time:.1f}s)")
            else:
                drone_state = DroneState.MISSION_COMPLETE

        elif drone_state == DroneState.APPROACHING_WAYPOINT:
            action = mission.current_action
            if current_trajectory is not None:
                waypoint = current_trajectory.waypoint_at(current_time)
            else:
                waypoint = action.target_xyz

            send_commands = np.array([
                waypoint[0], waypoint[1], waypoint[2], action.target_yaw,
            ], dtype=np.float64)

            if has_arrived_drone(pose, action.target_xyz):
                intention = action.intention
                flag_send_intention = True
                hold_start_time = current_time
                drone_state = DroneState.AT_WAYPOINT_HOLDING
                print(f"[STATE] ARRIVED at target at t={current_time:.1f}s; "
                      f"holding for {action.hold_duration}s with intention={action.intention}")

        elif drone_state == DroneState.AT_WAYPOINT_HOLDING:
            action = mission.current_action
            send_commands = np.array([
                action.target_xyz[0], action.target_xyz[1], action.target_xyz[2], action.target_yaw,
            ], dtype=np.float64)

            if hold_completed_drone(hold_start_time, current_time, duration=action.hold_duration):
                actual_hold = current_time - hold_start_time
                print(f"[STATE] HOLD COMPLETE at t={current_time:.1f}s: {action.description}, "
                      f"actual_duration={actual_hold:.2f}s")

                if action.intention == DRONE_INTENTION_PICKUP_SMALL:
                    mission.cargo_small_count += 1
                elif action.intention == DRONE_INTENTION_DROPOFF:
                    if mission.cargo_small_count > 0:
                        mission.cargo_small_count -= 1

                mission.advance()
                hold_start_time = None
                drone_state = DroneState.ACTION_COMPLETE
            else:
                if not has_arrived_drone(pose, action.target_xyz):
                    print(f"[STATE] Drifted out of tolerance during hold; resetting")
                    hold_start_time = None
                    current_trajectory = make_trajectory(
                        start_xyz=pose[:3].copy(), end_xyz=action.target_xyz,
                        current_time=current_time, duration=5.0,
                    )
                    drone_state = DroneState.APPROACHING_WAYPOINT

        elif drone_state == DroneState.ACTION_COMPLETE:
            if mission.is_complete:
                drone_state = DroneState.MISSION_COMPLETE
                print(f"[STATE] MISSION COMPLETE at t={current_time:.1f}s")
            else:
                action = mission.current_action
                is_window = action.target_xyz[2] >= WINDOW_Z_THRESHOLD_M
                flight_time = WINDOW_FLIGHT_TIME_SEC if is_window else DEFAULT_FLIGHT_TIME_SEC
                if action.flight_time_override is not None:
                    flight_time = action.flight_time_override
                via = list(action.via_points) if action.via_points is not None else None

                current_trajectory = make_trajectory(
                    start_xyz=pose[:3].copy(), end_xyz=action.target_xyz,
                    current_time=current_time, duration=flight_time, via_points=via,
                )
                intention = DRONE_INTENTION_NOTHING
                flag_send_intention = True
                drone_state = DroneState.APPROACHING_WAYPOINT
                print(f"[STATE] ACTION_COMPLETE -> APPROACHING_WAYPOINT "
                      f"(target=({action.target_xyz[0]:.2f}, "
                      f"{action.target_xyz[1]:.2f}, {action.target_xyz[2]:.2f}), "
                      f"flight_time={flight_time:.1f}s)")

        elif drone_state == DroneState.MISSION_COMPLETE:
            final_action = mission.actions[-1] if mission.actions else None
            if final_action is not None:
                send_commands = np.array([
                    final_action.target_xyz[0], final_action.target_xyz[1],
                    final_action.target_xyz[2], final_action.target_yaw,
                ], dtype=np.float64)
            else:
                send_commands = hover_command.copy()
            intention = DRONE_INTENTION_NOTHING

        if not client_drone.connected:
            client_drone.checkConnection(timeout=timeout)

        if client_drone.connected:
            if flag_send_intention:
                client_drone.send(np.array(intention, dtype=np.float64))
                flag_send_intention = False

        if not dataStream.connected:
            dataStream.checkConnection(timeout=timeout)

        if dataStream.connected:
            recvFlag, bytesReceived = dataStream.receive(iterations=2, timeout=timeout)

            if not recvFlag:
                receiveCounter += 1
                if receiveCounter > 1000:
                    print('Client stopped sending data over.')
            else:
                receiveCounter = 0
                receivedData = dataStream.receiveBuffer[0]
                pose = get_drone_pose_safe(receivedData, fallback_pose=send_commands)

            if useCameras and counter % CameraCounts == 0:
                frameLeft = camLeft.read()
                frameRight = camRight.read()
                frameBack = camBack.read()
                frameDown = camDown.read()
                realsense.read_RGB()
                realsense.read_depth()
                if frameLeft or frameRight or frameBack or frameDown:
                    cv2.imshow("Left Drone Image", camLeft.imageData)
                    cv2.imshow("Right Drone Image", camRight.imageData)
                    cv2.imshow("Back Drone Image", camBack.imageData)
                    cv2.imshow("Downwards Drone Image", camDown.imageData)
                    cv2.imshow("Front RGB Drone Image", realsense.imageBufferRGB)
                    cv2.imshow("Front Depth Drone Image", realsense.imageBufferDepthPX)
                    cv2.waitKey(1)

            counter += 1

            sentFlag = dataStream.send(send_commands)
            if sentFlag == -1:
                print('Server application not receiving.')
                break

        timer.sleep()

except KeyboardInterrupt:
    print("\nExiting due to keyboard interrupt.")
except Exception as e:
    import traceback
    print("\n" + "="*60)
    print("FATAL ERROR in main loop:")
    print(traceback.format_exc())
    print("="*60)
finally:
    print("\n--- Cleanup ---")
    try:
        if useCameras:
            realsense.terminate()
            camLeft.terminate()
            camBack.terminate()
            camRight.terminate()
            camDown.terminate()
    except Exception as cleanup_err:
        print(f"Camera cleanup error: {cleanup_err}")
    try:
        dataStream.terminate()
    except Exception as cleanup_err:
        print(f"dataStream cleanup error: {cleanup_err}")
    print("\nDrone script ending. Press Enter to close...")
    try:
        input()
    except Exception:
        pass