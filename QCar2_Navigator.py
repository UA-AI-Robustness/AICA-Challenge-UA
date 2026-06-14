# =============================================================================
# QCar2 Navigator — 5675 config + pose initialization race-condition fix
# -----------------------------------------------------------------------------
# Single change vs the GitHub 5675 version:
#   - Initial `pose` is set from spawn_locations.txt instead of zeros.
#     The original `pose = np.zeros(3)` caused a race where Frame 1 ran
#     Stanley and the safety check with pose=(0,0,0), distance to target
#     ~30m, so vel_cmd=0.15 was commanded. By the time real telemetry
#     arrived, the car had already received forward-motion commands and
#     could overshoot the pickup pad (especially with the spawn 1.97m
#     south of pickup).
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
from pal.utilities.vision import Camera2D
from hal.products.mats_aica import SDCSRoadMap

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List


class CarState(Enum):
    IDLE = auto()
    APPROACHING_NODE = auto()
    AT_NODE_HOLDING = auto()
    ACTION_COMPLETE = auto()
    MISSION_COMPLETE = auto()


INTENTION_NOTHING = 0
INTENTION_PICKUP_SMALL = 1
INTENTION_PICKUP_LARGE = 2
INTENTION_DROPOFF = 3
INTENTION_TRANSFER_FROM_DRONE = 4
INTENTION_TRANSFER_TO_DRONE = 5

ARRIVAL_TOLERANCE_M = 2
HOLD_DURATION_SEC = 3.05
ROADMAP_SCALE_FACTOR = 10.0

CAR_VEL_CMD = 0.15


@dataclass
class MissionAction:
    target_node: int
    intention: int
    description: str = ""
    hold_duration: float = HOLD_DURATION_SEC


@dataclass
class CarMissionState:
    actions: List[MissionAction] = field(default_factory=list)
    current_action_idx: int = 0
    cargo_small_count: int = 0
    cargo_large_count: int = 0

    @property
    def current_action(self) -> Optional[MissionAction]:
        if self.current_action_idx < len(self.actions):
            return self.actions[self.current_action_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_action_idx >= len(self.actions)

    def advance(self):
        self.current_action_idx += 1


def get_node_location_xy(roadmap, node_idx: int) -> np.ndarray:
    node_pose = ROADMAP_SCALE_FACTOR * roadmap.nodes[node_idx].pose.flatten()
    return np.array([node_pose[0], node_pose[1]])


def has_arrived(current_pose: np.ndarray, target_xy: np.ndarray,
                tolerance: float = ARRIVAL_TOLERANCE_M) -> bool:
    horizontal_dist = np.linalg.norm(current_pose[:2] - target_xy)
    return horizontal_dist <= tolerance


def hold_completed(hold_start_time: Optional[float], current_time: float,
                   duration: float = HOLD_DURATION_SEC) -> bool:
    if hold_start_time is None:
        return False
    return (current_time - hold_start_time) >= duration


def build_mission_deliveries_4_and_5() -> CarMissionState:
    mission = CarMissionState()
    mission.actions = [
        MissionAction(target_node=24, intention=INTENTION_PICKUP_SMALL,
                      description="Pickup small #1 at central pickup"),
        MissionAction(target_node=22, intention=INTENTION_DROPOFF,
                      description="Drop off small at Delivery 4"),
        MissionAction(target_node=24, intention=INTENTION_PICKUP_LARGE,
                      description="Pickup large at central pickup"),
        MissionAction(target_node=10, intention=INTENTION_DROPOFF,
                      description="Drop off large at Delivery 5"),
    ]
    return mission


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
    return np.array(values[0:4], dtype=np.float64)


def load_plan_file(plan_path: Path):
    qcar2_pathposes = np.load(plan_path, allow_pickle=True)
    return qcar2_pathposes


def stanley_controller(pose, vel_cmd, path_pose):
    epsilon = 1e-3
    k = 2.5
    delta_max = 0.6

    pose = np.asarray(pose, dtype=float)
    path_pose = np.asarray(path_pose, dtype=float)

    x, y, yaw = pose[0], pose[1], pose[2]
    path_x = path_pose[:, 0]
    path_y = path_pose[:, 1]
    path_yaw = path_pose[:, 2]

    dx = path_x - x
    dy = path_y - y
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    x_ref = path_x[target_idx]
    y_ref = path_y[target_idx]
    yaw_ref = path_yaw[target_idx]

    psi_e = (yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi

    dxl = x - x_ref
    dyl = y - y_ref
    e_ct = -np.sin(yaw_ref) * dxl + np.cos(yaw_ref) * dyl
    e_ct = -e_ct

    vel = vel_cmd * 13 / 0.2
    delta = psi_e + np.arctan2(k * e_ct, vel + epsilon)
    delta = np.clip(delta, -delta_max, delta_max)

    dist_to_end = np.linalg.norm(np.array([x, y]) - path_pose[-1, 0:2], ord=2)
    ang_to_end = abs(np.rad2deg(pose[2] - path_pose[-1, 2]))

    if dist_to_end < 10.0 and ang_to_end < 60.0:
        vel_cmd = dist_to_end / 10.0 * vel_cmd
    elif dist_to_end < 1.0 and ang_to_end < 60.0:
        vel_cmd = 0.0

    if abs(delta) >= 0.90 * delta_max:
        vel_cmd = min(vel_cmd, 0.04)
    elif abs(delta) >= 0.80 * delta_max:
        vel_cmd = min(vel_cmd, 0.06)
    elif abs(delta) >= 0.70 * delta_max:
        vel_cmd = min(vel_cmd, 0.08)
    elif abs(delta) >= 0.60 * delta_max:
        vel_cmd = min(vel_cmd, 0.10)
    elif abs(delta) >= 0.50 * delta_max:
        vel_cmd = min(vel_cmd, 0.12)

    return vel_cmd, delta


simulationTime = 1200
frequency = 200
frameRate = 30
CameraCounts = int(round(frequency / frameRate))
useCameras = False

counter = 0
receiveCounter = 0
receivedData = np.zeros(10)
receiveGameCounter = 0

initial_position = read_initial_positions(Path("spawn_locations.txt"))
pathposes_path = Path(r"tools\QCar2_PathPlanning\qcar2_pathposes.npy")
pathposes = load_plan_file(pathposes_path)

# RACE-CONDITION FIX: initialize pose from spawn coords (with yaw in radians)
# so that Frame 1 of the main loop runs Stanley and the safety check with
# correct position data. Previously pose = np.zeros(3) caused Frame 1 to
# compute distance-to-target ~30m, missing the within-tolerance safety stop
# and commanding forward motion before real telemetry arrived.
pose = np.array([
    initial_position[0],
    initial_position[1],
    np.deg2rad(initial_position[3]),
], dtype=np.float64)
print(f"[INIT] Car pose initialized from spawn: "
      f"({pose[0]:.2f}, {pose[1]:.2f}, yaw={np.rad2deg(pose[2]):.1f}°)")

if useCameras:
    camRight = Camera2D(cameraId="0@tcpip://localhost:18961", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camBack  = Camera2D(cameraId="1@tcpip://localhost:18962", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camLeft  = Camera2D(cameraId="3@tcpip://localhost:18964", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camFront = Camera2D(cameraId="2@tcpip://localhost:18963", frameWidth=640, frameHeight=480, frameRate=frameRate)


dataStream = BasicStream(
    'tcpip://localhost:18375',
    agent='C',
    sendBufferSize=1460,
    receiveBuffer=np.zeros((1,10), dtype=np.float64),
    recvBufferSize=1460,
    nonBlocking=False
)

client_car = BasicStream(
    'tcpip://localhost:19000',
    agent='C',
    sendBufferSize=8,
    receiveBuffer=np.zeros((1,3), dtype=np.float64),
    recvBufferSize=24,
    nonBlocking=False
)

timeout = Timeout(seconds=0, nanoseconds=10)


timer = QTimer(frequency, simulationTime)

flag_send_intention = True
intention = INTENTION_NOTHING
send_commands = np.array([0., 0.], dtype=np.float64)

roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

mission = build_mission_deliveries_4_and_5()
print(f"Car mission loaded with {len(mission.actions)} actions:")
for i, action in enumerate(mission.actions):
    print(f"  [{i}] {action.description}")
print(f"Settings: vel_cmd={CAR_VEL_CMD}, hold={HOLD_DURATION_SEC}s")

car_state = CarState.IDLE
hold_start_time: Optional[float] = None
current_start_node = 8

vel_cmd = 0.0
path_pose = pathposes[8, 8]


try:
    while timer.check():
        current_time = timer.get_current_time()

        if car_state == CarState.IDLE:
            if not mission.is_complete:
                action = mission.current_action
                target_node = action.target_node
                if target_node == current_start_node:
                    path_pose = pathposes[current_start_node, current_start_node]
                else:
                    path_pose = pathposes[current_start_node, target_node]
                vel_cmd = CAR_VEL_CMD
                intention = INTENTION_NOTHING
                flag_send_intention = True
                car_state = CarState.APPROACHING_NODE
                print(f"[STATE] IDLE -> APPROACHING_NODE (target={target_node})")
            else:
                car_state = CarState.MISSION_COMPLETE

        elif car_state == CarState.APPROACHING_NODE:
            action = mission.current_action
            target_xy = get_node_location_xy(roadmap, action.target_node)

            if has_arrived(pose, target_xy):
                vel_cmd = 0.0
                intention = action.intention
                flag_send_intention = True
                hold_start_time = current_time
                car_state = CarState.AT_NODE_HOLDING
                print(f"[STATE] ARRIVED at node {action.target_node} at t={current_time:.1f}s; "
                      f"holding for {action.hold_duration}s with intention={action.intention}")
            else:
                vel_cmd = CAR_VEL_CMD
                intention = INTENTION_NOTHING

        elif car_state == CarState.AT_NODE_HOLDING:
            action = mission.current_action

            if hold_completed(hold_start_time, current_time, duration=action.hold_duration):
                actual_hold = current_time - hold_start_time
                print(f"[STATE] HOLD COMPLETE at t={current_time:.1f}s: {action.description}, "
                      f"actual_duration={actual_hold:.2f}s")

                if action.intention == INTENTION_PICKUP_SMALL:
                    mission.cargo_small_count += 1
                elif action.intention == INTENTION_PICKUP_LARGE:
                    mission.cargo_large_count += 1
                elif action.intention == INTENTION_DROPOFF:
                    if mission.cargo_large_count > 0:
                        mission.cargo_large_count -= 1
                    elif mission.cargo_small_count > 0:
                        mission.cargo_small_count -= 1

                current_start_node = action.target_node
                mission.advance()
                hold_start_time = None
                car_state = CarState.ACTION_COMPLETE

        elif car_state == CarState.ACTION_COMPLETE:
            if mission.is_complete:
                car_state = CarState.MISSION_COMPLETE
                print(f"[STATE] MISSION COMPLETE at t={current_time:.1f}s")
            else:
                action = mission.current_action
                target_node = action.target_node
                if target_node == current_start_node:
                    path_pose = pathposes[current_start_node, current_start_node]
                else:
                    path_pose = pathposes[current_start_node, target_node]
                vel_cmd = CAR_VEL_CMD
                intention = INTENTION_NOTHING
                flag_send_intention = True
                car_state = CarState.APPROACHING_NODE
                print(f"[STATE] ACTION_COMPLETE -> APPROACHING_NODE (target={target_node})")

        elif car_state == CarState.MISSION_COMPLETE:
            vel_cmd = 0.0
            intention = INTENTION_NOTHING
            path_pose = pathposes[current_start_node, current_start_node]

        if not client_car.connected:
            client_car.checkConnection(timeout=timeout)

        if client_car.connected:
            if flag_send_intention:
                client_car.send(np.array(intention, dtype=np.float64))
                flag_send_intention = False

            recvFlag, _ = client_car.receive(iterations=2, timeout=timeout)

            if not recvFlag:
                receiveGameCounter += 1
                if receiveGameCounter > 1000:
                    print('QCar stopped receiving GPS data.')
            else:
                receiveGameCounter = 0
                pose = client_car.receiveBuffer[0]

        vel_cmd, steering_cmd = stanley_controller(pose, vel_cmd, path_pose)

        if car_state in (CarState.APPROACHING_NODE, CarState.AT_NODE_HOLDING):
            action = mission.current_action
            if action is not None:
                target_xy = get_node_location_xy(roadmap, action.target_node)
                dist_to_target = np.linalg.norm(pose[:2] - target_xy)

                if dist_to_target <= ARRIVAL_TOLERANCE_M:
                    vel_cmd = 0.0
                    steering_cmd = 0.0
                elif car_state == CarState.AT_NODE_HOLDING:
                    print(f"[STATE] Drifted out of target zone (dist={dist_to_target:.2f}m); resetting hold")
                    hold_start_time = None
                    car_state = CarState.APPROACHING_NODE

        send_commands = np.array([vel_cmd, steering_cmd], dtype=np.float64)

        if not dataStream.connected:
            dataStream.checkConnection(timeout=timeout)

        if dataStream.connected:
            recvFlag, _ = dataStream.receive(iterations=2, timeout=timeout)

            if not recvFlag:
                receiveCounter += 1
                if receiveCounter > 10:
                    print('Client stopped sending data over.')
            else:
                receiveCounter = 0
                receivedData = dataStream.receiveBuffer[0]

            if useCameras and counter % CameraCounts == 0:
                frameLeft = camLeft.read()
                frameRight = camRight.read()
                frameBack = camBack.read()
                frameFront = camFront.read()
                if frameLeft or frameRight or frameBack or frameFront:
                    cv2.imshow("Left Car Image", camLeft.imageData)
                    cv2.imshow("Right Car Image", camRight.imageData)
                    cv2.imshow("Back Car Image", camBack.imageData)
                    cv2.imshow("Front Car Image", camFront.imageData)
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
            camLeft.terminate()
            camBack.terminate()
            camRight.terminate()
            camFront.terminate()
    except Exception as cleanup_err:
        print(f"Camera cleanup error: {cleanup_err}")
    try:
        dataStream.terminate()
    except Exception as cleanup_err:
        print(f"dataStream cleanup error: {cleanup_err}")
    try:
        client_car.terminate()
    except Exception as cleanup_err:
        print(f"client_car cleanup error: {cleanup_err}")
    print("\nCar script ending. Press Enter to close...")
    try:
        input()
    except Exception:
        pass