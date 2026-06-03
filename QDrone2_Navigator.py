# =============================================================================
# Example Drone Navigator File
# -----------------------------------------------------------------------------
# This file is intended as a reference example. Competitors or developers can:
#   - Use it as a baseline for their own drone navigation systems
#   - Modify and extend it for improved performance or new features
#   - Integrate it into larger autonomous flight pipelines
#
# This script demonstrates a baseline implementation of a drone navigation and
# command system using waypoint interpolation, communication streams, and
# optional onboard camera feeds.
#
# The implementation includes:
#   - Fixed hover mode for simple position hold
#   - Time-parameterized waypoint trajectory execution
#   - TCP/IP communication with the simulation/client
#   - Optional multi-camera streaming
#   - Keyboard-based mode switching
# =============================================================================


# region: Python level imports

# Numerical and computer vision libraries
import numpy as np
import cv2

# File path handling
from pathlib import Path

# Quanser-specific communication timeout handling
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout

# Quanser platform utilities for streaming, timing, and cameras
from pal.utilities.stream import BasicStream
from pal.utilities.timing import QTimer
from pal.utilities.vision import Camera2D, Camera3D
# endregion

# AICA roadmap for node lookups (currently unused by drone but kept for parity)
# Drone uses 3D coordinates directly, not road nodes.

# =============================================================================
# State Machine Definitions
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List


class DroneState(Enum):
    """States in the QDrone2 mission state machine."""
    IDLE = auto()                  # Just started, hovering at spawn
    APPROACHING_WAYPOINT = auto()  # Flying toward a target location
    AT_WAYPOINT_HOLDING = auto()   # Arrived at target, holding for required duration
    ACTION_COMPLETE = auto()       # Just finished an action, ready to pick next one
    MISSION_COMPLETE = auto()      # All planned actions finished; hold position


# Action intentions per Scenario Rules (drone-specific)
# https://utadnclab.github.io/AICA-Competition-Documentation-2026/01_Core_Guides/Virtual_Stage_Detailed_Scenario.html#scenario-rules
DRONE_INTENTION_NOTHING = 0
DRONE_INTENTION_PICKUP_SMALL = 1
DRONE_INTENTION_DROPOFF = 2
DRONE_INTENTION_TRANSFER_FROM_CAR = 3
DRONE_INTENTION_TRANSFER_TO_CAR = 4

# Scenario constants from the documentation
DRONE_HORIZONTAL_TOLERANCE_M = 2.0    # Horizontal distance tolerance for arrival
# Vertical tolerance is asymmetric: drone can be up to 4m above target (for
# ground pads) but only 2m below (for windows, to avoid flying into a building).
DRONE_VERTICAL_TOLERANCE_BELOW_M = 2.0
DRONE_VERTICAL_TOLERANCE_ABOVE_M = 4.0
DRONE_HOLD_DURATION_SEC = 4.5         # Required hold time (mentor-tuned for drift margin)
DRONE_CRUISE_ALTITUDE_M = 3.0         # Default flying altitude for ground-level targets

# Z-buffer applied to window targets so we command slightly inside the valid
# vertical zone rather than at its edge. game.py accepts z in [window_z, window_z + 4],
# so adding 0.35 keeps us safely inside if the drone drifts down during hold.
WINDOW_Z_BUFFER_M = 0.35

# Z-threshold above which a target is treated as a window delivery (vs ground).
# Ground delivery pads are at z=0.05; cruise is z=3.0; windows are z=4.85+.
WINDOW_Z_THRESHOLD_M = 4.0


@dataclass
class DroneMissionAction:
    """A single action in the drone's mission plan."""
    target_xyz: np.ndarray         # 3D position to fly to (numpy array of shape (3,))
    target_yaw: float = 0.0        # Desired yaw at the target (radians)
    intention: int = DRONE_INTENTION_NOTHING
    description: str = ""
    hold_duration: float = DRONE_HOLD_DURATION_SEC


@dataclass
class DroneMissionState:
    """Tracks the drone's overall mission progress."""
    actions: List[DroneMissionAction] = field(default_factory=list)
    current_action_idx: int = 0
    cargo_small_count: int = 0     # Drone carries 1 small at a time

    @property
    def current_action(self) -> Optional[DroneMissionAction]:
        if self.current_action_idx < len(self.actions):
            return self.actions[self.current_action_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_action_idx >= len(self.actions)

    def advance(self):
        """Mark current action complete and move to next."""
        self.current_action_idx += 1


# =============================================================================
# State Machine Helper Functions
# =============================================================================

def has_arrived_drone(current_pose: np.ndarray, target_xyz: np.ndarray,
                      horizontal_tol: float = DRONE_HORIZONTAL_TOLERANCE_M,
                      vertical_tol_below: float = DRONE_VERTICAL_TOLERANCE_BELOW_M,
                      vertical_tol_above: float = DRONE_VERTICAL_TOLERANCE_ABOVE_M) -> bool:
    """
    Returns True if the drone is within tolerance of the target.

    Horizontal: within `horizontal_tol` meters in x/y.
    Vertical: drone's z is between (target_z - vertical_tol_below) and
              (target_z + vertical_tol_above).

    Asymmetric vertical tolerance handles both ground pickups (drone hovers
    above) and window deliveries (drone hovers near target z).

    current_pose is expected to be (x, y, z, yaw) from telemetry.
    """
    horizontal_dist = np.linalg.norm(current_pose[:2] - target_xyz[:2])
    if horizontal_dist > horizontal_tol:
        return False

    vertical_offset = current_pose[2] - target_xyz[2]
    if not (-vertical_tol_below <= vertical_offset <= vertical_tol_above):
        return False

    return True


def hold_completed_drone(hold_start_time: Optional[float], current_time: float,
                         duration: float = DRONE_HOLD_DURATION_SEC) -> bool:
    """
    Returns True if the position-hold has been maintained for `duration` seconds.
    """
    if hold_start_time is None:
        return False
    return (current_time - hold_start_time) >= duration


def build_drone_mission_simple() -> DroneMissionState:
    """
    Stub mission: fly from spawn to central pickup, pick up a small package,
    fly to delivery 1's window, drop it off (capturing the floor-4 window bonus).

    Coordinates from the competition Pickup and Delivery Table:
      Central pickup: [-2.50305, 29.6703, 0.05] (drone approaches at cruise altitude)
      Delivery 1 window: [15.1739, -18.04655, 9.65]

    This is a unit test for the drone state machine. The coordinated multi-vehicle
    strategy will come from the planner in Phase 3.
    """
    pickup_xyz = np.array([-2.50305, 29.6703, DRONE_CRUISE_ALTITUDE_M])
    delivery1_window_xyz = apply_window_z_buffer(np.array([15.1739, -18.04655, 9.65]))

    mission = DroneMissionState()
    mission.actions = [
        DroneMissionAction(
            target_xyz=pickup_xyz,
            intention=DRONE_INTENTION_PICKUP_SMALL,
            description="Pickup small package at central pickup"
        ),
        DroneMissionAction(
            target_xyz=delivery1_window_xyz,
            intention=DRONE_INTENTION_DROPOFF,
            description="Drop off at Delivery 1 window (floor 4)"
        ),
    ]
    return mission

# =============================================================================
# Trajectory Generation
# =============================================================================

# Fixed flight times (mentor-tuned for the AICA scenario distances).
DEFAULT_FLIGHT_TIME_SEC = 12.0      # Generic flights (~30m at ~2.5 m/s)
WINDOW_FLIGHT_TIME_SEC = 18.0       # Pickup ↔ window (~50m)
D2_WINDOW_FLIGHT_TIME_SEC = 28.0    # Delivery 2 (multi-waypoint, longer for building avoidance)


@dataclass
class DroneTrajectory:
    """
    Linear trajectory through one or more waypoints over a fixed duration.

    Supports both:
      - Single-segment: start_xyz -> end_xyz (direct flight)
      - Multi-segment: start_xyz -> via_points... -> end_xyz (building avoidance)

    Time is divided equally among segments.
    """
    waypoints: List[np.ndarray]
    start_time: float
    duration: float

    def waypoint_at(self, current_time: float) -> np.ndarray:
        """Return the interpolated waypoint at the given simulation time."""
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
    """
    Build a linear trajectory from start to end over a fixed duration.
    Optional via_points support multi-segment paths (e.g., building avoidance).
    """
    waypoints = [start_xyz.copy()]
    if via_points:
        waypoints.extend(np.array(p, dtype=np.float64) for p in via_points)
    waypoints.append(end_xyz.copy())

    return DroneTrajectory(
        waypoints=waypoints,
        start_time=current_time,
        duration=duration,
    )


def apply_window_z_buffer(target_xyz: np.ndarray) -> np.ndarray:
    """
    For window targets (z >= WINDOW_Z_THRESHOLD_M), apply a small upward buffer.
    Keeps commanded altitude safely inside the valid vertical zone.
    """
    target = target_xyz.copy()
    if target[2] >= WINDOW_Z_THRESHOLD_M:
        target[2] += WINDOW_Z_BUFFER_M
    return target


def get_drone_pose_safe(received_data: np.ndarray, fallback_pose: np.ndarray) -> np.ndarray:
    """
    Extract pose from telemetry, falling back to a sensible default if
    telemetry returns all zeros (first few frames before simulator publishes data).
    """
    pose_from_telemetry = np.asarray(received_data[16:20], dtype=np.float64)
    if np.allclose(pose_from_telemetry, 0.0):
        return fallback_pose.astype(np.float64)
    return pose_from_telemetry

# =============================================================================
# Utility Functions
# =============================================================================

def read_initial_positions(filepath: Path) -> np.ndarray:
    """
    Reads initial spawn positions from a text file.

    The file is expected to contain comma-separated numeric values.
    Lines starting with '#' are treated as comments.

    Returns:
        np.ndarray: Drone initial pose values [x, y, z, yaw].
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Spawn file not found: {filepath}")

    values: list[float] = []

    with filepath.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            try:
                parts = [float(value.strip()) for value in line.split(",")]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in {filepath} on line {line_number}: {raw_line.strip()}"
                ) from exc

            values.extend(parts)

    # Ensure minimum required values exist
    if len(values) < 8:
        raise ValueError(
            f"{filepath} must contain at least 8 numeric values, but found {len(values)}."
        )

    # Drone pose is stored in the second group of 4 values
    return np.array(values[4:8], dtype=np.float64)


def load_plan_file(plan_path: Path):
    """
    Loads precomputed drone waypoint trajectory data from a NumPy archive.

    Returns:
        tuple:
            qdrone2_wp1 (ndarray): Waypoint matrix
            qdrone2_t1  (ndarray): Time vector corresponding to waypoints
    """
    plan_data = np.load(plan_path, allow_pickle=True)

    qdrone2_wp1 = np.asarray(plan_data["qdrone2_wp1"])
    qdrone2_t1 = np.asarray(plan_data["qdrone2_t1"]).flatten()

    return qdrone2_wp1, qdrone2_t1


def interpolate_waypoint(
    t_query: float, t_vec: np.ndarray, wp_mat: np.ndarray
) -> np.ndarray:
    """
    Linearly interpolates the commanded waypoint at a requested time.

    Args:
        t_query: Query time
        t_vec: Monotonic time vector
        wp_mat: Waypoint matrix aligned with t_vec

    Returns:
        np.ndarray: Interpolated waypoint command
    """
    if t_query <= t_vec[0]:
        return wp_mat[0].astype(np.float64)

    if t_query >= t_vec[-1]:
        return wp_mat[-1].astype(np.float64)

    idx_right = np.searchsorted(t_vec, t_query, side="right")
    idx_left = idx_right - 1

    t0 = t_vec[idx_left]
    t1 = t_vec[idx_right]

    wp0 = wp_mat[idx_left]
    wp1 = wp_mat[idx_right]

    alpha = (t_query - t0) / (t1 - t0)

    return ((1.0 - alpha) * wp0 + alpha * wp1).astype(np.float64)


# =============================================================================
# Main Execution Configuration
# =============================================================================

# region: Experiment constants
simulationTime = 10000   # Total simulation duration (seconds)
frequency = 200          # Control loop frequency (Hz)
frameRate = 30           # Camera frame rate
CameraCounts = int(round(frequency / frameRate))
useCameras = False       # Enable/disable camera streams
# endregion


# =============================================================================
# State Variables and Initialization
# =============================================================================

counter = 0
receiveCounter = 0
receivedData = np.zeros(16)

# Load initial spawn pose and planned trajectory
initial_position = read_initial_positions(Path("spawn_locations.txt"))
plan_path = Path(r"tools\QDrone2_PathPlanning\qdrone2_plans.npz")
qdrone2_wp1, qdrone2_t1 = load_plan_file(plan_path)


# =============================================================================
# Camera Initialization (Optional)
# =============================================================================
# Cameras simulate multiple onboard viewpoints around the drone
if useCameras:
    realsense = Camera3D(
        deviceId="0@tcpip://localhost:18986",
        mode='RGB&DEPTH',
        frameWidthRGB=640,
        frameHeightRGB=480,
        frameRateRGB=frameRate,
        frameWidthDepth=640,
        frameHeightDepth=480,
        frameRateDepth=frameRate,
        readMode=0
    )

    camRight = Camera2D(
        cameraId="0@tcpip://localhost:18982",
        frameWidth=640,
        frameHeight=480,
        frameRate=frameRate
    )

    camBack = Camera2D(
        cameraId="1@tcpip://localhost:18983",
        frameWidth=640,
        frameHeight=480,
        frameRate=frameRate
    )

    camLeft = Camera2D(
        cameraId="2@tcpip://localhost:18984",
        frameWidth=640,
        frameHeight=480,
        frameRate=frameRate
    )

    camDown = Camera2D(
        cameraId="3@tcpip://localhost:18985",
        frameWidth=640,
        frameHeight=480,
        frameRate=frameRate
    )


# =============================================================================
# Communication Streams Setup
# =============================================================================

# Main data stream used to send position commands and receive telemetry
dataStream = BasicStream(
    'tcpip://localhost:18373',
    agent='C',
    sendBufferSize=1460,
    receiveBuffer=np.zeros((1, 20), dtype=np.float64),
    recvBufferSize=1460,
    nonBlocking=False
)

# Client stream used to communicate intention/mode information
client_drone = BasicStream(
    'tcpip://localhost:19001',
    agent='C',
    sendBufferSize=8,
    receiveBuffer=np.zeros((1, 3), dtype=np.float64),
    recvBufferSize=24,
    nonBlocking=False
)

timeout = Timeout(seconds=0, nanoseconds=10)
prev_con = False
prev_game_con = False


# =============================================================================
# Control Loop Initialization
# =============================================================================

timer = QTimer(frequency, simulationTime)

# Flags and command variables
flag_send_intention = True
intention = DRONE_INTENTION_NOTHING

# Default hover command: at spawn location, 3 m above ground
hover_command = initial_position + np.array([0.0, 0.0, DRONE_CRUISE_ALTITUDE_M, 0.0], dtype=np.float64)
send_commands = hover_command.copy()

# Build the mission (hardcoded for now; will be replaced by planner later)
mission = build_drone_mission_simple()
print(f"Drone mission loaded with {len(mission.actions)} actions:")
for i, action in enumerate(mission.actions):
    print(f"  [{i}] {action.description}")

# State machine state
drone_state = DroneState.IDLE
hold_start_time: Optional[float] = None    # Timestamp when we entered hold state
current_trajectory: Optional[DroneTrajectory] = None  # Active flight trajectory


# Drone pose: (x, y, z, yaw) from telemetry — initialized to spawn, will be
# overwritten by dataStream telemetry once connected.
pose = np.array([initial_position[0], initial_position[1], 0.0, initial_position[3]])


# =============================================================================
# Main Control Loop
# =============================================================================

try:
    while timer.check():
        current_time = timer.get_current_time()

# ---------------- State Machine ----------------
# Handle current state; may transition to next state.

        if drone_state == DroneState.IDLE:
            # First-time setup: build trajectory from current pose to first target
            if not mission.is_complete:
                action = mission.current_action

                is_window = action.target_xyz[2] >= WINDOW_Z_THRESHOLD_M
                flight_time = WINDOW_FLIGHT_TIME_SEC if is_window else DEFAULT_FLIGHT_TIME_SEC

                # If we're starting near ground level, add a takeoff via-point
                # so the drone climbs straight up before translating horizontally.
                start = pose[:3].copy()
                via = None
                if start[2] < 1.5:  # near ground
                    takeoff_xyz = np.array([start[0], start[1], DRONE_CRUISE_ALTITUDE_M])
                    via = [takeoff_xyz]
                    flight_time += 4.0  # add takeoff time
                    print(f"[TRAJ] Adding takeoff via-point at {takeoff_xyz}")

                current_trajectory = make_trajectory(
                    start_xyz=start,
                    end_xyz=action.target_xyz,
                    current_time=current_time,
                    duration=flight_time,
                    via_points=via,
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
            # Compute current waypoint from active trajectory (smooth interpolation)
            action = mission.current_action

            if current_trajectory is not None:
                waypoint = current_trajectory.waypoint_at(current_time)
            else:
                waypoint = action.target_xyz

            send_commands = np.array([
                waypoint[0],
                waypoint[1],
                waypoint[2],
                action.target_yaw,
            ], dtype=np.float64)

            if has_arrived_drone(pose, action.target_xyz):
                # Arrived; set the action's intention and start hold timer
                intention = action.intention
                flag_send_intention = True
                hold_start_time = current_time
                drone_state = DroneState.AT_WAYPOINT_HOLDING
                print(f"[STATE] ARRIVED at target; holding for "
                      f"{action.hold_duration}s with intention={action.intention}")

        elif drone_state == DroneState.AT_WAYPOINT_HOLDING:
            # Keep commanding the target position (the flight controller maintains hover)
            action = mission.current_action
            send_commands = np.array([
                action.target_xyz[0],
                action.target_xyz[1],
                action.target_xyz[2],
                action.target_yaw,
            ], dtype=np.float64)

            if hold_completed_drone(hold_start_time, current_time,
                                    duration=action.hold_duration):
                actual_hold = current_time - hold_start_time
                print(f"[STATE] HOLD COMPLETE: {action.description}, "
                      f"actual_duration={actual_hold:.2f}s")

                # Track cargo changes
                if action.intention == DRONE_INTENTION_PICKUP_SMALL:
                    mission.cargo_small_count += 1
                elif action.intention == DRONE_INTENTION_DROPOFF:
                    if mission.cargo_small_count > 0:
                        mission.cargo_small_count -= 1

                mission.advance()
                hold_start_time = None
                drone_state = DroneState.ACTION_COMPLETE
            else:
                # Drift check: if the drone has wandered out of tolerance, rebuild
                # a short recovery trajectory back to the target.
                if not has_arrived_drone(pose, action.target_xyz):
                    print(f"[STATE] Drifted out of tolerance during hold; resetting")
                    hold_start_time = None
                    current_trajectory = make_trajectory(
                        start_xyz=pose[:3].copy(),
                        end_xyz=action.target_xyz,
                        current_time=current_time,
                        duration=5.0,
                    )
                    drone_state = DroneState.APPROACHING_WAYPOINT

        elif drone_state == DroneState.ACTION_COMPLETE:
            # Decide whether mission is done or pick the next action
            if mission.is_complete:
                drone_state = DroneState.MISSION_COMPLETE
                print(f"[STATE] MISSION COMPLETE at t={current_time:.1f}s")
            else:
                # Build trajectory for next target
                action = mission.current_action

                is_window = action.target_xyz[2] >= WINDOW_Z_THRESHOLD_M
                flight_time = WINDOW_FLIGHT_TIME_SEC if is_window else DEFAULT_FLIGHT_TIME_SEC

                current_trajectory = make_trajectory(
                    start_xyz=pose[:3].copy(),
                    end_xyz=action.target_xyz,
                    current_time=current_time,
                    duration=flight_time,
                )
                intention = DRONE_INTENTION_NOTHING
                flag_send_intention = True
                drone_state = DroneState.APPROACHING_WAYPOINT
                print(f"[STATE] ACTION_COMPLETE -> APPROACHING_WAYPOINT "
                      f"(target=({action.target_xyz[0]:.2f}, "
                      f"{action.target_xyz[1]:.2f}, {action.target_xyz[2]:.2f}), "
                      f"flight_time={flight_time:.1f}s)")

        elif drone_state == DroneState.MISSION_COMPLETE:
            # Mission done; hold at the final action's location indefinitely
            # (Do NOT fly back to spawn; that wastes time and looks like "falling.")
            final_action = mission.actions[-1] if mission.actions else None
            if final_action is not None:
                send_commands = np.array([
                    final_action.target_xyz[0],
                    final_action.target_xyz[1],
                    final_action.target_xyz[2],
                    final_action.target_yaw,
                ], dtype=np.float64)
            else:
                send_commands = hover_command.copy()
            intention = DRONE_INTENTION_NOTHING

        # ---------------- Client Communication ----------------
        # Ensure connection and send intention when needed
        if not client_drone.connected:
            client_drone.checkConnection(timeout=timeout)

        if client_drone.connected:
            if flag_send_intention:
                client_drone.send(np.array(intention, dtype=np.float64))
                flag_send_intention = False

        # ---------------- Server Communication ----------------
        # Ensure connection to the drone/server data stream
        if not dataStream.connected:
            dataStream.checkConnection(timeout=timeout)

        # Execute the main telemetry and command exchange
        if dataStream.connected:
            recvFlag, bytesReceived = dataStream.receive(iterations=2, timeout=timeout)

            if not recvFlag:
                receiveCounter += 1
                if receiveCounter > 1000:
                    print('Client stopped sending data over.')
            else:
                receiveCounter = 0
                receivedData = dataStream.receiveBuffer[0]
                # Telemetry structure:
                # [0]     Stream connection flag
                # [1:4]   IMU gyroscope data (rad/s)
                # [4:7]   IMU accelerometer data (m/s^2)
                # [7:10]  Estimated angular position (rad)
                # [10:13] Estimated angular rates (rad/s)
                # [13:16] Estimated angular acceleration
                # [16:20] Pose x, y, z, yaw (m, m, m, rad)

                # Extract pose with fallback for early-frame zero telemetry
                pose = get_drone_pose_safe(receivedData, fallback_pose=send_commands)       
            # ---------------- Camera Handling ----------------
            if useCameras and counter % CameraCounts == 0:
                frameLeft = camLeft.read()
                frameRight = camRight.read()
                frameBack = camBack.read()
                frameDown = camDown.read()
                realsense.read_RGB()
                realsense.read_depth()

                if frameLeft or frameRight or frameBack or frameDown:
                    imageLeft = camLeft.imageData
                    imageRight = camRight.imageData
                    imageBack = camBack.imageData
                    imageDown = camDown.imageData
                    imageRGB = realsense.imageBufferRGB
                    imageDepth = realsense.imageBufferDepthPX
                    # NOTE:
                    # imageDepth contains values mapped approximately from
                    # 0-255 over a depth range of about 0-9.44 meters.

                    cv2.imshow("Left Drone Image", imageLeft)
                    cv2.imshow("Right Drone Image", imageRight)
                    cv2.imshow("Back Drone Image", imageBack)
                    cv2.imshow("Downwards Drone Image", imageDown)
                    cv2.imshow("Front RGB Drone Image", imageRGB)
                    cv2.imshow("Front Depth Drone Image", imageDepth)

                    cv2.waitKey(1)

            counter += 1

            # ---------------- Command Transmission ----------------
            # Send the current position/yaw command to the drone
            sentFlag = dataStream.send(send_commands)
            if sentFlag == -1:
                print('Server application not receiving.')
                break

        # Maintain loop timing
        timer.sleep()

except KeyboardInterrupt:
    print("\nExiting due to keyboard interrupt.")
except Exception as e:
    import traceback
    print("\n" + "="*60)
    print("FATAL ERROR in main loop:")
    print(traceback.format_exc())
    print("="*60)
    input("Press Enter to close...")


# =============================================================================
# Cleanup
# =============================================================================

if useCameras:
    realsense.terminate()
    camLeft.terminate()
    camBack.terminate()
    camRight.terminate()
    camDown.terminate()

dataStream.terminate()