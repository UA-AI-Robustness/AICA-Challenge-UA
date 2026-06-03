# =============================================================================
# Example Car Navigator File
# -----------------------------------------------------------------------------
# This file is intended as a reference example. Competitors or developers can:
#   - Use it as a baseline for their own navigation systems
#   - Modify and extend it for improved performance or new features
#   - Integrate it into larger autonomous driving pipelines
#
# This script demonstrates a complete baseline implementation of a car
# navigation and control system using path planning, communication streams,
# and a Stanley controller for steering.
#
# The implementation includes:
#   - Path following via Stanley controller
#   - TCP/IP communication with simulation/client
#   - Optional camera streaming
#   - Keyboard-based mode switching
#
# The QCar2 Limits:
#   - The velocity command must be in the interval [-0.2, 0.2]
#   - The steering command must be in the interval  [-0.6, 0.6]
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
from pal.utilities.vision import Camera2D
# endregion 
# AICA roadmap for node lookups
from hal.products.mats_aica import SDCSRoadMap

# =============================================================================
# State Machine Definitions
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List


class CarState(Enum):
    """States in the QCar2 mission state machine."""
    IDLE = auto()                  # Just started, nothing to do yet
    APPROACHING_NODE = auto()      # Driving toward a target node
    AT_NODE_HOLDING = auto()       # Arrived at target, holding position for required duration
    ACTION_COMPLETE = auto()       # Just finished an action, ready to pick next one
    MISSION_COMPLETE = auto()      # All planned actions finished; stop


# Action intentions per Scenario Rules
# https://utadnclab.github.io/AICA-Competition-Documentation-2026/01_Core_Guides/Virtual_Stage_Detailed_Scenario.html#scenario-rules
INTENTION_NOTHING = 0
INTENTION_PICKUP_SMALL = 1
INTENTION_PICKUP_LARGE = 2
INTENTION_DROPOFF = 3
INTENTION_TRANSFER_FROM_DRONE = 4
INTENTION_TRANSFER_TO_DRONE = 5

# Scenario constants from the documentation
ARRIVAL_TOLERANCE_M = 2          # Horizontal distance tolerance for arrival
HOLD_DURATION_SEC = 3.1            # Required hold time for any action
ROADMAP_SCALE_FACTOR = 10.0        # Matches setup_env.py


@dataclass
class MissionAction:
    """A single action in the car's mission plan."""
    target_node: int               # Which node to drive to
    intention: int                 # Action intention to set when arrived
    description: str = ""          # Human-readable label for logging
    hold_duration: float = HOLD_DURATION_SEC   # default 3.8s, can override


@dataclass
class CarMissionState:
    """Tracks the car's overall mission progress."""
    actions: List[MissionAction] = field(default_factory=list)
    current_action_idx: int = 0
    cargo_small_count: int = 0 # Up to 2 small packages
    cargo_large_count: int = 0 # Up to 1 large package

    @property
    def current_action(self) -> Optional[MissionAction]:
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

def get_node_location_xy(roadmap, node_idx: int) -> np.ndarray:
    """
    Returns the (x, y) world position of a roadmap node.
    Matches the scaling used in setup_env.py.
    """
    node_pose = ROADMAP_SCALE_FACTOR * roadmap.nodes[node_idx].pose.flatten()
    return np.array([node_pose[0], node_pose[1]])


def has_arrived(current_pose: np.ndarray, target_xy: np.ndarray,
                tolerance: float = ARRIVAL_TOLERANCE_M) -> bool:
    """
    Returns True if the car is within `tolerance` meters of the target (horizontal).
    """
    horizontal_dist = np.linalg.norm(current_pose[:2] - target_xy)
    return horizontal_dist <= tolerance


def hold_completed(hold_start_time: Optional[float], current_time: float,
                   duration: float = HOLD_DURATION_SEC) -> bool:
    """
    Returns True if the position-hold has been maintained for `duration` seconds.
    `hold_start_time` is the timestamp when we first entered the hold state.
    """
    if hold_start_time is None:
        return False
    return (current_time - hold_start_time) >= duration


def build_mission_delivery4_only() -> CarMissionState:
    """
    Hardcoded mission: pick up one small package, deliver to Delivery 4 (node 22).
    This is a stub; the planner will replace this later.

    Delivery 4 details (from competition docs):
      - Node 22 (Python indexing) at coordinates (-19.84, 29.67, 0.05)
      - Small package, ground drop-off only (no window option)
    """
    mission = CarMissionState()
    mission.actions = [
        MissionAction(
            target_node=24,
            intention=INTENTION_PICKUP_SMALL,
            description="Pickup small package #1 at central pickup"
        ),
        MissionAction(
            target_node=22,
            intention=INTENTION_DROPOFF,
            description="Drop off at Delivery 4"
        ),
    ]
    return mission

def build_mission_delivery5_only() -> CarMissionState:
    """
    Stub mission for Phase 2: drive to central pickup, pick up the large
    package (delivery 5), drive to delivery 5's drop-off, drop it off.

    Tests the new INTENTION_PICKUP_LARGE intention end-to-end.

    Coordinates from the competition Pickup and Delivery Table:
      Central pickup (node 24): [-2.50305, 29.6703]
      Delivery 5 ground (node 10): [-12.8205, -4.5991]  (large package only)
    """
    mission = CarMissionState()
    mission.actions = [
        MissionAction(
            target_node=24,
            intention=INTENTION_PICKUP_LARGE,
            description="Pickup large package at central pickup"
        ),
        MissionAction(
            target_node=10,
            intention=INTENTION_DROPOFF,
            description="Drop off large package at Delivery 5"
        ),
    ]
    return mission


def build_mission_deliveries_4_and_5() -> CarMissionState:
    """
    Combined mission: deliver D4 (small) and D5 (large) in one run.

    The car must visit pickup separately for each because it can't batch
    a small with a large (different package types).

    Sequence:
      pickup → D4 → pickup → D5
    """
    mission = CarMissionState()
    mission.actions = [
        MissionAction(
            target_node=24,
            intention=INTENTION_PICKUP_SMALL,
            description="Pickup small package #1 at central pickup"
        ),
        MissionAction(
            target_node=22,
            intention=INTENTION_DROPOFF,
            description="Drop off small package at Delivery 4"
        ),
        MissionAction(
            target_node=24,
            intention=INTENTION_PICKUP_LARGE,
            description="Pickup large package at central pickup"
        ),
        MissionAction(
            target_node=10,
            intention=INTENTION_DROPOFF,
            description="Drop off large package at Delivery 5"
        ),
    ]
    return mission
    
# =============================================================================
# Utility Functions
# =============================================================================

def read_initial_positions(filepath: Path) -> np.ndarray:
    """
    Reads initial spawn positions from a text file.

    The file is expected to contain comma-separated numeric values.
    Lines starting with '#' are treated as comments.

    Returns:
        np.ndarray: First 4 numeric values representing initial pose data.
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

    return np.array(values[0:4], dtype=np.float64)





def load_plan_file(plan_path: Path):
    """
    Loads precomputed path planning data from a NumPy file.

    Returns:
        ndarray: Path poses used for navigation
    """
    qcar2_pathposes = np.load(plan_path, allow_pickle=True)
    return qcar2_pathposes


def stanley_controller(pose, vel_cmd, path_pose):
    """
    Stanley Controller Implementation

    Computes steering and velocity adjustments to follow a reference path.

    Core idea:
        - Minimize heading error (orientation mismatch)
        - Minimize cross-track error (lateral deviation)

    Returns:
        vel_cmd: Updated velocity command
        delta: Steering angle command
    """

    # ---------------- Controller Parameters ----------------
    epsilon = 1e-3   # Prevent division by zero
    k = 2.5          # Control gain
    delta_max = 0.6  # Steering saturation limit

    pose = np.asarray(pose, dtype=float)
    path_pose = np.asarray(path_pose, dtype=float)

    x, y, yaw = pose[0], pose[1], pose[2]
    path_x = path_pose[:, 0]
    path_y = path_pose[:, 1]
    path_yaw = path_pose[:, 2]

    # ---------------- Steering Control ----------------

    # Compute distance to all path points
    dx = path_x - x
    dy = path_y - y
    d = np.hypot(dx, dy)

    # Select closest path point
    target_idx = np.argmin(d)

    x_ref = path_x[target_idx]
    y_ref = path_y[target_idx]
    yaw_ref = path_yaw[target_idx]

    # Heading error (wrapped to [-pi, pi])
    psi_e = (yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi

    # Cross-track error (signed lateral distance)
    dxl = x - x_ref
    dyl = y - y_ref
    e_ct = -np.sin(yaw_ref) * dxl + np.cos(yaw_ref) * dyl
    e_ct = -e_ct

    # Stanley control law
    vel = vel_cmd * 13 / 0.2
    delta = psi_e + np.arctan2(k * e_ct, vel + epsilon)

    # Apply steering limits
    delta = np.clip(delta, -delta_max, delta_max)

    # ---------------- Velocity Control ----------------

    # Reduce speed near final target
    dist_to_end = np.linalg.norm(np.array([x, y]) - path_pose[-1, 0:2], ord=2)
    ang_to_end = abs(np.rad2deg(pose[2] - path_pose[-1, 2]))

    if dist_to_end < 10.0 and ang_to_end < 60.0:
        vel_cmd = dist_to_end / 10.0 * vel_cmd
    elif dist_to_end < 1.0 and ang_to_end < 60.0:
        vel_cmd = 0.0

    # Reduce speed during sharp turns
    if abs(delta) >= 0.90 * delta_max:
        vel_cmd = min(vel_cmd, 0.04)
    elif abs(delta) >= 0.80 * delta_max:
        vel_cmd = min(vel_cmd, 0.06)
    elif abs(delta) >= 0.70 * delta_max:
        vel_cmd = min(vel_cmd, 0.08)
    elif abs(delta) >= 0.60 * delta_max:
        vel_cmd = min(vel_cmd, 0.10)
    elif abs(delta) >= 0.50 * delta_max:
        vel_cmd = min(vel_cmd, 0.15)

    return vel_cmd, delta


# =============================================================================
# Main Execution Configuration
# =============================================================================

# region: Experiment constants
simulationTime = 1200   # Total simulation duration (seconds)
frequency = 200         # Control loop frequency (Hz)
frameRate = 30          # Camera frame rate
CameraCounts = int(round(frequency / frameRate))
useCameras = False      # Enable/disable camera streams
# endregion 


# =============================================================================
# State Variables and Initialization
# =============================================================================

counter = 0
receiveCounter = 0
receivedData = np.zeros(10)

receiveGameCounter = 0
pose = np.zeros(3)

# Load initial spawn position and planned paths
initial_position = read_initial_positions(Path("spawn_locations.txt"))
pathposes_path = Path(r"tools\QCar2_PathPlanning\qcar2_pathposes.npy")
pathposes = load_plan_file(pathposes_path)


# =============================================================================
# Camera Initialization (Optional)
# =============================================================================
# Cameras simulate multiple viewpoints around the vehicle
if useCameras:
    camRight = Camera2D(cameraId="0@tcpip://localhost:18961", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camBack  = Camera2D(cameraId="1@tcpip://localhost:18962", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camLeft  = Camera2D(cameraId="3@tcpip://localhost:18964", frameWidth=640, frameHeight=480, frameRate=frameRate)
    camFront = Camera2D(cameraId="2@tcpip://localhost:18963", frameWidth=640, frameHeight=480, frameRate=frameRate)


# =============================================================================
# Communication Streams Setup
# =============================================================================

# Main data stream (control commands and telemetry)
dataStream = BasicStream(
    'tcpip://localhost:18375',
    agent='C',
    sendBufferSize=1460,
    receiveBuffer=np.zeros((1,10), dtype=np.float64),
    recvBufferSize=1460,
    nonBlocking=False
)

# Client stream for receiving vehicle pose
client_car = BasicStream(
    'tcpip://localhost:19000',
    agent='C',
    sendBufferSize=8,
    receiveBuffer=np.zeros((1,3), dtype=np.float64),
    recvBufferSize=24,
    nonBlocking=False
)

timeout = Timeout(seconds=0, nanoseconds=10)


# =============================================================================
# Control Loop Initialization
# =============================================================================

timer = QTimer(frequency, simulationTime)

# Flags and control variables
flag_send_intention = True
intention = INTENTION_NOTHING
send_commands = np.array([0., 0.], dtype=np.float64)

# Initialize roadmap for node coordinate lookups
roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

# Build the mission (hardcoded for now; will be replaced by planner later)
mission = build_mission_deliveries_4_and_5()
print(f"Mission loaded with {len(mission.actions)} actions:")
for i, action in enumerate(mission.actions):
    print(f"  [{i}] {action.description}")

# State machine state
car_state = CarState.IDLE
hold_start_time: Optional[float] = None    # Timestamp when we entered the hold state
current_start_node = 8                     # Spawn node; updates as we move

# Default to no movement until the state machine takes over
vel_cmd = 0.0
path_pose = pathposes[8, 8]                # Stationary path (used when IDLE/MISSION_COMPLETE)


# =============================================================================
# Main Control Loop
# =============================================================================
try:
    while timer.check():

        current_time = timer.get_current_time()

        # ---------------- State Machine ----------------
        # Handle current state, possibly transitioning to next

        if car_state == CarState.IDLE:
            # First-time setup: start moving toward the first action's target
            if not mission.is_complete:
                action = mission.current_action
                target_node = action.target_node
                path_pose = pathposes[current_start_node, target_node]
                vel_cmd = 0.1
                intention = INTENTION_NOTHING   # No intention until we arrive
                flag_send_intention = True
                car_state = CarState.APPROACHING_NODE
                print(f"[STATE] IDLE -> APPROACHING_NODE (target={target_node})")
            else:
                car_state = CarState.MISSION_COMPLETE

        elif car_state == CarState.APPROACHING_NODE:
            # Drive toward target; check for arrival
            action = mission.current_action
            target_xy = get_node_location_xy(roadmap, action.target_node)

            if has_arrived(pose, target_xy):
                # We're at the target; switch to hold state and set the right intention
                vel_cmd = 0.0
                intention = action.intention
                flag_send_intention = True
                hold_start_time = current_time
                car_state = CarState.AT_NODE_HOLDING
                print(f"[STATE] ARRIVED at node {action.target_node}; "
                      f"holding for {HOLD_DURATION_SEC}s with intention={action.intention}")
            else:
                # Still driving; keep the same path and velocity (Stanley will steer)
                vel_cmd = 0.1
                intention = INTENTION_NOTHING


            if has_arrived(pose, target_xy):
                # We're at the target; switch to hold state and set the right intention
                vel_cmd = 0.0
                intention = action.intention
                flag_send_intention = True
                hold_start_time = current_time
                car_state = CarState.AT_NODE_HOLDING
                print(f"[STATE] ARRIVED at node {action.target_node}; "
                        f"holding for {HOLD_DURATION_SEC}s with intention={action.intention}")
            else:
                # Still driving; keep the same path and velocity (Stanley will steer)
                vel_cmd = 0.1
                intention = INTENTION_NOTHING

        elif car_state == CarState.AT_NODE_HOLDING:
            # Stay put with current intention until 3-second hold completes
            # intention already set in previous transition

            if hold_completed(hold_start_time, current_time, duration=action.hold_duration):
                action = mission.current_action
                actual_hold = current_time - hold_start_time
                print(f"[STATE] HOLD COMPLETE: {action.description}, "
                      f"actual_duration={actual_hold:.2f}s")

            # Track cargo changes
                if action.intention == INTENTION_PICKUP_SMALL:
                    mission.cargo_small_count += 1
                elif action.intention == INTENTION_PICKUP_LARGE:
                    mission.cargo_large_count += 1
                elif action.intention == INTENTION_DROPOFF:
                    # Drop off whatever we're carrying; large takes precedence
                    if mission.cargo_large_count > 0:
                        mission.cargo_large_count -= 1
                    elif mission.cargo_small_count > 0:
                        mission.cargo_small_count -= 1

                # Update where we are; next path will start from here
                current_start_node = action.target_node
                mission.advance()
                hold_start_time = None
                car_state = CarState.ACTION_COMPLETE

        elif car_state == CarState.ACTION_COMPLETE:
            # Decide whether mission is done or pick up the next action
            if mission.is_complete:
                car_state = CarState.MISSION_COMPLETE
                print(f"[STATE] MISSION COMPLETE at t={current_time:.1f}s")
            else:
                # Start moving toward the next target
                action = mission.current_action
                target_node = action.target_node
                path_pose = pathposes[current_start_node, target_node]
                vel_cmd = 0.1
                intention = INTENTION_NOTHING
                flag_send_intention = True
                car_state = CarState.APPROACHING_NODE
                print(f"[STATE] ACTION_COMPLETE -> APPROACHING_NODE (target={target_node})")

        elif car_state == CarState.MISSION_COMPLETE:
            # Mission done; just hold still
            vel_cmd = 0.0
            intention = INTENTION_NOTHING
            path_pose = pathposes[current_start_node, current_start_node]

        # ---------------- Client Communication ----------------
        # Ensure connection and receive vehicle pose
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

# ---------------- Control Computation ----------------
        # Compute velocity and steering using Stanley controller
        vel_cmd, steering_cmd = stanley_controller(pose, vel_cmd, path_pose)

        # Force hard stop when we're at the target during a hold phase.
        # This is critical for the game's pickup/drop-off detection.
        if car_state in (CarState.APPROACHING_NODE, CarState.AT_NODE_HOLDING):
            action = mission.current_action
            if action is not None:
                target_xy = get_node_location_xy(roadmap, action.target_node)
                dist_to_target = np.linalg.norm(pose[:2] - target_xy)

                if dist_to_target <= ARRIVAL_TOLERANCE_M:
                    vel_cmd = 0.0
                    steering_cmd = 0.0
                elif car_state == CarState.AT_NODE_HOLDING:
                    # Car drifted out of the target zone — reset hold
                    print(f"[STATE] Drifted out of target zone (dist={dist_to_target:.2f}m); resetting hold")
                    hold_start_time = None
                    car_state = CarState.APPROACHING_NODE

        send_commands = np.array([vel_cmd, steering_cmd], dtype=np.float64)

        # ---------------- Server Communication ----------------
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
                # Telemetry structure:
                # [0] Motor Power Consumption
                # [1] Battery Level
                # [2] Car Speed (m/s)
                # [3-5] Gyroscope data
                # [6-8] Accelerometer data
                # [9] Connection status flag

            # ---------------- Camera Handling ----------------
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

            # ---------------- Command Transmission ----------------
            # Send computed velocity and steering commands
            sentFlag = dataStream.send(send_commands)
            if sentFlag == -1:
                print('Server application not receiving.')
                break

        # Maintain loop timing
        timer.sleep()

except KeyboardInterrupt:
    print("\nExiting due to keyboard interrupt.")


# =============================================================================
# Cleanup
# =============================================================================

if useCameras:
    camLeft.terminate()
    camBack.terminate()
    camRight.terminate()
    camFront.terminate()

dataStream.terminate()
client_car.terminate()