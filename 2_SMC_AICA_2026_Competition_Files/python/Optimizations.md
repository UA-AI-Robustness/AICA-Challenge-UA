# Optimization Notes — AICA Challenge 2026 (UA Team)

## Score Progression

|Configuration|Score|Δ vs Previous|Δ vs Baseline|
|---|---|---|---|
|Mentor's prior hand-tuned baseline|5322|—|—|
|5543 baseline (D D D C C, no transfers)|5543|+221|+221|
|+ Car velocity bump (0.10 → 0.13)|5554|+11|+232|
|+ Drone spawn at pickup (z=3 hover)|5616|+62|+294|
|+ Per-leg drone flight times + hold trim 3.5→3.2|5622|+6|+300|
|**+ Tighter flight times + hold 3.05 + car 0.15**|**5648**|**+26**|**+326**|

Total improvement over prior baseline: **+326 points**.

---

## Main Focus of This Push

This push consolidates **single-vehicle assignment optimization** under tight time pressure. The drone handles all three window deliveries (D1, D2, D3) and the car handles ground-only deliveries (D4, D5). Optimization effort centered on minimizing mission completion time within strict compliance with the AICA documented limits.

Our prior analysis at slower vehicle speeds (`vel_cmd = 0.10`, baseline drone flight times) suggested that multi-agent transfers underperformed pure single-assignment due to synchronization wait times that exceeded cooperation benefits. **That finding was measured under different vehicle speed parameters and will be re-evaluated** — see the Future Work section.

---

## Architectural Decisions

### Strategy: Single-Vehicle Assignment (Current)

- **Drone**: handles all 3 window deliveries (D1, D2, D3) in order D3 → D1 → D2
    - D3 first because it is closest to pickup (17 m vs 51 m for D1)
    - D1 second because the +400 bonus is largest, recovering more of the delivery time penalty
    - D2 last because its building-avoidance via-point makes it the longest delivery
- **Car**: handles ground-only deliveries (D4 small, then D5 large)

### Mission Order Verified by Enumeration

All 6 permutations of D1/D2/D3 ordering were scored against the empirical timing model. D3 → D1 → D2 yielded the highest drone-side score (3732 vs 3700–3726 for alternatives).

### Multi-Agent Transfers: Empirical Status

|Transfer variant|Empirical score|Δ vs no-transfer baseline|Conditions tested under|
|---|---|---|---|
|D2 handoff (drone D3+D1 solo, car delivers D2 ground for drone to take up)|5527|−16|`vel_cmd = 0.10`, drone flight times 18/28s|
|D1 handoff|5373|−170|`vel_cmd = 0.10`, baseline drone flight times|

**Important caveat**: these results were measured _before_ the current speed optimizations. With car `vel_cmd = 0.15` (50 % faster than the original test) and tightened drone flight times, the synchronization timing math has changed. The drone now arrives at any potential rendezvous earlier _and_ the car arrives sooner — but not proportionally, since spawn-at-pickup gave the drone a larger relative speedup than the car velocity bump gave the car. Whether the wait gap shrinks enough to make a transfer net-positive at current speeds is **not yet empirically verified** and is on the future-work list.

---

## Optimizations Applied (with Compliance References)

### 1. Drone Spawn at Pickup Hover Altitude

**Change**: `spawn_locations.txt` drone position changed from `(0, 0, 0, 0)` (default ground spawn) to `(-2.50305, 29.6703, 3.0, 0)` — directly above the central pickup pad at cruise altitude.

**Effect**: Eliminates the ~16 s initial flight from origin to pickup. The drone enters the pickup hold immediately at t=0.

**Why this is the largest single gain (+62 points)**: Time saved cascades to all three drone deliveries — D3, D1, and D2 all complete ~16 s earlier.

**Compliance**: Operational Guide § 4 explicitly states: _"Initial vehicle positions and headings are defined in the Spawn Locations (`spawn_locations.txt`), which **can be modified** to set custom spawn locations."_ Listed under "Additional Information for Advanced Development."

**Code adjustments required**:

- Initial pose vector uses `initial_position[2]` instead of hardcoded `0.0` (so the drone doesn't try to fly DOWN to z=0 on the first frame).
- Hover command preserves spawn altitude when `z ≥ 1.5` (drone is already at cruise altitude, no takeoff needed).

### 2. Per-Leg Flight Time Customization

Different delivery legs have very different distances. The example navigator used uniform 18 s window-flight times, which is overkill for D3 (17 m) and overly conservative for D1 (51 m). Per-leg overrides give us tighter trajectories tailored to each leg's actual length.

|Leg|Distance|Original|This push|
|---|---|---|---|
|Pickup → D3 window|17 m, +2 m climb|18 s|**10 s**|
|D3 → pickup return|17 m, descent|12 s|**9 s**|
|Pickup → D1 window|51 m, +7 m climb|18 s|**13 s**|
|D1 → pickup return|51 m, descent|12 s|**10 s**|
|Pickup → D2 window|~40 m with via-point|28 s|**22 s**|

**Compliance**: No documented limit on waypoint timing exists. The drone's physical capability is enforced by the immutable Virtual FlightStack (`virtual_FlightStack.rt-win64`), which we cannot modify. We simply ask the drone to track waypoints sooner; it tracks them as fast as physics allows.

### 3. Hold Time Trim Toward 3.0 s Minimum

|Hold|Was|Now|
|---|---|---|
|Drone pickup/dropoff holds|4.5 s (baseline default)|**3.05 s**|
|Car pickup/dropoff holds|3.1 s|**3.05 s**|

3.05 s leaves a 0.05 s buffer above the documented 3.0 s minimum to handle frame-timing jitter between the navigator and the game scoring logic.

**Compliance**: Detailed Scenario § Scenario Rules: _"maintaining, **for at least 3 seconds**"_. Both car and drone holds meet this requirement.

### 4. Car Velocity Command 0.10 → 0.15

The example navigator used `vel_cmd = 0.1` as a conservative starting value. The documented interval is `[-0.2, 0.2]` (per the comment block at the top of the original `QCar2_Navigator.py`). We use **0.15**, which is 75 % of the documented maximum.

**Compliance**: Original `QCar2_Navigator.py` example code states explicitly:

```python
# The QCar2 Limits:
#   - The velocity command must be in the interval [-0.2, 0.2]
#   - The steering command must be in the interval  [-0.6, 0.6]
```

0.15 is well inside this interval. The original conservative speed-reduction tier on sharp turns (`0.04, 0.06, 0.08, 0.10, 0.12`) is preserved unchanged, so turns are taken at the same speeds as the proven baseline.

### 5. State-Machine Mission Execution

Both navigators replace the example's keyboard-driven task switching with explicit state machines:

- **Car states**: IDLE → APPROACHING_NODE → AT_NODE_HOLDING → ACTION_COMPLETE → MISSION_COMPLETE
- **Drone states**: IDLE → APPROACHING_WAYPOINT → AT_WAYPOINT_HOLDING → ACTION_COMPLETE → MISSION_COMPLETE

Each mission is built declaratively as a list of `MissionAction` / `DroneMissionAction` dataclasses. This makes the planning logic decoupled from the control logic and easy to modify.

---

## Compliance Audit Summary

|Requirement|Source|Our value|Status|
|---|---|---|---|
|`vel_cmd ∈ [-0.2, 0.2]`|`QCar2_Navigator.py` example comment|0.15|✓|
|`steering ∈ [-0.6, 0.6]`|`QCar2_Navigator.py` example comment|Clipped at ±0.6|✓|
|Hold ≥ 3 s (all actions)|Detailed Scenario § Scenario Rules|3.05 s|✓|
|2.0 m horizontal tolerance|Detailed Scenario|Enforced by `game.py`|✓|
|0–4 m vertical tolerance (drone)|Detailed Scenario|Enforced by `game.py`|✓|
|`setup_env.py` unmodified|Op. Guide § 4|Untouched|✓|
|`game.py` unmodified|Op. Guide § 4|Untouched|✓|
|`Virtual_DriveStack.rt-win64` unmodified|Op. Guide § 4|Untouched|✓|
|`virtual_FlightStack.rt-win64` unmodified|Op. Guide § 4|Untouched|✓|
|`QCar2_Workspace.rt-win64` unmodified|Op. Guide § 4|Untouched|✓|
|`QDrone2_Open_Workspace.rt-win64` unmodified|Op. Guide § 4|Untouched|✓|
|Action intentions (0–5 car, 0–4 drone)|Detailed Scenario|Documented values only|✓|

---

## Files Modified

- `2_SMC_AICA_2026_Competition_Files/python/QCar2_Navigator.py` — state machine, mission planner, Stanley controller with original speed-reduction tier, `vel_cmd = 0.15`
- `2_SMC_AICA_2026_Competition_Files/python/QDrone2_Navigator.py` — state machine, mission planner, per-leg flight time overrides, time-parameterized linear trajectories with via-points, spawn-at-altitude handling
- `2_SMC_AICA_2026_Competition_Files/python/spawn_locations.txt` — drone spawn at central pickup hover altitude

## Files NOT Modified

- `setup_env.py`
- `game.py`
- `Virtual_DriveStack.rt-win64`
- `virtual_FlightStack.rt-win64`
- `QCar2_Workspace.rt-win64`
- `QDrone2_Open_Workspace.rt-win64`

---

## Future Work

### Single-assignment exploration (priority)

Still some headroom to extract from the current architecture before structural changes:

- **Car spawn offset**: relocating the car spawn closer to pickup without spawning directly on the pickup pad (the on-pad spawn caused a pose-stream failure in earlier testing). Estimated potential: +10–30 points if a stable offset location is found.
- **More aggressive drone speeds** (D1 flight 13 → 12 s, D2 flight 22 → 20 s): empirical testing required, risk of flight-stack tracking instability.
- **Voxel-map Dijkstra for D2 routing**: replacing the hand-tuned via-point with a planner-generated optimal route through `qdrone2_plans.npz`'s occupancy grid. Higher implementation cost.
- **Loosened arrival tolerance** (2.0 → 2.5 m): could trigger action intentions sooner if game.py's internal tolerance is more permissive than ours.

### Multi-agent re-evaluation (after single-assignment saturates)

The empirical transfer results that informed our single-assignment decision were measured at **slower vehicle speeds**. With current optimizations:

- The car reaches any rendezvous point ~33 % faster than in the original transfer tests.
- The drone reaches potential rendezvous points earlier as well, but the _relative_ speedup is larger for the drone (spawn-at-pickup saved ~16 s of initial flight).

These two changes shift the synchronization-wait math, and the previous "drone always finishes first and waits" finding may no longer hold for all transfer geometries. Specifically worth re-testing:

- **D2 handoff at the D2 ground pad** with the new car ordering (D4 → D2 ground transfer → return → D5, taking advantage of the car's higher speed to shorten the D4-to-rendezvous leg).
- **Modified D2 transfer routing** where car batches 2 small packages (capacity allows), delivering D4 first then proceeding directly to the D2 rendezvous, while drone uses D3 → D1 → fly directly to D2 ground (skipping the return to pickup).

A clean re-test would isolate the transfer overhead at current speeds and produce a defensible comparison against the 5648 single-assignment score. If the gap closes, transfers may become net-positive; if it widens, we have stronger evidence that single-assignment is optimal across reasonable parameter ranges.

---

## Theoretical Ceiling

The current score of **5648** corresponds to approximately **98.7 %** of the theoretical maximum (~5670) given the immutable physical constraints (drone flight stack speed limit, car drive stack acceleration profile, building geometry for D2). Remaining single-assignment gains are bounded; substantial further improvement likely requires either (a) the multi-agent re-evaluation above or (b) a structural change such as Dijkstra-planned drone routing.