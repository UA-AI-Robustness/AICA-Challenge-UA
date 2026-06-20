# AICA Perception Pipeline

Off-sim machinery for the YOLO + RGB-D perception module. Everything here runs
on the laptop (CPU) except model training, which goes to Colab's free GPU.
Build and test all of this without the simulator; slot in real images once
capture is sorted (see PERCEPTION_PLAN.md for the hardware-capture constraint).

## Files

| File | What it is | When you use it |
|---|---|---|
| `perception.py` | Inference module: YOLO detection → depth distance → 3D back-projection → world coords. Device-agnostic (CPU). | Phases 2–4. Testable now with `yolo11n.pt`. |
| `setup_dataset.py` | Organizes captured frames; splits locally-labeled data into train/val. | Phase 1, after capture/labeling. |
| `data.yaml` | YOLO dataset config (classes, train/val paths). Template for the local labeling route. | Phase 1–2. |
| `TRAINING_COLAB.md` | Step-by-step guide to train the detector on Colab GPU. | Phase 2. |

## Order of operations

1. **Capture** RGB frames of the targets from diverse viewpoints (survey mode in
   the drone navigator). *(Blocked on capture hardware — see plan.)*
2. **Collect** them for labeling: `python setup_dataset.py collect`
3. **Label** in Roboflow (cloud, easy) or CVAT/LabelImg (local). Decide your
   classes here and keep them consistent everywhere.
4. **Train** on Colab GPU following `TRAINING_COLAB.md`; download `best.pt`.
5. **Detect** on the laptop: `python perception.py --model best.pt --image <frame>`
6. **Localize in 3D** (Phase 3): fill in the camera intrinsics `K` and
   `depth_scale` in `perception.py`, then use `estimate_world_point()`.
   Validate detected window positions against the known `game.py` coordinates.
7. **Integrate** with navigation (Phase 4): replace hardcoded targets with
   perception output.

## What still needs real values (flagged in code)

- **Camera intrinsics `K`** — fill from `Camera3D.intrinsics_rgb/_depth` at your
  capture resolution. Until then, 3D back-projection refuses to run (fails loud).
- **`depth_scale`** — Quanser uses 5.5 for the QCar; verify for the QDrone2 depth.
- **Camera→body extrinsics + yaw convention** — the `camera_to_world` transform
  has documented placeholders; calibrate against `game.py` ground truth in Phase 3.

## Smoke test right now (no sim, no trained model)

```powershell
python perception.py --model yolo11n.pt
```
Confirms the inference stack loads and runs on your CPU. With `--image <path>`
it runs a generic COCO detection so you can see the detection path working
before you have a custom model.
