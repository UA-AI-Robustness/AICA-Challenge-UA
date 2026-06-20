"""
perception.py — YOLO detection + RGB-D 3D localization for the AICA perception module.

Device-agnostic (runs on CPU; no .cuda() calls), so it works on this laptop.
Borrows the robust technique from Quanser's pit/YOLO (median depth over the
detected region) but is written clean against the Ultralytics API directly.

Pipeline this supports:
    RGB frame --(YOLO)--> detections (class, confidence, bounding box)
    + depth  --(median over box)--> distance to object
    + intrinsics --(back-project)--> 3D point in CAMERA frame
    + drone pose + extrinsics --(transform)--> 3D point in WORLD frame

Phases:
  - Phase 2/3: detect() works as soon as you have a trained model (.pt).
  - Phase 3:   distance_for_box() + backproject_pixel_to_camera() need depth +
               intrinsics. estimate_world_point() additionally needs the drone
               pose and the camera mounting (extrinsics) — flagged below.

Runnable now for a smoke test:
    python perception.py --model yolo11n.pt --image some.jpg
    python perception.py --model yolo11n.pt           # no image: just confirms load
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Config you will calibrate (placeholders flagged clearly)
# =============================================================================

# Depth scale: raw depth-PX values are divided by this to get metric units.
# Quanser's QCar code uses 5.5; VERIFY this for the QDrone2 depth stream by
# comparing a known distance (Phase 0). Wrong scale => wrong 3D, linearly.
DEPTH_SCALE = 5.5

# Camera intrinsics (3x3 K). MUST be filled from the QDrone2 RealSense
# (Camera3D.intrinsics_depth / intrinsics_rgb) at the SAME resolution you
# capture/infer at. NaN here on purpose so misuse fails loudly rather than
# silently producing garbage 3D coordinates.
DEFAULT_K = np.array([
    [np.nan, 0.0,    np.nan],
    [0.0,    np.nan, np.nan],
    [0.0,    0.0,    1.0],
], dtype=np.float64)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    # bounding box in pixels, [x1, y1, x2, y2]
    box_xyxy: Tuple[int, int, int, int]
    # convenience: box center pixel (u, v)
    center_uv: Tuple[float, float] = field(init=False)
    # filled in later stages (optional)
    distance_m: Optional[float] = None          # metric distance from camera
    point_camera: Optional[Tuple[float, float, float]] = None  # 3D in camera frame
    point_world: Optional[Tuple[float, float, float]] = None    # 3D in world frame

    def __post_init__(self):
        x1, y1, x2, y2 = self.box_xyxy
        self.center_uv = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self):
        d = asdict(self)
        return d


class PerceptionModule:
    def __init__(self,
                 model_path: str,
                 device: str = "cpu",
                 confidence: float = 0.3,
                 class_filter: Optional[List[int]] = None,
                 K: np.ndarray = DEFAULT_K,
                 depth_scale: float = DEPTH_SCALE):
        """
        model_path: path to a YOLO .pt (your trained model, or yolo11n.pt to smoke-test).
        device: 'cpu' (this laptop) or 'cuda'/'0' on a GPU machine.
        confidence: detection threshold.
        class_filter: restrict to these class ids (None = all).
        K: 3x3 camera intrinsics; required for 3D back-projection.
        depth_scale: raw-PX -> metric divisor.
        """
        # Imported here so the module can be inspected without ultralytics present.
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.class_filter = class_filter
        self.K = np.asarray(K, dtype=np.float64)
        self.depth_scale = float(depth_scale)
        self.names = self.model.names  # {id: name}
        print(f"[perception] loaded {model_path} on '{device}', "
              f"{len(self.names)} classes, conf>={confidence}")

    # ----- Detection (Phase 2/3) -------------------------------------------
    def detect(self, rgb: np.ndarray, verbose: bool = False) -> List[Detection]:
        """Run YOLO on an RGB image. Returns a list of Detection (no 3D yet)."""
        results = self.model.predict(
            rgb,
            device=self.device,
            conf=self.confidence,
            classes=self.class_filter,
            verbose=verbose,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cid in zip(boxes, confs, clss):
            x1, y1, x2, y2 = [int(v) for v in box]
            out.append(Detection(
                cls_id=int(cid),
                cls_name=self.names.get(int(cid), str(cid)),
                confidence=float(conf),
                box_xyxy=(x1, y1, x2, y2),
            ))
        return out

    # ----- Distance from depth (Phase 3) -----------------------------------
    def distance_for_box(self, depth_raw: np.ndarray, det: Detection) -> Optional[float]:
        """Median metric depth over the detection's box region.

        Median (not mean) is robust to background pixels and depth holes — the
        same idea Quanser uses over a segmentation mask. raw-PX -> metric via
        depth_scale. Returns None if no valid depth in the region.
        """
        if depth_raw is None:
            return None
        h, w = depth_raw.shape[:2]
        x1, y1, x2, y2 = det.box_xyxy
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        region = np.asarray(depth_raw[y1:y2, x1:x2], dtype=np.float32).reshape(-1)
        region = region[np.isfinite(region)]
        region = region[region > 0]
        if region.size == 0:
            return None
        dist = float(np.median(region)) / self.depth_scale
        det.distance_m = dist
        return dist

    # ----- Pixel + depth -> 3D in camera frame (Phase 3) -------------------
    def backproject_pixel_to_camera(self, u: float, v: float, Z: float) -> Optional[Tuple[float, float, float]]:
        """Standard pinhole back-projection using intrinsics K.
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = Z
        Returns None (and warns) if K is not populated.
        """
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        if not np.all(np.isfinite([fx, fy, cx, cy])):
            print("[perception] WARNING: intrinsics K not set; cannot back-project. "
                  "Fill K from Camera3D.intrinsics_* (Phase 0).")
            return None
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return (float(X), float(Y), float(Z))

    def localize_box_camera(self, depth_raw: np.ndarray, det: Detection) -> Optional[Tuple[float, float, float]]:
        """Combine distance + back-projection: 3D point of a detection in CAMERA frame."""
        Z = self.distance_for_box(depth_raw, det)
        if Z is None:
            return None
        u, v = det.center_uv
        pt = self.backproject_pixel_to_camera(u, v, Z)
        det.point_camera = pt
        return pt

    # ----- Camera frame -> world frame (Phase 3/4) -------------------------
    def camera_to_world(self,
                        point_cam: Tuple[float, float, float],
                        drone_pose_xyzyaw: np.ndarray,
                        cam_offset_body: np.ndarray = np.array([0.0, 0.0, 0.0]),
                        cam_R_body: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Transform a camera-frame point to world coordinates.

        drone_pose_xyzyaw: [x, y, z, yaw] of the drone in world (from telemetry).
        cam_offset_body:   camera position relative to drone body origin (meters).
                           Pull from Camera3D extrinsics; placeholder is zero.
        cam_R_body:        3x3 rotation from camera axes to body axes. If None,
                           assumes camera forward = body forward (identity). This
                           and the yaw convention MUST be validated against the
                           game.py ground-truth window coords (Phase 3); this is
                           the single most error-prone transform, so we keep it
                           explicit and testable rather than hidden.
        """
        p = np.asarray(point_cam, dtype=np.float64)
        if cam_R_body is None:
            cam_R_body = np.eye(3)
        # point in body frame
        p_body = cam_R_body @ p + np.asarray(cam_offset_body, dtype=np.float64)
        # rotate body -> world by drone yaw about Z, then translate by drone xyz
        x, y, z, yaw = (drone_pose_xyzyaw[0], drone_pose_xyzyaw[1],
                        drone_pose_xyzyaw[2], drone_pose_xyzyaw[3])
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0, 0.0, 1.0]])
        p_world = Rz @ p_body + np.array([x, y, z])
        return (float(p_world[0]), float(p_world[1]), float(p_world[2]))

    def estimate_world_point(self,
                             depth_raw: np.ndarray,
                             det: Detection,
                             drone_pose_xyzyaw: np.ndarray,
                             **xform_kwargs) -> Optional[Tuple[float, float, float]]:
        """Full chain: detection + depth + intrinsics + pose -> world 3D point."""
        pcam = self.localize_box_camera(depth_raw, det)
        if pcam is None:
            return None
        pw = self.camera_to_world(pcam, drone_pose_xyzyaw, **xform_kwargs)
        det.point_world = pw
        return pw


def _smoke_test(model_path: str, image_path: Optional[str]):
    """Confirm the inference path runs on this machine (CPU)."""
    pm = PerceptionModule(model_path=model_path, device="cpu", confidence=0.25)
    if image_path:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"[smoke] could not read image: {image_path}")
            return
        dets = pm.detect(img)
        print(f"[smoke] {len(dets)} detection(s) in {image_path}:")
        for d in dets:
            print(f"   {d.cls_name:>15s}  conf={d.confidence:.2f}  box={d.box_xyxy}")
    else:
        print("[smoke] model loaded OK. Pass --image PATH to run a detection.")
        print(f"[smoke] class names: {pm.names}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AICA perception module smoke test")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="path to YOLO weights (.pt); yolo11n.pt for a generic smoke test")
    ap.add_argument("--image", default=None, help="optional image to run detection on")
    args = ap.parse_args()
    _smoke_test(args.model, args.image)
