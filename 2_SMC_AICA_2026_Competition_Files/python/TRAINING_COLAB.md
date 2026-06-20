# Training the AICA detector on Google Colab (free GPU)

Your laptop is CPU-only (AMD), so training happens on Colab's free GPU. You
label and use the model locally; only the training step goes to the cloud.
Inference (`perception.py`) runs fine on your laptop CPU afterward.

## Workflow at a glance

```
capture frames (sim)  ->  label (Roboflow or CVAT)  ->  train (Colab GPU)
   ->  download best.pt  ->  perception.py on laptop (CPU)  ->  integrate
```

---

## Step 1 — Get a labeled dataset

**Roboflow route (easiest):** upload your images, draw boxes, then
*Generate → Export → YOLOv11 → "show download code"*. You'll get a snippet like:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
dataset = rf.workspace("ws").project("proj").version(1).download("yolov11")
```
That downloads a ready dataset folder with its own `data.yaml`.

**Local route (CVAT/LabelImg):** label locally, then on your laptop run
`setup_dataset.py split ...` to make `dataset/` with train/val, copy `data.yaml`
into it, and zip the `dataset/` folder to upload to Colab.

> Confirm with your PI whether lab images may go on Roboflow's cloud. If not,
> use the local route.

---

## Step 2 — Open a GPU Colab notebook

Go to colab.research.google.com → New notebook → **Runtime → Change runtime
type → GPU (T4)**. Then paste the cells below in order.

### Cell 1 — confirm GPU + install Ultralytics
```python
!nvidia-smi
!pip -q install ultralytics
import ultralytics, torch
print("ultralytics", ultralytics.__version__, "| CUDA:", torch.cuda.is_available())
```

### Cell 2 — get your dataset into Colab

*Roboflow:* paste your download snippet (from Step 1). Note the printed
`dataset.location` — that folder holds `data.yaml`.

*Local zip:* upload and unzip:
```python
from google.colab import files
up = files.upload()                      # choose your dataset.zip
import zipfile, os
name = next(iter(up))
with zipfile.ZipFile(name) as z:
    z.extractall("dataset")
print(os.listdir("dataset"))             # expect images/, labels/, data.yaml
```

### Cell 3 — train
```python
from ultralytics import YOLO

# start from a small pretrained model and fine-tune on your classes.
# yolo11n = nano (fast, light). Bump to yolo11s for more accuracy if needed.
model = YOLO("yolo11n.pt")

results = model.train(
    data="dataset/data.yaml",   # or dataset.location + "/data.yaml" for Roboflow
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,                # early-stop if val stops improving
    project="aica",
    name="detector_v1",
)
```

### Cell 4 — check results, then download the weights
```python
# validation metrics (mAP etc.) print during/after training.
# best weights are saved here:
best = "aica/detector_v1/weights/best.pt"
from google.colab import files
files.download(best)            # saves best.pt to your computer
```

Also glance at `aica/detector_v1/` for `results.png` (training curves) and the
confusion matrix — those are figures for your poster.

---

## Step 3 — Use the model on your laptop

Put the downloaded `best.pt` next to `perception.py` and smoke-test on a frame:
```powershell
python perception.py --model best.pt --image captures/rgb/frame_0000.png
```
You should see detections printed. From there, `perception.py` feeds the
3D-localization and (later) navigation-integration phases.

---

## Tips

- **Start small:** even 100–300 labeled images per class gives a usable first
  model. Get the loop working end-to-end before scaling the dataset.
- **mAP is your headline metric.** Report it overall and per class. The
  validation set should include harder viewpoints (far/oblique/occluded) so the
  number reflects real robustness — that's your research claim.
- **Class consistency:** the class names/order in `data.yaml` must match what
  you labeled and what `perception.py` expects. Decide them once, keep them fixed.
- **Re-train is cheap:** as you capture more/better images, re-run Step 2–3.
  Keep each model versioned (`detector_v1`, `v2`, ...) so you can compare.
