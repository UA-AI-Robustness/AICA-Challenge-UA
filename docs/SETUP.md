# Setup Notes

Installing the AICA Challenge environment is non-trivial. This document captures everything that worked and every gotcha we hit.

## Prerequisites

- Windows 10/11
- ~8 GB free disk space
- Python 3.12 (NOT 3.13 — Quanser wheels target 3.12 specifically)

## Step-by-Step Install

### 1. Install Python 3.12

Download from https://www.python.org/downloads/release/python-3120/

During install, check **"Add Python to PATH"**.

### 2. Install QLabs and QUARC

Obtain access through the AICA competition registration. Follow Quanser's installer; accept defaults.

### 3. Run Quanser's Python setup script

Run with admin PowerShell from the Quanser install folder.

**Known issue:** The script may fail with this error:
```
ERROR: Invalid requirement: 'python"': Expected semicolon...
```

This is a quoting bug in Quanser's batch script. It still installs `requirements.txt` dependencies successfully, but skips installing the main Quanser API wheels.

**Workaround:** Install Quanser wheels manually:

```powershell
cd "C:\Program Files\Quanser\Quanser SDK\python"
python -m pip install --user --no-index --find-links . .\quanser_common-*.whl
python -m pip install --user --no-index --find-links . .\quanser_communications-*.whl
python -m pip install --user --no-index --find-links . .\quanser_multimedia-*.whl
python -m pip install --user --no-index --find-links . .\quanser_devices-*.whl
python -m pip install --user --no-index --find-links . .\quanser_hardware-*.whl
python -m pip install --user --no-index --find-links . .\quanser_image_processing-*.whl
python -m pip install --user --no-index --find-links . .\quanser_api-*.whl
```

The `--no-index --find-links .` flags tell pip to use local wheels only.

### 4. Set environment variables

In Windows Environment Variables → User variables:

```
PYTHONPATH = C:\Users\<USERNAME>\Documents\AICA Challenge\Quanser\0_libraries\python
QAL_DIR    = C:\Users\<USERNAME>\Documents\AICA Challenge\Quanser
RTMODELS_DIR = C:\Users\<USERNAME>\Documents\AICA Challenge\Quanser\0_libraries\resources\rt_models
```

Restart all terminals after changing.

### 5. Verify install

Run our `check_install.py` script (when we add one to `tools/`):

```powershell
python tools/check_install.py
```

All checks should show ✓.

## Troubleshooting

### "ModuleNotFoundError: No module named 'qvl'"

PYTHONPATH isn't pointing to the right folder. Verify with `$env:PYTHONPATH` in PowerShell.

### "The remote peer refused the connection"

Either QLabs isn't running, `setup_env.py` didn't run successfully, or the real-time models didn't start. Run components one at a time to isolate which step fails.

### Add more issues here as the team encounters them