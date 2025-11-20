# equalizer

Minimal audio equalizer example (Python).

Prerequisites
 - Python 3.8 or newer
 - pip

### Install

1. Create and activate a virtual environment:

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS (bash / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies from the provided requirements file:

```powershell
pip install -r requirements.txt
```

### Usage

Run the main application (the project entrypoint is `main.py`):

```powershell
python main.py
```

Other files
 - `equalizer_app.py` — GUI / application logic (if present)
 - `audio_equalizer.py` — audio processing utilities

### Notes
- If you add music files place them in the `music/` folder.
- If things fail, run the commands above in an activated virtual environment and ensure you have the correct Python version.

#### Tkinter

This project uses a Tk-based GUI (`tkinter`). Tkinter is part of the Python standard library on many distributions, so it is not listed in `requirements.txt` (it's not a pip package). However, some systems require installing a system package to provide the Tcl/Tk libraries.
