# FaceReg

[Russian version / Русская версия](README-ru.md)

FaceReg is an offline face recognition project built around Python (`OpenCV` + `face_recognition`) with a Java desktop UI for running scripts, viewing the camera preview, and managing the local face database.

The project is aimed at practical local use:
- webcam face recognition without cloud services
- local database in `known_faces.pkl`
- automatic creation of IDs for unknown faces
- additional face snapshots and encoding accumulation
- per-ID logs plus a global log
- Java UI for launching modes, previewing camera frames, and managing IDs

## What Is Included

- Python recognition modes in `src/`
- Java Swing UI in `UI/`
- batch photo import
- duplicate encoding cleanup
- merge / rename / delete utilities for local IDs

## Install

### Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Java

You need a JDK (not only a JRE) to compile and run the UI.

## Quick Start

### Run the Java UI

```bash
javac UI/*.java
java UI.base
```

The UI can:
- start and stop Python modes
- show the `mainJV.py` camera preview inside the window
- show recent log lines
- rename, delete, merge, and optimize IDs

### Run Python Directly

```bash
python src/low_main.py
# or
python src/main.py
# or
python src/ceiling_main.py
# or
python src/mainJV.py
```

## Python Modes

### `src/low_main.py`

The lightest mode for weaker machines.

Use it when:
- CPU is limited
- camera angle is simple
- lower lag matters more than maximum recognition quality

### `src/main.py`

The default balanced mode.

It uses `multiprocessing` so detection and encoding run in separate processes, which keeps the main loop more responsive.

Use it when:
- you want the best general-purpose mode
- the machine has a normal modern CPU
- the camera is roughly at face level

### `src/ceiling_main.py`

The heavier mode for difficult camera angles.

Use it when:
- camera is mounted high
- faces are smaller because of distance
- people appear from top-down or side angles

### `src/mainJV.py`

This mode is designed for the Java UI.

Instead of opening an OpenCV window, it writes:
- `ui_frames/mainJV_frame.jpg`
- `ui_frames/mainJV_status.txt`

The Java UI reads these files and shows the camera preview inside the application window.

## Java UI

The Java UI lives in `UI/` and is the main desktop control panel.

Main features:
- start `src/mainJV.py`, `src/main.py`, `src/low_main.py`, `src/ceiling_main.py`
- run utilities such as photo import and duplicate cleanup
- show camera preview from `mainJV.py`
- show recent lines from `log.txt`
- force-stop Python processes when closing the app

Main UI actions:
- `Run` — starts the selected Python mode
- `Stop` — stops the running Python mode and its child worker processes
- `Optimize` — runs `src/check.py`
- `Rename` — rename an ID from the database
- `Delete` — delete an ID and related local files
- `Merge` — merge a duplicate ID into the original profile using a selection dialog
- `Close` — shuts down the app and cleans temporary UI files

## Face Database Utilities

### Import Photos

```bash
python src/import_foto_p.py
```

Expected input layout:

```text
foto_p/
  PersonA/
    1.jpg
    2.jpg
  PersonB/
    photo1.jpg
```

What it does:
- creates a new ID if the person does not exist
- appends encodings if the person already exists
- copies accepted images into `faces/<user_id>/`
- saves boxed previews into `foto_p_result/<person_name>/`

### Manage IDs

```bash
python src/manage_faces.py rename <id> "<new name>"
python src/manage_faces.py delete <id>
python src/manage_faces.py merge <duplicate_id> <target_id>
```

### Optimize Duplicate Encodings

```bash
python src/check.py
```

This checks each ID separately and removes very similar duplicate encodings to reduce database size.

## Project Layout

- `UI/` — Java Swing UI
- `src/` — Python scripts
- `src/compare_tools/` — comparison tools
- `known_faces.pkl` — local face database
- `faces/` — saved face images grouped by ID
- `logs/` — per-ID logs
- `log.txt` — global log
- `foto_p/` — source import photos
- `foto_p_result/` — processed import previews
- `ui_frames/` — temporary camera preview files for the Java UI

## Controls During Python Camera Modes

For camera modes that still open an OpenCV window:
- `q` — quit
- `w` — rename an ID
- `d` — delete an ID

## Important Notes

- The face database and photos are local files, not cloud data.
- Runtime data such as logs, photos, preview frames, and `known_faces.pkl` should not be committed to Git.
- The provided `.gitignore` is configured to ignore these generated files.

## Choosing the Right Workflow

- Use the Java UI if you want one place to run modes and manage the database.
- Use `src/mainJV.py` if you want the camera inside the Java window.
- Use `src/main.py` if you want the best general-purpose direct Python mode.
- Use `src/low_main.py` for lighter CPU usage.
- Use `src/ceiling_main.py` for difficult angles.
