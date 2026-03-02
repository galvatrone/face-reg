# FaceReg

[Russian version / Русская версия](README-ru.md)

FaceReg is an offline Python face recognition project built around `OpenCV` and `face_recognition` (`dlib`).
It captures frames from a webcam, detects faces, compares them against a local database, and keeps a stable identity for the same person even when recognition briefly becomes noisy.

The project is focused on practical use:
- automatic creation of a new UUID for unknown faces
- saving the first face photo and additional face snapshots
- gradual accumulation of new encodings for the same person
- global logging plus a separate log file for each detected ID
- multiple runtime modes for different hardware and camera angles

![FaceReg preview](https://www.oneix.ltd/assets/images/project-single/2-facer-reg.png)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python low_main.py
# or
python main.py
# or
python ceiling_main.py
```

## Program Modes

### `low_main.py`

This is the lightest version.
It is intended for weaker computers, older laptops, or machines with fewer CPU cores.

Main characteristics:
- simplest pipeline
- lowest CPU usage
- least aggressive face detection settings
- best choice when you need the app to keep running with minimal lag

Tradeoff:
- lower accuracy at distance
- weaker performance on difficult face angles
- less tolerant to brief recognition failures

Use this mode when:
- the computer is slow
- the camera is mostly frontal
- stability of the UI matters more than maximum detection quality

### `main.py`

This is the main everyday version.
It moves heavy face detection and encoding work into a separate process using `multiprocessing`, so the camera window and the main loop stay more responsive.

Main characteristics:
- balanced mode between speed and recognition quality
- lower visible lag than the lightweight single-process version
- better recovery from short misses thanks to held tracks and visibility timeout
- good default choice for normal desktop use

Tradeoff:
- uses more CPU than `low_main.py`
- still optimized for more typical camera angles, not the hardest top-down views

Use this mode when:
- you have a normal modern CPU
- you want the best general-purpose version
- the camera is roughly at face level or only slightly tilted

### `ceiling_main.py`

This is the heavy angle-tuned version.
It is designed for more difficult cameras, especially ceiling-mounted or strongly tilted cameras that look down and from the side.

Compared with `main.py`, this mode:
- uses a larger detection scale
- runs detection more often
- keeps tracks alive longer through short recognition dropouts
- uses stronger preprocessing for difficult lighting and smaller faces
- scans several rotated views of the frame to recover angled faces
- can request higher camera resolution for distant subjects

Tradeoff:
- highest CPU usage
- slower than the other modes
- more expensive detection pipeline because it processes multiple rotated variants

Use this mode when:
- the camera is mounted high
- people appear under an angle
- faces are smaller because of distance
- accuracy in difficult geometry matters more than speed

## Photo Import

`import_foto_p.py` imports faces from folders of photos into the local database.

How it works:
- create a folder `foto_p/`
- inside it, create one subfolder per person
- the subfolder name becomes the person name
- every valid photo in that folder is used to build face encodings

Example:

```text
foto_p/
  Misha/
    1.jpg
    2.jpg
  Alex/
    a.jpg
    b.jpg
```

Import behavior:
- if the name does not exist yet, a new UUID is created
- if the name already exists, the script appends new encodings to the existing person
- valid photos are copied into `faces/<user_id>/`
- preview copies with a face box are saved into `foto_p_result/<person_name>/`
- photos with zero faces or multiple faces are skipped

Run:

```bash
python import_foto_p.py
```

## Data Layout

- `known_faces.pkl` — local face database with names and face encodings
- `faces/<user_id>/` — saved face images for each person
- `logs/` — per-person log files
- `log.txt` — global event log
- `foto_p/` — source folders for batch photo import
- `foto_p_result/` — annotated import previews with face boxes

## Controls

During webcam execution:
- `q` — quit
- `w` — rename an ID
- `d` — delete an ID and related files

## Choosing the Right Mode

Use `low_main.py` if the machine is weak and you want the lightest possible load.

Use `main.py` if you want the best general-purpose version and a good balance between speed and stability.

Use `ceiling_main.py` if your camera is mounted high, angled, or looking at people from above and from the side.
