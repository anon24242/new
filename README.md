# School Surveillance Face Checker

This repository provides a Python script to compare student faces from a set of known images against images captured by a school surveillance system.

## Requirements

- Python 3.9+
- `face_recognition`
- `opencv-python`

Example install:

```bash
pip install face_recognition opencv-python
```

## Usage

1. Place known student images in a folder (e.g., `data/known`). The student name is derived from the filename (e.g., `alex_smith.jpg`).
2. Place images to scan in another folder (e.g., `data/unknown`).
3. Run the script:

```bash
python surveillance.py \
  --known data/known \
  --unknown data/unknown \
  --tolerance 0.5 \
  --annotate output/annotated \
  --csv output/matches.csv
```

The script reports matches to stdout, writes a CSV if requested, and can optionally save annotated images with bounding boxes.
