"""School surveillance face check utility.

Loads known student images, encodes faces, then scans incoming images and
reports matches. Requires `face_recognition` (dlib) and `opencv-python`.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import face_recognition


@dataclass(frozen=True)
class MatchResult:
    image_path: Path
    student_name: str
    distance: float


def iter_image_paths(folder: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        yield from folder.rglob(ext)


def load_known_students(known_dir: Path) -> Tuple[List[str], List[List[float]]]:
    """Load known student face encodings.

    Expects images under known_dir. The student name is derived from the
    image filename stem (e.g., `alex_smith.jpg` -> `alex_smith`).
    """
    names: List[str] = []
    encodings: List[List[float]] = []

    for image_path in iter_image_paths(known_dir):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            logging.warning("No faces found in %s", image_path)
            continue
        if len(face_locations) > 1:
            logging.warning("Multiple faces in %s; using first", image_path)
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        names.append(image_path.stem)
        encodings.append(face_encoding)

    return names, encodings


def match_faces(
    known_names: List[str],
    known_encodings: List[List[float]],
    target_image: Path,
    tolerance: float,
) -> List[MatchResult]:
    image = face_recognition.load_image_file(target_image)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    matches: List[MatchResult] = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) == 0:
            continue
        best_index = int(distances.argmin())
        best_distance = float(distances[best_index])
        if best_distance <= tolerance:
            matches.append(
                MatchResult(
                    image_path=target_image,
                    student_name=known_names[best_index],
                    distance=best_distance,
                )
            )
    return matches


def annotate_image(image_path: Path, matches: List[MatchResult], output_dir: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        logging.warning("Could not read image %s", image_path)
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)

    for (top, right, bottom, left), match in zip(face_locations, matches):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{match.student_name} ({match.distance:.2f})"
        cv2.putText(
            image,
            label,
            (left, max(top - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    output_path = output_dir / image_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def write_csv(results: List[MatchResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "student", "distance"])
        for match in results:
            writer.writerow([match.image_path.name, match.student_name, f"{match.distance:.4f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="School surveillance face check.")
    parser.add_argument("--known", type=Path, required=True, help="Folder with known student images")
    parser.add_argument("--unknown", type=Path, required=True, help="Folder with images to scan")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Match tolerance (lower = stricter)")
    parser.add_argument("--annotate", type=Path, help="Folder to write annotated images")
    parser.add_argument("--csv", type=Path, help="Write results to CSV file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    known_names, known_encodings = load_known_students(args.known)
    if not known_names:
        raise SystemExit("No known student faces found. Check --known directory.")

    results: List[MatchResult] = []
    for image_path in iter_image_paths(args.unknown):
        matches = match_faces(known_names, known_encodings, image_path, args.tolerance)
        results.extend(matches)
        if args.annotate and matches:
            annotate_image(image_path, matches, args.annotate)

    if args.csv:
        write_csv(results, args.csv)

    if results:
        for match in results:
            logging.info("%s -> %s (%.3f)", match.image_path.name, match.student_name, match.distance)
    else:
        logging.info("No matches found.")


if __name__ == "__main__":
    main()
