"""Utility for collecting context-specific gesture sequences with a webcam.

This tool mirrors the naming/layout of the existing dataset in ``archive/`` so
new clips can be blended into the training CSVs. Each capture produces a folder
containing ``N`` PNG frames (default: 30) plus an optional manifest entry.

Usage examples
--------------
Record new samples for the five in-game gestures while keeping the default
layout (``archive/train/train``) and appending to a manifest ``new_train.csv``::

    python capture_gesture_dataset.py --manifest ../archive/new_train.csv

Capture sequences with MediaPipe hand crops and store them in a separate folder::

    python capture_gesture_dataset.py --output-root ../archive/custom --use-mediapipe

Press the number associated with a gesture to trigger a countdown and
recording. Press ``q`` to quit.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

try:  # MediaPipe remains optional
    import mediapipe as mp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    mp = None


DEFAULT_GESTURES = [
    "Thumbs Up",
    "Thumbs Down",
    "Left Swipe",
    "Right Swipe",
    "Stop",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    return slug.strip("_").lower()


@dataclass
class CaptureConfig:
    gestures: list[str]
    sequence_length: int
    fps: float
    output_root: Path
    split: str
    prefix: str
    context: str
    prepare_seconds: float
    cooldown_seconds: float
    mirror: bool
    use_mediapipe: bool
    manifest: Optional[Path]

    @property
    def split_dir(self) -> Path:
        # Mirror the historical layout (e.g. archive/train/train)
        return self.output_root / self.split / self.split


class HandPreprocessor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and mp is not None
        self._hands = None
        if self.enabled:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.45,
            )

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def extract(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or self._hands is None:
            return frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return frame

        h, w, _ = frame.shape
        xs = [landmark.x for landmark in results.multi_hand_landmarks[0].landmark]
        ys = [landmark.y for landmark in results.multi_hand_landmarks[0].landmark]

        min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
        min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
        if max_x - min_x < 0.02 or max_y - min_y < 0.02:
            return frame

        margin = 0.4
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        half = max(max_x - min_x, max_y - min_y) / 2 + margin

        x1 = max(0, int((cx - half) * w))
        y1 = max(0, int((cy - half) * h))
        x2 = min(w, int((cx + half) * w))
        y2 = min(h, int((cy + half) * h))
        if x2 <= x1 or y2 <= y1:
            return frame

        cropped = frame[y1:y2, x1:x2]
        return cropped if cropped.size else frame


def parse_args(argv: Optional[Iterable[str]] = None) -> CaptureConfig:
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir.parent / "archive"

    parser = argparse.ArgumentParser(description="Capture gesture frame sequences")
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=DEFAULT_GESTURES,
        help="Ordered gesture list; indices map to dataset labels",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Frames per capture",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target capture FPS (used for pacing saves)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output,
        help="Base directory that contains train/ and val/ folders",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to capture into",
    )
    parser.add_argument(
        "--prefix",
        default="REC",
        help="Prefix for sample folder names",
    )
    parser.add_argument(
        "--context",
        default="context",
        help="Short scene descriptor appended to folder names",
    )
    parser.add_argument(
        "--prepare-seconds",
        type=float,
        default=2.5,
        help="Countdown before recording each clip",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=1.0,
        help="Cooldown interval after each recording",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror frames horizontally",
    )
    parser.add_argument(
        "--use-mediapipe",
        action="store_true",
        help="Crop around the detected hand using MediaPipe Hands",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional CSV manifest to append with sample metadata",
    )

    args = parser.parse_args(argv)

    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    config = CaptureConfig(
        gestures=args.gestures,
        sequence_length=args.sequence_length,
        fps=args.fps,
        output_root=args.output_root,
        split=args.split,
        prefix=args.prefix,
        context=args.context,
        prepare_seconds=args.prepare_seconds,
        cooldown_seconds=args.cooldown_seconds,
        mirror=args.mirror,
        use_mediapipe=args.use_mediapipe,
        manifest=args.manifest,
    )
    return config


def ensure_directories(config: CaptureConfig) -> None:
    config.split_dir.mkdir(parents=True, exist_ok=True)
    if config.manifest:
        config.manifest.parent.mkdir(parents=True, exist_ok=True)
        if not config.manifest.exists():
            with config.manifest.open("w", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["sample", "label", "label_index"])


def build_keymap(gestures: list[str]) -> dict[int, tuple[int, str]]:
    keymap: dict[int, tuple[int, str]] = {}
    for idx, gesture in enumerate(gestures):
        key = ord(str((idx + 1) % 10)) if idx < 9 else ord(chr(ord("a") + idx - 9))
        keymap[key] = (idx, gesture)
    return keymap


def draw_overlay(
    frame: np.ndarray,
    text_lines: list[str],
    color: tuple[int, int, int] = (0, 200, 255),
    y_start: int = 24,
    line_height: int = 24,
) -> None:
    for i, line in enumerate(text_lines):
        y = y_start + i * line_height
        cv2.putText(frame, line, (16, y), FONT, 0.6, color, 2, cv2.LINE_AA)


def create_sample_folder(config: CaptureConfig, gesture: str, gesture_index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")[:-3]
    slug = slugify(gesture)
    folder = f"{config.prefix}_{timestamp}_{slug}_{config.context}"
    sample_dir = config.split_dir / folder
    if sample_dir.exists():
        counter = 1
        while True:
            candidate = config.split_dir / f"{folder}_{counter:02d}"
            if not candidate.exists():
                sample_dir = candidate
                break
            counter += 1
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def append_manifest(config: CaptureConfig, sample_dir: Path, gesture: str, label_index: int) -> None:
    if not config.manifest:
        return
    with config.manifest.open("a", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([sample_dir.name, slugify(gesture), label_index])


def capture_sequence(
    cap: cv2.VideoCapture,
    config: CaptureConfig,
    preprocessor: HandPreprocessor,
    sample_dir: Path,
) -> None:
    frame_count = 0
    interval = 1.0 / config.fps
    last_save = 0.0
    while frame_count < config.sequence_length:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        if config.mirror:
            frame = cv2.flip(frame, 1)
        processed = preprocessor.extract(frame)
        file_name = f"frame_{frame_count:03d}.png"
        file_path = sample_dir / file_name
        frame_to_save = processed if processed is not None else frame
        cv2.imwrite(str(file_path), frame_to_save)
        frame_count += 1
        now = time.time()
        if frame_count == 1:
            last_save = now
        else:
            sleep_time = interval - (now - last_save)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_save = time.time()


def run_capture(config: CaptureConfig) -> None:
    if config.use_mediapipe and mp is None:
        print("⚠️ MediaPipe chưa được cài đặt. Tiếp tục với khung hình gốc.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam (index 0)")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    preprocessor = HandPreprocessor(config.use_mediapipe)
    keymap = build_keymap(config.gestures)

    state = "idle"
    target_gesture: Optional[str] = None
    target_index: Optional[int] = None
    countdown_deadline = 0.0
    cooldown_deadline = 0.0
    sample_dir: Optional[Path] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ Camera trả về frame rỗng. Dừng.")
                break

            if config.mirror:
                frame = cv2.flip(frame, 1)

            overlay_lines = ["Press gesture key to capture", "q: quit"]
            for raw_key, (idx, gesture) in keymap.items():
                key_label = chr(raw_key).upper()
                overlay_lines.append(f"[{key_label}] {gesture}")

            if state == "idle":
                draw_overlay(frame, overlay_lines)
            elif state == "countdown":
                remaining = max(0.0, countdown_deadline - time.time())
                draw_overlay(
                    frame,
                    [f"Chuẩn bị: {target_gesture}", f"Bắt đầu sau {remaining:0.1f}s"],
                    color=(0, 255, 0),
                    y_start=frame.shape[0] - 80,
                )
                if remaining <= 0:
                    state = "recording"
                    sample_dir = create_sample_folder(config, target_gesture, target_index)
                    print(f"→ Ghi hình {target_gesture} vào {sample_dir.relative_to(config.output_root)}")
                    capture_sequence(cap, config, preprocessor, sample_dir)
                    append_manifest(config, sample_dir, target_gesture, target_index)
                    state = "cooldown"
                    cooldown_deadline = time.time() + config.cooldown_seconds
                    preprocessor.extract(frame)  # refresh bounding box
            elif state == "cooldown":
                remaining = max(0.0, cooldown_deadline - time.time())
                draw_overlay(
                    frame,
                    [f"Đang nghỉ {remaining:0.1f}s"],
                    color=(255, 170, 0),
                    y_start=frame.shape[0] - 40,
                )
                if remaining <= 0:
                    state = "idle"
                    target_gesture = None
                    target_index = None
                    sample_dir = None

            cv2.imshow("Gesture Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if state == "idle" and key in keymap:
                target_index, target_gesture = keymap[key]
                countdown_deadline = time.time() + config.prepare_seconds
                state = "countdown"

    finally:
        cap.release()
        preprocessor.close()
        cv2.destroyAllWindows()


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        config = parse_args(argv)
        ensure_directories(config)
        run_capture(config)
        return 0
    except KeyboardInterrupt:
        print("\nBị ngắt bởi người dùng.")
        return 130
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
