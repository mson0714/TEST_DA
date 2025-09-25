"""
Gesture racing game runner for 5-class gesture model with optional MediaPipe hand
cropping support and sprite-based visuals.
"""

from __future__ import annotations

import argparse
import os
import time
import threading
from collections import deque

import cv2
import numpy as np
import pygame
import tensorflow as tf

try:  # MediaPipe is optional
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

# Silence TensorFlow logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_5 = os.path.normpath(os.path.join(BASE_DIR, "..", "gesture_model_20250924_102037"))
MODEL_CANDIDATES_5 = [
    os.path.join(MODEL_DIR_5, "final_gesture_model.h5"),
    os.path.join(MODEL_DIR_5, "best_model.h5"),
]
ASSET_DIR = os.path.join(BASE_DIR, "assets", "images")

GESTURE_CLASSES = {
    0: "Thumbs Up",
    1: "Thumbs Down",
    2: "Left Swipe",
    3: "Right Swipe",
    4: "Stop",
}

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 60
ROAD_LEFT, ROAD_RIGHT = 120, SCREEN_WIDTH - 120
CAR_WIDTH, CAR_HEIGHT = 64, 110

HAS_MEDIAPIPE = mp is not None


def _load_image(name: str, size: tuple[int, int] | None = None) -> pygame.Surface:
    path = os.path.join(ASSET_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy asset: {path}")
    image = pygame.image.load(path)
    image = image.convert_alpha()
    if size is not None:
        image = pygame.transform.smoothscale(image, size)
    return image


# -----------------------------------------------------------------------------
# Hand preprocessing with MediaPipe
# -----------------------------------------------------------------------------
class HandPreprocessor:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled and HAS_MEDIAPIPE
        self._hands = None
        self._last_bbox: tuple[int, int, int, int] | None = None
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
            self._last_bbox = None
            return frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            self._last_bbox = None
            return frame

        h, w, _ = frame.shape
        landmark = results.multi_hand_landmarks[0]
        xs = [pt.x for pt in landmark.landmark]
        ys = [pt.y for pt in landmark.landmark]

        min_x = max(0.0, min(xs))
        max_x = min(1.0, max(xs))
        min_y = max(0.0, min(ys))
        max_y = min(1.0, max(ys))

        if max_x - min_x < 0.02 or max_y - min_y < 0.02:
            self._last_bbox = None
            return frame

        margin = 0.4
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        half = max(max_x - min_x, max_y - min_y) / 2 + margin

        x1 = int(max(0, (cx - half) * w))
        y1 = int(max(0, (cy - half) * h))
        x2 = int(min(w, (cx + half) * w))
        y2 = int(min(h, (cy + half) * h))

        if x2 <= x1 or y2 <= y1:
            self._last_bbox = None
            return frame

        self._last_bbox = (x1, y1, x2, y2)
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame
        return cropped

    def annotate(self, frame: np.ndarray) -> None:
        if not self.enabled or self._last_bbox is None:
            return
        x1, y1, x2, y2 = self._last_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)


# -----------------------------------------------------------------------------
# Gesture recogniser
# -----------------------------------------------------------------------------
class GestureRecognizer5:
    def __init__(self, model_path: str, use_mediapipe: bool = False):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.cap: cv2.VideoCapture | None = None
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=30)
        self.current_gesture = 4
        self.confidence = 0.0
        self.running = False
        self.hand_preprocessor = HandPreprocessor(enabled=use_mediapipe)
        self.last_probs = np.zeros(len(GESTURE_CLASSES), dtype=float)
        if use_mediapipe and not HAS_MEDIAPIPE:
            print("⚠️ MediaPipe chưa được cài. Sử dụng chế độ chuẩn hoá mặc định.")

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        return True

    def stop(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.hand_preprocessor.close()

    @staticmethod
    def _normalise(frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (224, 224))
        frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame.astype(np.float32))
        return frame

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        cropped = self.hand_preprocessor.extract(frame)
        return self._normalise(cropped)

    def _predict(self) -> None:
        if not self.frame_buffer:
            return

        frames = list(self.frame_buffer)
        if len(frames) < self.frame_buffer.maxlen:
            last = frames[-1]
            frames.extend([last] * (self.frame_buffer.maxlen - len(frames)))

        sequence = np.expand_dims(np.array(frames), axis=0)
        logits = self.model.predict(sequence, verbose=0)[0]
        self.last_probs = logits
        idx = int(np.argmax(logits))
        conf = float(np.max(logits))
        if conf >= 0.30:
            self.current_gesture = idx
            self.confidence = conf

    def loop(self) -> None:
        while self.running:
            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            processed = self._preprocess(frame)
            self.frame_buffer.append(processed)
            self._predict()

            self.hand_preprocessor.annotate(frame)

            cv2.putText(frame, f"Gesture: {GESTURE_CLASSES.get(self.current_gesture, 'Unknown')}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {self.confidence:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("5-Class Gesture Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break
            time.sleep(0.01)

    @property
    def mediapipe_active(self) -> bool:
        return self.hand_preprocessor.enabled


# -----------------------------------------------------------------------------
# Car & game logic
# -----------------------------------------------------------------------------
class Car:
    def __init__(self, sprite: pygame.Surface | None = None):
        self.sprite = sprite
        self.width = sprite.get_width() if sprite is not None else CAR_WIDTH
        self.height = sprite.get_height() if sprite is not None else CAR_HEIGHT
        self.x = float((ROAD_LEFT + ROAD_RIGHT - self.width) / 2)
        self.y = float(SCREEN_HEIGHT - self.height - 40)
        self.target_x = self.x
        self.speed = 4.5
        self.target_speed = 4.5

    def update_gesture(self, gesture_id: int) -> None:
        if gesture_id == 0:  # Thumbs Up
            self.target_speed = min(11.0, self.target_speed + 0.35)
        elif gesture_id == 1:  # Thumbs Down
            self.target_speed = max(1.2, self.target_speed - 0.4)
        elif gesture_id == 2:  # Left Swipe
            self.target_x = max(ROAD_LEFT, self.target_x - 140)
        elif gesture_id == 3:  # Right Swipe
            self.target_x = min(ROAD_RIGHT - self.width, self.target_x + 140)
        elif gesture_id == 4:  # Stop
            self.target_speed = max(1.2, self.target_speed * 0.92)

    def update_keyboard(self, keys) -> None:
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.target_x = max(ROAD_LEFT, self.target_x - 6)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.target_x = min(ROAD_RIGHT - self.width, self.target_x + 6)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.target_speed = min(12.0, self.target_speed + 0.3)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.target_speed = max(1.2, self.target_speed - 0.3)

    def tick(self) -> None:
        self.x += (self.target_x - self.x) * 0.18
        self.speed += (self.target_speed - self.speed) * 0.12

    def draw(self, surface: pygame.Surface) -> None:
        if self.sprite is not None:
            surface.blit(self.sprite, (int(self.x), int(self.y)))
        else:
            rect = pygame.Rect(int(self.x), int(self.y), self.width, self.height)
            pygame.draw.rect(surface, (200, 40, 60), rect)
            pygame.draw.rect(surface, (255, 255, 255), rect, 3)


# -----------------------------------------------------------------------------
# Game wrapper
# -----------------------------------------------------------------------------
class GestureGame5:
    def __init__(self, model_path: str, use_mediapipe: bool = False):
        pygame.init()
        pygame.display.set_caption("Gesture Racing - 5 Classes")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)

        def try_load(name: str, size: tuple[int, int] | None = None):
            try:
                return _load_image(name, size)
            except FileNotFoundError as exc:
                print(f"⚠️ {exc}")
                return None

        icon_surface = try_load("icon.png")
        if icon_surface is not None:
            pygame.display.set_icon(icon_surface)

        self.background = try_load("background.png", (SCREEN_WIDTH, SCREEN_HEIGHT))
        car_sprite = try_load("car.png", (CAR_WIDTH, CAR_HEIGHT))

        self.car = Car(sprite=car_sprite)
        self.bg_offset = 0.0
        self.hud_panel = pygame.Surface((280, 170), pygame.SRCALPHA)
        self.hud_panel.fill((12, 12, 12, 170))

        self.score = 0.0
        self.recognizer = GestureRecognizer5(model_path, use_mediapipe=use_mediapipe)
        self.camera_thread: threading.Thread | None = None

    def start_gesture_loop(self) -> None:
        if not self.recognizer.start():
            print("⚠️ Không mở được camera. Sẽ dùng bàn phím.")
            return
        self.camera_thread = threading.Thread(target=self.recognizer.loop, daemon=True)
        self.camera_thread.start()

    def shutdown(self) -> None:
        self.recognizer.stop()
        pygame.quit()

    def run(self) -> None:
        print(f"🎮 Chạy game với model: {os.path.basename(self.recognizer.model_path)}")
        print(f"📷 MediaPipe hỗ trợ: {'BẬT' if self.recognizer.mediapipe_active else 'TẮT'}")
        self.start_gesture_loop()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            if self.recognizer.running and self.recognizer.model is not None:
                self.car.update_gesture(self.recognizer.current_gesture)
            if any(keys[k] for k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN,
                                     pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s)):
                self.car.update_keyboard(keys)

            self.car.tick()
            self.score += self.car.speed * 0.08

            self._draw_playfield()
            self.car.draw(self.screen)
            self._draw_hud()

            pygame.display.flip()
            self.clock.tick(FPS)

        self.shutdown()

    def _draw_hud(self) -> None:
        gesture_text = GESTURE_CLASSES.get(self.recognizer.current_gesture, "Unknown")
        hud_pos = (15, 15)
        self.screen.blit(self.hud_panel, hud_pos)

        probs = self.recognizer.last_probs
        if probs.size:
            top_indices = np.argsort(probs)[::-1][:3]
            prob_lines = [
                f"{GESTURE_CLASSES[idx]}: {probs[idx] * 100:.0f}%"
                for idx in top_indices
            ]
        else:
            prob_lines = ["Chưa có dự đoán"]

        entries = [
            f"Score: {int(self.score)}",
            f"Speed: {self.car.speed:.1f}",
            f"Gesture: {gesture_text}",
            f"Confidence: {self.recognizer.confidence:.2f}",
            f"MediaPipe: {'ON' if self.recognizer.mediapipe_active else 'OFF'}",
            "ESC để thoát, Q để đóng camera",
        ] + prob_lines

        for idx, text in enumerate(entries):
            surface = self.font.render(text, True, (245, 245, 245))
            self.screen.blit(surface, (hud_pos[0] + 16, hud_pos[1] + 16 + idx * 26))

    def _draw_playfield(self) -> None:
        self.bg_offset = (self.bg_offset + self.car.speed * 3.2) % SCREEN_HEIGHT
        if self.background is not None:
            first_y = -self.bg_offset
            self.screen.blit(self.background, (0, int(first_y)))
            self.screen.blit(self.background, (0, int(first_y + SCREEN_HEIGHT)))
        else:
            self.screen.fill((18, 18, 18))
            pygame.draw.rect(self.screen, (90, 90, 90), (ROAD_LEFT, 0, ROAD_RIGHT - ROAD_LEFT, SCREEN_HEIGHT))

        pygame.draw.line(self.screen, (220, 220, 220), (ROAD_LEFT, 0), (ROAD_LEFT, SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, (220, 220, 220), (ROAD_RIGHT, 0), (ROAD_RIGHT, SCREEN_HEIGHT), 3)

        center_lane = (ROAD_LEFT + ROAD_RIGHT) // 2
        dash_height = 40
        gap = 30
        lane_offset = int(self.bg_offset) % (dash_height + gap)
        y = -lane_offset
        while y < SCREEN_HEIGHT + dash_height:
            pygame.draw.rect(self.screen, (255, 255, 255), (center_lane - 4, y, 8, dash_height))
            y += dash_height + gap


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def _select_model() -> str:
    for candidate in MODEL_CANDIDATES_5:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy file model cho bản 5 lớp. Hãy đặt final_gesture_model.h5 hoặc best_model.h5 trong gesture_model_20250924_102037"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gesture Racing - 5 Class Runner")
    parser.add_argument(
        "--use-mediapipe",
        action="store_true",
        help="Dùng MediaPipe Hands để crop bàn tay trước khi dự đoán (cần cài mediapipe).",
    )
    args = parser.parse_args()

    if args.use_mediapipe and not HAS_MEDIAPIPE:
        print("⚠️ MediaPipe chưa được cài đặt. Chạy pip install mediapipe==0.10.14 để bật tính năng này.")

    model_path = _select_model()
    game = GestureGame5(model_path, use_mediapipe=args.use_mediapipe)
    try:
        game.run()
    finally:
        game.shutdown()


if __name__ == "__main__":
    main()
