# Gesture Racing Game 🎮

Đề tài: xây dựng trò chơi đua xe điều khiển hoàn toàn bằng cử chỉ tay qua
webcam. Pipeline gồm:

- **Nhận diện thời gian thực**: MobileNetV2 trích xuất đặc trưng từng frame,
   GRU xử lý chuỗi 30 frame để nhận diện 5 cử chỉ (Thumbs Up/Down, Left/Right
   Swipe, Stop).
- **Tiền xử lý tuỳ chọn**: MediaPipe Hands để crop bàn tay trước khi đưa vào
   model, giúp ổn định trong môi trường thực tế.
- **Thu thập & fine-tune nhanh**: Script hỗ trợ ghi thêm dữ liệu đúng bối cảnh
   và notebook train lại mô hình trong vài giờ trên GPU laptop.

## 🧱 Thành phần chính

- `run_game_5class.py`: Game runner 5 lớp với HUD hiển thị top-3 xác suất và
   tuỳ chọn `--use-mediapipe`.
- `capture_gesture_dataset.py`: Công cụ ghi dữ liệu mới, lưu đúng cấu trúc
   `archive/<split>/<split>/<sample>` và append vào CSV manifest.
- `gesture_model_20250924_102037/`: Thư mục mô hình mà game nạp khi khởi động
   (đã được cập nhật bằng bản fine-tune ngày 27/09/2025).
- `gesture_model_20250927_005401/`: Checkpoint mới nhất (best/final + lịch sử
   train) – dùng để deploy hoặc tham khảo khi train lại.
- `assets/images/`: Sprite đường đua, xe, HUD.

## ⚙️ Chuẩn bị môi trường

1. Cài Python 3.10.
2. Cài các gói cần thiết (khuyến khích dùng virtualenv):
    ```cmd
    pip install -r requirements.txt
    ```
    > Tối thiểu để chạy game: `tensorflow==2.13.0`, `opencv-python`, `pygame`,
    > `numpy`, và `mediapipe` (nếu muốn crop tay).
3. Đảm bảo mô hình `.h5` mới nhất nằm trong thư mục gốc `gesture_model_20250924_102037/`.

## 🚗 Chạy game nhanh

```cmd
cd gesture_racing_game
python run_game_5class.py --use-mediapipe
```

Nếu chưa muốn dùng MediaPipe, bỏ cờ `--use-mediapipe`. Điều khiển:

- 👍 Thumbs Up → tăng tốc
- 👎 Thumbs Down → giảm tốc
- 👈 Left Swipe → rẽ trái
- 👉 Right Swipe → rẽ phải
- ✋ Stop → phanh khẩn cấp

Phím `ESC` thoát game, `Q` đóng feed camera, WASD/mũi tên hoạt động như dự
phòng. HUD góc trái hiển thị xác suất hiện tại để bạn debug nhanh các cử chỉ.

## 📹 Thu thêm dữ liệu đúng bối cảnh

```cmd
python capture_gesture_dataset.py --manifest ..\archive\new_train.csv --use-mediapipe
```

- Nhấn phím 1–5 để ghi từng cử chỉ; script sẽ đếm ngược rồi lưu 30 frame vào
   `archive/<split>/<split>/REC_YYYYMMDD_HH_MM_SS_<label>_<context>`.
- Manifest CSV (nếu bật) sẽ append dòng `sample;label;label_index`, giúp bạn
   merge nhanh vào `train.csv` / `val.csv` trước khi train.
- Mẹo: giữ tay trong crop, mô phỏng đúng khoảng cách khi chơi game, thu nhiều
   người & ánh sáng để phân biệt Left/Right tốt hơn.

## 🧪 Fine-tune mô hình

1. Gộp các sample mới vào `archive/train.csv` (và `val.csv` nếu có) – đã có sẵn
    snippet merge trong notebook.
2. Mở `gesture_recognition_minimal.ipynb`, chạy tuần tự các cell (đã cấu hình
    MobileNetV2 fine-tune 24 layer cuối, batch size 12, 35 epoch, augmentation
    nhẹ).
3. Sau khi train xong, copy `best_model.h5` hoặc `final_gesture_model.h5` mới
    vào thư mục gốc `gesture_model_20250924_102037/` rồi chạy game để kiểm tra.

## 🔍 Troubleshooting

- **Model không load** → Kiểm tra đường dẫn `gesture_model_20250924_102037/*.h5`
   và phiên bản TensorFlow.
- **Game lag** → Tắt HUD camera (`Q`), giảm camera resolution trong code, hoặc
   dùng GPU.
- **Left/Right còn nhầm** → Thu thêm dữ liệu có biên độ trái-phải lớn, giữ tay
   trong crop MediaPipe, hoặc bổ sung smoothing xác suất trong runner.

Chúc bạn điều khiển đường đua bằng cử chỉ thật mượt và mở rộng dataset thành
công! 🚀