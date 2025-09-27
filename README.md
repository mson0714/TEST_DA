# 🎮 Gesture Racing Game
*Điều khiển xe đua bằng cử chỉ tay với AI thông minh!*

---

## 🚗 Tổng quan dự án

Game đua xe điều khiển bằng **cử chỉ tay**, sử dụng mô hình **MobileNetV2 + GRU** để nhận diện 5 động tác chính trong thời gian thực. Dự án cũng bao gồm công cụ thu thập dữ liệu mới giúp bạn mở rộng bộ dữ liệu theo đúng bối cảnh sử dụng.

### 🎯 5 Cử chỉ điều khiển:
| Cử chỉ | Biểu tượng | Chức năng |
|--------|------------|-----------|
| Thumbs Up | 👍 | Tăng tốc |
| Thumbs Down | 👎 | Giảm tốc |
| Left Swipe | 👈 | Đánh lái trái |
| Right Swipe | 👉 | Đánh lái phải |
| Stop | ✋ | Phanh khẩn cấp |

---

## 📁 Thành phần chính

- 🏃‍♂️ `run_game_5class.py`: Runner 5 cử chỉ với HUD hiển thị top xác suất và tuỳ chọn MediaPipe để crop bàn tay
- 📊 `run_game_10class.py`: Phiên bản 10 lớp cũ (chưa ổn định, dùng để tham khảo)
- 📹 `capture_gesture_dataset.py`: Script thu thập dữ liệu mới, lưu theo layout `archive/<split>/<split>/<sample>/frame_XXX.png`
- 🎨 `assets/images/`: Sprite cho nền đường đua, xe, HUD
- 🧠 `gesture_model_20250924_102037/`: Mô hình 5 lớp đã train sẵn

---

## ⚙️ Chuẩn bị môi trường

### 1️⃣ Cài đặt Python
```bash
# Yêu cầu Python 3.10 trở lên
python --version
```

### 2️⃣ Cài đặt dependencies
```bash
# Tạo virtual environment (khuyến khích)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Cài đặt packages
pip install -r requirements.txt
```

> 💡 **Packages tối thiểu**: `opencv-python`, `tensorflow==2.13.0`, `pygame`, `numpy`, `mediapipe` (tùy chọn)

---

## 🎮 Chạy game

### 🚀 Khởi động nhanh
```bash
cd gesture_racing_game
python run_game_5class.py --use-mediapipe
```

### ⌨️ Phím điều khiển dự phòng
- `ESC`: Thoát game
- `Q`: Đóng/mở camera feed  
- `WASD` / `Arrow keys`: Điều khiển thủ công

---

## 📹 Thu thập dữ liệu mới

### 🎯 Script thu thập
```bash
python capture_gesture_dataset.py --manifest ../archive/new_train.csv --use-mediapipe
```

### 📝 Quy trình thu thập
1. 🎥 **Chuẩn bị**: Đặt camera và màn hình ở bối cảnh thực tế
2. ⌨️ **Ghi hình**: Nhấn phím 1-5 tương ứng với từng cử chỉ
3. ⏰ **Đếm ngược**: Script tự động đếm ngược và ghi 30 frame
4. 💾 **Lưu trữ**: Dữ liệu lưu vào `archive/train/train/` với tên `REC_YYYYMMDD_HH_MM_SS_<label>_context`
5. 📊 **Manifest**: Tự động append vào CSV với format `sample_folder;label_slug;label_index`

### 💡 Mẹo thu dữ liệu chất lượng
- ✅ Giữ tay trong khung hình suốt 30 frame
- 🌈 Thu ở nhiều điều kiện ánh sáng khác nhau  
- 👥 Ghi với nhiều người để tăng đa dạng
- 🎮 Thu trong môi trường game thật để model quen với HUD

---

## 🔄 Huấn luyện lại model

### 1️⃣ Chuẩn bị dữ liệu
```bash
# Gộp dữ liệu mới vào train.csv
type archive\new_train.csv >> archive\train.csv
```

### 2️⃣ Huấn luyện
- 📓 Mở notebook `gesture_recognition_minimal.ipynb`
- ▶️ Chạy tuần tự các cell
- ⚙️ Config: MobileNetV2 + GRU, batch size 12, 35 epochs

### 3️⃣ Deploy model mới
```bash
# Copy model mới vào thư mục game
copy gesture_model_20250927_005401\best_model.h5 gesture_model_20250924_102037\
```

---

## 🔧 Troubleshooting

| ❌ Vấn đề | 💡 Giải pháp |
|-----------|--------------|
| Model không load | Kiểm tra đường dẫn `.h5` và TensorFlow version |
| Game bị lag | Tắt camera HUD (`Q`), giảm resolution, hoặc bật GPU |
| Cử chỉ trái/phải nhầm lẫn | Thu thêm data đa dạng góc nhìn + smoothing |

---

## 📊 Kết quả hiện tại

- 🎯 **Độ chính xác**: ~97.7% trên validation set
- ⚡ **Tốc độ**: Real-time trên GPU RTX 3050 Ti
- 🎮 **Trải nghiệm**: Mượt mà với MediaPipe preprocessing

---

## 🚀 Hướng phát triển

- [ ] 🎪 Thêm cử chỉ mới (giữ lái, nitro)
- [ ] 🎵 Tích hợp âm thanh phản hồi
- [ ] 🏁 Nhiều màn chơi & điểm số
- [ ] 👥 Chế độ 2 người chơi
- [ ] 🥽 Hỗ trợ VR/AR

---
# 🧑‍💻 Nhóm phát triển
- Thành viên DTU-K28HP-TBM 👨‍👩‍👧‍👦
---

**🎉 Chúc bạn chơi game và thu thập dữ liệu vui vẻ!**

---
*⭐ Star repo này nếu bạn thấy hữu ích! | 📧 Báo lỗi qua Issues*