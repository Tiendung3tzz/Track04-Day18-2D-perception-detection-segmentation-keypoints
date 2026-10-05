# Chạy bản notebook đã hoàn thiện

Mở `lab_2d_perception_student.ipynb` bằng Jupyter, VS Code, Colab hoặc Kaggle. Notebook đã có code cho toàn bộ TODO, Q1–Q12 và cả 20 điểm bonus theo rubric. Kết quả thực nghiệm được tạo khi chạy.

## Chạy từ đầu

1. Dùng GPU NVIDIA và PyTorch CUDA, hoặc bật GPU trên Kaggle/Colab. Bật Internet để cài package và tải model/dataset.
2. Restart kernel/session, rồi **Run All**. Ô đầu cài `ultralytics==8.4.171` và dependency ONNX. Nếu đã import phiên bản Ultralytics khác trước khi cài, restart rồi chạy lại từ đầu.
3. Bài chính train model giải phẫu **40 epoch, imgsz=640, batch=16, seed=0**. Phần 4C train thêm model đồng nhất với cùng cấu hình để so sánh công bằng. Notebook dừng trước train nếu không có GPU.
4. Cờ `RUN_4C`, `TRAIN_IDENTITY` và `RUN_ONNX_BONUS` đều bật sẵn. Phần ONNX đo CPU dù phần huấn luyện dùng GPU. Đầu vào benchmark cố định 640×640 FP32, batch 1, 5 warm-up và 30 lần đo cho mỗi cấu hình.
5. Nếu hết bộ nhớ GPU, đổi batch 16 thành 8 trong các ô train/val của **cả hai model**, rồi restart và chạy lại. `workers=0` phù hợp Jupyter trên Windows.

## Các kết quả sẽ được tạo

```text
submission/
├── ket_qua.json
├── autolabel/bus.txt
├── report.md
├── bonus_report.md
├── bonus_4c_diagnostics.json
├── onnx_cpu_latency.json
└── diagnostics/
    ├── worst6.png
    ├── keypoint_errors.png
    └── error_analysis.json
submission.zip
```

Trên Kaggle, các file nằm trong `/kaggle/working`; local/Colab dùng thư mục làm việc của notebook. Model ONNX ở `bonus_exports/`, weights pose ở `runs_lab/`; không cần đưa các file model lớn lên repo nộp bài.

## Kiểm tra trước khi nộp

- Checklist cuối phải đạt các tiêu chí bắt buộc, AP, hai model của 4C và bốn cấu hình ONNX CPU. Các `assert` cuối sẽ chỉ ra nếu thiếu kết quả.
- Q2, Q4, Q6–Q11 lấy số liệu từ lần chạy. **Xem sáu ảnh và đối chiếu Q11**: câu trả lời dẫn tên ảnh, tọa độ và sai số thực tế; chỉ thêm kết luận che khuất/cắt mép khi nhìn thấy hiện tượng đó trên ảnh.
- Giữ toàn bộ output khi lưu notebook. Chưa chạy GPU thì chưa có bằng chứng đạt yêu cầu fine-tune hay điểm thực nghiệm.
- Đẩy notebook đã chạy và thư mục `submission/` lên GitHub public. Mở URL file notebook trong repo của bạn, điền vào `NOTEBOOK_URL` ở ô báo cáo rồi chạy lại ô báo cáo và ô cuối, hoặc bổ sung trực tiếp dòng `Link notebook đã chạy:` trong `submission/report.md` sau khi giải nén.
- Nộp **URL repo public** lên LMS, theo README/rubric. Không mở PR, không tạo thư mục lồng `submission/submission/`.

## Nội dung bonus

| Mục | Bằng chứng sau khi chạy | Điểm theo rubric |
|---|---|---:|
| 1D | `average_precision` qua bộ check, đồ thị PR và AP ví dụ | 5 |
| 4C | Hai model trên val gốc/val gương, giải thích metric che lỗi, chẩn đoán keypoint | 10 |
| Bài tập về nhà ONNX | Hai graph/head riêng, latency CPU ở conf 0.25 và 0.001, báo cáo ngắn | 5 |

Các mục có số liệu và giải thích nằm trong `submission/bonus_report.md`; file được tạo từ kết quả chạy, không dùng mAP/latency mẫu trong README làm số đo của bạn.
