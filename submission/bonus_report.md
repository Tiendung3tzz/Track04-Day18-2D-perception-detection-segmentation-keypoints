# Báo cáo bonus — 20 điểm theo rubric

## 1D — AP 101 điểm (5 điểm)

AP ví dụ 10 GT, 15 dự đoán: 0.535007; đã qua check_average_precision(). Tính envelope precision và trung bình trên 101 mức recall; giữ đồ thị PR trong notebook.

## 4C — Hai model, val gốc/val gương (10 điểm)

| Model | Pose mAP50-95 — val gốc | Pose mAP50-95 — val lật gương |
|---|---|---|
| flip_idx giải phẫu | 0.4776 | 0.3785 |
| flip_idx đồng nhất | 0.5414 | 0.3109 |

Model giải phẫu: val gốc 0.4776, val gương 0.3785, chênh gốc−gương 9.91 điểm phần trăm; model đồng nhất: 0.5414 và 0.3109, chênh 23.04 điểm phần trăm. Model đồng nhất giảm nhiều hơn model giải phẫu trên ảnh gương. Metric có thể che lỗi là Pose mAP trên val gốc chỉ gồm hổ quay phải: nó không kiểm tra nhãn trái/phải khi đổi hướng. Pose mAP50 cũng dễ che sai số hơn mAP50-95, và sigma đồng đều 1/12 có thể dung thứ lỗi định vị chân. Cần đọc thêm lỗi từng keypoint và số ca nghi đảo nhãn trên val gương, xây dựng val/test độc lập có cả hai hướng, nhiều tư thế/che khuất, nhãn giải phẫu được kiểm tra; val gương là stress test, không thay thế ảnh quay trái thực tế.

Chẩn đoán theo keypoint trên val gương được lưu ở `bonus_4c_diagnostics.json`.

## Bài tập về nhà — ONNX CPU (5 điểm)

| Cấu hình | conf | N | preprocess (ms) | inference (ms) | postprocess (ms) | wall mean (ms) | wall median (ms) | wall p95 (ms) | số box | output shape |
|---|---|---|---|---|---|---|---|---|---|---|
| one-to-many + NMS | 0.2500 | 30 | 3.8203 | 62.9055 | 1.4580 | 68.6689 | 68.4015 | 71.7347 | 5 | [[1, 84, 8400]] |
| one-to-many + NMS | 0.0010 | 30 | 3.8604 | 63.1658 | 2.2461 | 69.6843 | 69.4614 | 71.5286 | 186 | [[1, 84, 8400]] |
| one-to-one NMS-free | 0.2500 | 30 | 4.0555 | 65.6200 | 0.4759 | 70.5859 | 68.9607 | 78.4264 | 5 | [[1, 300, 6]] |
| one-to-one NMS-free | 0.0010 | 30 | 4.0339 | 101.1884 | 0.5491 | 106.2191 | 105.7093 | 110.1939 | 177 | [[1, 300, 6]] |

Môi trường đo: `{"platform": "Linux-6.18.48+-x86_64-with-glibc2.39", "cpu": "x86_64", "logical_cpus": 4, "onnxruntime": "1.22.1", "providers": ["CPUExecutionProvider"], "input": [1, 3, 640, 640], "precision": "FP32", "warmup": 5, "runs": 30, "image": "bus.jpg", "includes_disk_io": false}`.

Hai graph dùng cùng weights, cùng input 640×640 FP32 batch 1, cùng ảnh đã đọc vào RAM; one-to-many xuất raw output và NMS ngoài graph, one-to-one xuất prediction không NMS. Khởi tạo model và export không tính vào latency; Results.speed đo các giai đoạn, wall-clock đo cả lời gọi predict, gồm thêm overhead của wrapper Python. p95 lấy từ 30 lần đo sau 5 warm-up; conf thay đổi số ứng viên/postprocess chứ không đổi graph.

Ở conf=0.25, wall mean many=68.67 ms, one=70.59 ms; tỷ số many/one=0.973. Postprocess tương ứng 1.46/0.48 ms.

Ở conf=0.001, wall mean many=69.68 ms, one=106.22 ms; tỷ số many/one=0.656. Postprocess tương ứng 2.25/0.55 ms.

NMS-free bỏ bước suppression, nhưng mức lợi phụ thuộc backend/CPU/số ứng viên và nhiễu đo; không suy ra CPU luôn nhanh hơn GPU hay head one-to-one luôn chính xác hơn. Đo thêm ảnh cảnh đông, cố định CPU threads và báo phân phối latency để đánh giá triển khai.
