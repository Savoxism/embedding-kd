# HeatGeo MPS FP32 Stability Design

## Bối cảnh và nguyên nhân

HeatGeo dừng ở `epoch=0, step=2` vì student checkpoint
`jim12345/MiniLMv2-L6-H384-distilled-from-BERT-Base` được Transformers load
toàn bộ ở `torch.float16`. Sau optimizer update thứ hai trên Apple MPS, 101
tensor tham số của student chứa `NaN` hoặc `Inf`.

Các artifact đầu vào không phải nguyên nhân: teacher embedding cache,
`teacher_probs`, `candidate_indices`, và `spectral_coords` đều hữu hạn. Một
smoke test chỉ thay đổi student sang `torch.float32` đã hoàn thành năm optimizer
steps với loss hữu hạn và không làm hỏng tham số hay optimizer state.

AdamW cập nhật tham số theo:

\[
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}.
\]

Với FP16, giá trị mặc định \(\epsilon=10^{-8}\) có thể underflow về \(0\).
Điều này làm phép chia mất ổn định khi \(\hat v_t\) nhỏ.

## Thiết kế được chọn

HeatGeo sẽ load và train student bằng FP32 trên MPS. Config khai báo rõ
`student_dtype = "float32"`; model loader chuyển giá trị config thành
`torch.float32` và truyền dtype khi gọi `AutoModel.from_pretrained`.

Sau khi model được load, code kiểm tra:

- Student có đúng dtype theo config.
- Tất cả student parameters đều hữu hạn.
- Dtype thực tế được in ra terminal để reproduction log ghi nhận precision.

Teacher precision và các cache hiện tại không thay đổi.

## Chẩn đoán lỗi hữu hạn

HeatGeo criterion sẽ kiểm tra các đầu vào và từng loss con:

- Student anchor embeddings.
- Student candidate embeddings.
- Teacher probabilities.
- Teacher anchor embeddings.
- Task loss.
- Diffusion loss.
- Anchor loss.
- Spectral loss khi được kích hoạt.
- Total loss.

Nếu một tensor không hữu hạn, exception sẽ nêu tên tensor hoặc loss, số lượng
`NaN`, số lượng `Inf`, shape và dtype. Training vẫn fail-fast; code không thay
`NaN` bằng số 0 vì việc đó che giấu lỗi tối ưu hóa.

Sau backward, code tiếp tục bỏ qua update nếu gradient không hữu hạn như hành vi
hiện tại. Sau optimizer step, HeatGeo kiểm tra student parameters để lỗi được
phát hiện tại chính update gây hỏng model, thay vì ở forward pass kế tiếp.

## Phạm vi thay đổi

Thay đổi tập trung vào:

- `config/heatgeo_config.py`: cấu hình student FP32.
- `distiller.py`: load dtype, log precision, kiểm tra parameters sau optimizer
  step và cung cấp diagnostic rõ ràng.
- `src/criterions/heatgeo_distillation.py`: kiểm tra đầu vào và loss con.

Không rebuild kNN graph, không thay đổi công thức HeatGeo, loss weights,
learning rate, training dataset, benchmark protocol hoặc model architecture.

## Xác minh

Bản vá đạt yêu cầu khi:

1. Static compilation của các Python file thay đổi thành công.
2. Student được báo là `torch.float32` trên MPS.
3. Teacher cache và HeatGeo artifact cũ vẫn load được.
4. Ít nhất mười HeatGeo optimizer steps chạy với mọi loss, gradient, optimizer
   state và student parameter hữu hạn.
5. W&B tiếp tục nhận các loss component hiện có.

Không cài package mới và mọi lệnh kiểm tra sử dụng `.venv` của project.
