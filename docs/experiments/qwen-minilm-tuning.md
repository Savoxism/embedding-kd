# Qwen3-Embedding-0.6B → MiniLMv2-H384 Loss Tuning

Thiết lập cố định: seed `42`, 5 epochs, learning rate `2e-5`. Hệ số của
`L_rel` được giữ cố định ở `1.0`; bảng dưới tuning `L_row` với non-backtracking
walk và `row_start_epoch=2` (đánh số epoch từ 1).

| Objective | `row_weight` | `num_walks` | `walk_length` | Avg-In | Avg-Out | Avg |
|---|---:|---:|---:|---:|---:|---:|
| $L_{rel}$ | 0.00 | 0 | 4 | 67.76 | 76.82 | 73.80 |
| $L_{rel} + 0.1L_{row}$ | 0.10 | 4 | 4 | — | — | — |
| $L_{rel} + 0.3L_{row}$ | 0.30 | 4 | 4 | — | — | — |
| $L_{rel} + 0.5L_{row}$ | 0.50 | 4 | 4 | 68.00 | 77.94 | 74.63 |
| $L_{rel} + 1.0L_{row}$ | 1.00 | 4 | 4 | 68.13 | 78.25 | 74.88 |

Các giá trị `Avg-In`, `Avg-Out` và `Avg` được ghi theo thang `[0, 100]`, làm
tròn đến hai chữ số sau dấu thập phân.

## Rút gọn `L_row` về một cơ chế duy nhất

Walk chỉ dùng để *chọn* row — trajectory không bao giờ là target — nên `num_walks`
và `walk_length` không xuất hiện trong objective. Cơ chế thay thế: mọi pool column
được teacher chọn (diffusion support, không tính hard/uniform negative) được
promote thành auxiliary row, trọng số **đều**. Batch anchor bị loại vì `L_rel` đã
khớp transition row của chúng ở scale r=1. Row set là hàm tất định của candidate
pool.

Tất cả các dòng dưới: `row_weight=1.0`, seed `42`, 5 epochs, lr `2e-5`.

| Cơ chế chọn row | Trọng số | HP | Avg-In | Avg-Out | Avg |
|---|---|---:|---:|---:|---:|
| non-backtracking walk | visit count | 2 | 68.13 | 78.25 | 74.88 |
| **teacher-selected columns** | **uniform** | **0** | — | — | **74.86** |
| teacher-selected columns | exposed mass $m_B(j)$ | 0 | — | — | 74.76 |
| teacher-selected columns (start epoch 1) | uniform | 0 | 67.94 | 78.26 | 74.82 |
| + 1/c_B(j) + ambient r=0 mỗi row | — | 0 | 67.93 | 77.96 | 74.62 |

**Kết luận: uniform giữ nguyên hiệu năng của walk (−0.02, dưới nhiễu single seed)
và bỏ được cả hai hyperparameter.** Ba biến thể còn lại đã bị **xóa khỏi code**
(còn trong git history tại `b1f683b` trở về trước).

### Tại sao walk và uniform bằng nhau

| | walk | uniform |
|---|---:|---:|
| `row_count`/batch | 897 | 843 |
| `row_node_hit_ratio` | 0.997 | 1.000 |
| `row_exposed_mass` | — | 0.44 |

Hai cơ chế cấp gần như cùng lượng supervision (843 so với 897 row, tức 94%).
Con số 843 khớp dự đoán từ quota: 64 anchor × 14 diffusion support = 896 slot,
dedup còn ~843.

Đáng chú ý: `row_node_hit_ratio` của walk đạt 0.997 — walk gần như không bao giờ
trượt ra ngoài pool, vì `sample_with_rows` **chèn** node vừa thăm vào candidate
array, thay chỗ uniform negative. Walk không *khám phá* row có sẵn trong pool; nó
sửa pool để chứa row của chính nó.

### Vì sao ba biến thể kia thua

**Trọng số theo exposed mass (74.76).** Xác suất một node được promote đã tỷ lệ
với teacher mass mà các anchor đặt lên nó, nên weight theo mass lần nữa là bình
phương selection bias — dồn trọng số về hub nằm sâu trong neighborhood của anchor,
đúng chỗ transition target của row gần trùng với target r=1 mà `L_rel` đã cấp cho
anchor đó.

**Trọng số 1/c_B(j) (trơ).** Inverse-inclusion plug-in trong phạm vi batch. Đo
được: `row_eff_count` 833.6 so với `row_count` 844.3 — lệch khỏi uniform 1.3%.
Với corpus 13.5k và 64×14 support mỗi batch, c_B(j)=1 cho ~99% row, nên plug-in
within-batch không nhìn thấy inclusion bias vốn tác động xuyên batch.

**Ambient r=0 cho mỗi row (−0.30 out-of-domain).** Động lực là
`row_exposed_mass = 0.44`: restricted target chỉ chuẩn hóa trên 44% transition
mass thật, trong một method mà mọi truncation khác được giữ ở 1%. Kết quả đảo
ngược giả thuyết: `avg_in` không đổi (67.93 vs 67.94) còn `avg_out` giảm 0.30 —
đúng nhóm benchmark nhạy calibration mà term này nhắm tới. Cơ chế bên trong vẫn
chạy đúng (`kl_amb` của anchor giảm 0.170 vs 0.187, `student_top1` tăng 0.206 vs
0.197), nhưng nó chiếm nửa gradient budget của mỗi row và kết quả downstream tệ đi.

**Đây là một falsification sạch, đáng một dòng trong ablation của paper:** caveat
exposed-mass **không binding về mặt thực nghiệm**. Restricted transition matching
mới là active ingredient của `L_row`; việc renormalize trên support hở không phải
điểm yếu cần bù.

## Hyperparameter còn lại

`row_start_epoch=1` là default (knob trở nên inert): diagnostics epoch 1 của run
start-at-1 giống hệt các epoch sau — `row_count=844`, `row_exposed_mass=0.4407` —
và `loss_rel` cuối còn thấp hơn run start-at-2 (0.6502 so với 0.6686). Warm-up
tồn tại vì walk sinh row ngẫu nhiên và nhiễu; promoted column thì không.

$$
\boxed{L_{\text{row}} \text{ chỉ còn đúng một hyperparameter: } \texttt{row\_weight}}
$$

### Cảnh báo khi đọc bảng

74.88 → 74.86 → 74.82 đều nằm trong nhiễu **nếu xét từng bước**, nhưng ba bước
cùng chiều đi xuống. Single seed không phân biệt được "nhiễu" với "chi phí nhỏ
tích lũy". Trade 0.06 để bỏ ba hyperparameter là rõ ràng đáng, nhưng **trước khi
chốt số cho paper phải chạy 3 seed** và so với 3 seed của walk.

### Hai arm chưa chạy

| Run | Thay đổi | Giả thuyết |
|---|---|---|
| A1 | `row_weight=1.5` | tương đương α=0.6 ở dạng tổng-weight-1 (AdamW bất biến với global scale); bảng monotone 0.5→74.63, 1.0→74.88 và dừng đúng ở mép |
| A2 | `lr=3e-5` | underfitting trong cap 5 epoch: mọi KL còn giảm ở epoch 5, `loss_excess=0.66`, `student_top1` 0.21 vs teacher 0.33 |

Chạy tách riêng từng arm; run stack `ht + ambient` cho thấy nếu không có metric
tách bạch thì không quy được nguyên nhân cho arm nào.
