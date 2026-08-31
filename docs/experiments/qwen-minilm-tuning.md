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

## `row_mode`: bỏ hyperparameter của random walk

Walk chỉ dùng để *chọn* row — trajectory không bao giờ là target — nên
`num_walks` và `walk_length` không xuất hiện trong objective. Hai closure mode
thay cơ chế chọn row bằng chính các pool column mà teacher đã chọn khi dựng
`L_rel` (batch anchor bị loại vì `L_rel` đã match transition row của chúng ở
scale r=1), khác nhau duy nhất ở trọng số `nu_B`.

Tất cả các dòng dưới dùng `row_weight=1.0`, `row_start_epoch=2`, seed `42`.

| `row_mode` | Row selection | Trọng số | Walk HP | Avg |
|---|---|---|---:|---:|
| `walk` | non-backtracking walk | visit count | 2 | 74.88 |
| `closure_u` | teacher-selected columns | uniform | 0 | 74.86 |
| `closure_m` | teacher-selected columns | exposed mass $m_B(j)$ | 0 | 74.76 |

**Kết luận: `closure_u` giữ nguyên hiệu năng của walk (−0.02, dưới mức nhiễu của
single seed) và bỏ được cả hai hyperparameter.** Đây là arm được chọn, và là
default của `HeatGeoConfig`.

### Tại sao hai cơ chế cho kết quả bằng nhau

Diagnostics trung bình mỗi epoch (batch 64, `candidates_per_anchor=66`):

| | `walk` | `closure_u` |
|---|---:|---:|
| `row_count` (số row/batch) | 897 | 843 |
| `row_node_hit_ratio` | 0.997 | 1.000 |
| `row_valid_ratio` | 0.996 | 0.996 |
| `row_exposed_mass` | — | 0.44 |

Hai cơ chế cấp gần như cùng một lượng supervision: 843 so với 897 row, tức 94%.
Điều đó giải thích vì sao điểm số gần như trùng nhau.

Đáng chú ý là `row_node_hit_ratio` của walk đạt 0.997 — walk gần như không bao
giờ "trượt" ra ngoài pool. Lý do là `sample_with_rows` **chèn** node walk vừa thăm
vào candidate array, thay chỗ uniform negative. Nghĩa là walk không thật sự
*khám phá* row có sẵn trong pool; nó sửa pool để chứa row của chính nó. Closure
thì dùng đúng các support column mà `L_rel` đã trả tiền để encode. Con số 843
khớp với dự đoán từ quota: 64 anchor × 14 diffusion support = 896 slot, dedup
còn ~843 column duy nhất.

`row_exposed_mass=0.44` là con số cần lưu ý cho phần theory: pool chỉ expose
trung bình 44% teacher transition mass của mỗi row được promote, nên restricted
KL của `L_row` đang chuẩn hóa trên chưa tới một nửa mass thật. Đây đúng là
calibration caveat mà paper đã nêu; `closure_m` cố xử lý nó bằng trọng số và
thất bại, nên nếu muốn giải quyết thì phải bằng cơ chế khác (ví dụ một ambient
scale cho auxiliary row, giống r=0 của anchor).

`closure_m` thua `closure_u` 0.10. Đúng với dự đoán trước khi chạy: row có
$m_B(j)$ cao là row nằm sâu trong neighborhood của một anchor nào đó, tức là row
mà supervision gần trùng với target r=1 mà `L_rel` đã cấp cho anchor đó; mass
weighting vì vậy dồn trọng số về phía phần redundant và bỏ rơi các row xa. Hai
metric `row_eff_count` (số row hiệu dụng, $1/\sum\nu^2$) và `row_exposed_mass`
được log mỗi epoch để kiểm chứng: dưới `closure_m`, `row_eff_count` phải thấp
hơn `row_count` rõ rệt trong khi `closure_u` giữ hai giá trị bằng nhau.

Lưu ý khi đọc bảng: bỏ walk cũng khôi phục các uniform negative mà walk-visited
node từng thay thế trong candidate draw, nên `L_rel` cũng đổi nhẹ — so sánh
không hoàn toàn isolated.

### Còn lại

`row_start_epoch=2` là hyperparameter cuối cùng của `L_row`. Closure không tạo
row ngẫu nhiên nên có thể không cần warm-up; chạy `closure_u` với
`row_start_epoch=1` để kiểm tra. Nếu ngang bằng thì `L_row` chỉ còn đúng một
knob là `row_weight`.
