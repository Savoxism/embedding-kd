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
