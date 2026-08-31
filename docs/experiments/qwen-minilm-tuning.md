# Qwen3-Embedding-0.6B → MiniLMv2-H384 Loss Tuning

Thiết lập cố định: seed `42`, 5 epochs, learning rate `2e-5`. Hệ số của
`L_rel` được giữ cố định ở `1.0`; bảng dưới chỉ tuning các loss bổ sung.

| Objective | `mass_weight` | `geo_weight` | `sym_weight` | Avg-In | Avg-Out | Avg |
|---|---:|---:|---:|---:|---:|---:|
| $L_{rel}$| 0.00 | 0.00 | 0.00 | 67.47 | 76.82 | 73.71 |
| $L_{rel} + 0.5L_{mass}$ | 0.50 | 0.00 | 0.00 | 68.32 | 76.93 | 74.06 |
| $L_{rel} + 1.0L_{mass}$ | 1.00 | 0.00 | 0.00 | 68.24 | 76.58 | 73.80 |
| $L_{rel} + 0.5L_{geo}$ | 0.00 | 0.50 | 0.00 | — | — | — |
| $L_{rel} + 0.5L_{geo} + 0.5L_{mass}$ | 0.50 | 0.50 | 0.00 | — | — | — |
| $L_{rel} + 0.5L_{sym}$ | 0.00 | 0.00 | 0.50 | — | — | — |
| $L_{rel} + 0.5L_{sym} + 0.5L_{mass}$ | 0.50 | 0.00 | 0.50 | — | — | — |

Các giá trị `Avg-In`, `Avg-Out` và `Avg` được ghi theo thang `[0, 100]`, làm
tròn đến hai chữ số sau dấu thập phân.
