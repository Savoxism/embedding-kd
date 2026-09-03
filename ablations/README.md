# Kết quả ablation GGPKD

Bản sao kết quả từ `H200_Tensara:/home/tensara/projects/heatgeo`, được gom về
repository local ngày 03/09/2026. Bao gồm **43/43 run đã hoàn tất**, các bảng,
hình, log và metrics; giữ nguyên số liệu gốc.

## Bắt đầu từ đâu?

- [results.csv](results.csv): bảng tổng hợp chính, **43 dòng × 52 cột**; mỗi dòng
  là một run, gồm điểm từng benchmark, các điểm trung bình, geometry, số texts /
  tokens được encode, thời gian, peak VRAM và cấu hình.
- [logs/controller.log](logs/controller.log): bảng tổng hợp dạng text theo từng
  cặp model và từng nhóm ablation, gồm mean/std, paired delta và hubness.
- [coverage.csv](coverage.csv): **60 dòng** replay coverage theo policy, seed và
  epoch, kèm epsilon.
- [tables/](tables/): **7 bảng LaTeX**, tương ứng S1–S4, G1, N1, X1.
- [figures/](figures/): Figure 1 và Figure 2, mỗi hình có cả PDF và PNG.
- [runs/](runs/): log và metrics chi tiết của toàn bộ 43 run.
- [logs/](logs/): log controller, warm-up, từng nhóm và log sửa Figure 2.

## Phạm vi run

| Cặp model | Nhóm | Run hoàn tất | Seeds |
|---|---|---:|---|
| Qwen3-0.6B → MiniLMv2-H384 | Full | 3 | 42, 43, 44 |
| Qwen3-0.6B → MiniLMv2-H384 | S1 support / baselines | 15 | 42, 43, 44 |
| Qwen3-0.6B → MiniLMv2-H384 | S2 local-only | 3 | 42, 43, 44 |
| Qwen3-0.6B → MiniLMv2-H384 | S3 direct target | 3 | 42, 43, 44 |
| Qwen3-0.6B → MiniLMv2-H384 | S4 ambient × row | 9 | 42, 43, 44 |
| Qwen3-0.6B → MiniLMv2-H384 | G1 kNN | 2 | 42 |
| Qwen3-0.6B → MiniLMv2-H384 | N1 negatives | 2 | 42 |
| BGE-M3 → MiniLMv2-H768 | Full + X1 uniform | 6 | 42, 43, 44 |
| **Tổng** | | **43** | |

Các run dùng 5 epoch, batch size 64, learning rate `3e-5`, và chỉ GPU vật lý 0.
Cấu hình cụ thể của từng arm nằm trong `run.json` và `arm.json` của run đó.
G1/N1 chỉ có một seed mỗi arm: không diễn giải độ lệch chuẩn bằng 0 trong bảng
tổng hợp như bằng chứng về độ ổn định qua nhiều seed.

## Tra cứu một run

Đường dẫn local:

```text
runs/<pair>/<ablation>/<arm>/seed<seed>/
    .done
    arm.json
    run.json
    train.log
    metrics.jsonl
    epochs.jsonl
    step_metrics.jsonl
```

- `.done`: marker hoàn tất của launcher.
- `arm.json`: danh tính arm, seed, graph key, thời gian và cờ bổ sung.
- `run.json`: cấu hình và thông tin môi trường được runtime ghi nhận.
- `metrics.jsonl`: metrics training từng epoch và kết quả final test.
- `epochs.jsonl`: diagnostics từng epoch, gồm geometry probe.
- `step_metrics.jsonl`: telemetry ở mức từng step.
- `train.log`: log đầy đủ của run.

Các đường dẫn nằm bên trong JSON/CSV vẫn là đường dẫn nguồn trên server;
chúng không bị viết lại khi sao chép. Ví dụ `runs/ablation/<pair>/...` trên server
tương ứng với `ablations/runs/<pair>/...` trong repository local.

## Nguồn và bản sửa Figure 2

- Code training: branch `nqd_ablation`, commit `576299f`.
- Training cuối cùng hoàn tất lúc **03:03:13 ngày 03/09/2026**, giờ Việt Nam.
- Coverage, CSV và bảng được tạo bởi `scripts/ablation/run_all.sh`.
- Figure 2 đã được sửa bằng commit local `aaeae41`: chỉ ghép các run S1 với
  full của **cùng cặp model**, không gộp full BGE vào kết quả S1 Qwen. Hình trong
  thư mục này là **bản đã sửa**, tạo lúc 03:26 ngày 03/09/2026.
- Xem [logs/figure2-pair-fix.log](logs/figure2-pair-fix.log). Bản sửa đã qua
  77 test và không thay đổi hoặc chạy lại training.
- Bản hình cũ không được đưa vào bộ kết quả này; bản sao lưu vẫn ở server:
  `/home/tensara/projects/heatgeo/runs/ablation/_dispatch/figure2-before-pair-fix-20260903/`.

## Những file lớn vẫn giữ trên server

Bộ kết quả local không sao chép weights, teacher cache hoặc graph artifacts.
Các file này vẫn còn nguyên tại:

- Weights từng run:
  `/home/tensara/projects/heatgeo/runs/ablation/<pair>/<ablation>/<arm>/seed<seed>/weights/student_epoch_5.pt`
- Teacher embeddings và graph:
  `/home/tensara/projects/heatgeo/cache/ggpkd/<pair>/`
- Log chi tiết của graph builder:
  `/home/tensara/projects/heatgeo/logs/ggpkd/<pair>/`

File nguồn trên server không bị di chuyển hay xóa. `.gitignore` hiện tại có thể
bỏ qua một số log/metrics trong thư mục này; chúng vẫn có đầy đủ trên filesystem.
