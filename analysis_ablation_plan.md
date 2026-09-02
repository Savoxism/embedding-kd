# Kế hoạch ablation cho GGPKD

## Kết luận từ bằng chứng hiện có

- Main table đã cho thấy GGPKD hơn TALAS về `Avg` ở cả ba setting: **+0.19 / +0.66 / +1.89** điểm. Tuy nhiên, gain chủ yếu đến từ STS (xấp xỉ **+1.47 / +4.10 / +4.82**), còn pair-classification giảm ở cả ba setting. Vì vậy ablation phải báo riêng `STS Avg`, `Pair-cls Avg` và `Avg`; chỉ báo `Avg` sẽ che trade-off.
- Đề xuất gốc đúng ở causal story: **support coverage → geometry preservation → downstream**. Khoảng trống hiện tại là main results chưa chứng minh chuỗi này và chưa loại trừ lời giải thích “GGPKD encode/scoring nhiều hơn”.
- Các kết quả row trong `docs/experiments/qwen-minilm-tuning.md` chỉ là seed 42, LR `2e-5` và setup cũ; không nên đưa thẳng vào ablation table của paper hiện tại (`3e-5`, graph/objective hiện tại).

## Protocol chung (khóa trước khi chạy)

- Setting chính: **Qwen3-0.6B → MiniLMv2-H384**, 5 epochs, LR `3e-5`, seeds **42/43/44**. Đây là setting rẻ nhất và hiện có gain lớn nhất.
- Mọi so sánh support phải match theo **số unique student texts và số tokens được encode/update**; đồng thời báo pairwise scores, wall-clock và peak VRAM. Candidate quota bằng nhau chưa đủ vì mức dedup khác nhau.
- Dùng cùng train/validation split và cùng rule chọn checkpoint bằng validation; tuyệt đối không chọn bằng test. Báo mean ± std và seed-wise paired delta.
- Trước training, tạo một **fixed held-out geometry probe** độc lập với support draws. Định nghĩa rõ cùng một dense teacher distribution `p_i^T` cho mọi arm, rồi đo:
  - cumulative mass coverage `1 - delta_T`;
  - selected-support distortion `epsilon`;
  - held-out teacher-weighted global distortion `E_hat`;
  - teacher–student similarity Spearman.
- Instrumentation cần bổ sung trước: chế độ geometry estimator **diagnostic-only** (không cộng vào loss), cumulative exposed-mass theo anchor/epoch, unique texts/tokens encoded. Probe hiện tại chưa truyền teacher embeddings nên chưa ghi teacher–student Spearman.

## Các run cần chạy

### P0 — bắt buộc cho main paper

| ID | So sánh (cùng encoder budget) | Seeds | Tín hiệu cần thấy | Nó hỗ trợ paper như nào |
|---|---|---:|---|---|
| S1 | Uniform/random support; teacher top-K; teacher-proportional; **head + proportional tail (GGPKD)** | 42–44 | Hybrid có coverage ban đầu gần top-K nhưng tiếp tục tăng; `E_hat` thấp hơn random/top-K | Kiểm chứng core thesis: teacher relevance, không phải random co-occurrence hay thêm compute, quyết định supervision hữu ích |
| S2 | Full `R={1,2,4}` vs local-only `R={1}` | 42–44 | Multi-scale giảm `E_hat` và tăng STS ở cùng số encoded texts | Chứng minh diffusion tạo reach ngoài local neighborhood |
| S3 | Full diffusion target vs **đúng cùng selected nodes** nhưng target là direct teacher cosine | 42–44 | Diffusion target tốt hơn direct target | Tách giá trị của composed graph relations khỏi lợi ích đơn thuần do chọn được farther nodes |
| S4 | Factorial ambient × row: full; no ambient; no row (`row_weight=0`); neither | 42–44 | Ambient cải thiện calibration/pair-cls hoặc global geometry; row tăng hiệu quả/score với 0 extra encodes | Kiểm chứng riêng reconnect và amortization, đồng thời phát hiện interaction giữa hai term |

`Full` chỉ chạy một lần mỗi seed và tái sử dụng trong S1–S4 nếu commit, artifact, budget và checkpoint rule hoàn toàn giống nhau. Với 3 seeds, P0 cần **24 new training runs** nếu full 3-seed hiện tại tái sử dụng được; nếu không thì 27.

### P1 — củng cố claim hoặc appendix

| ID | So sánh | Cách chạy | Giá trị cho paper |
|---|---|---|---|
| G1 | Directed kNN; symmetrized kNN; mutual kNN | Screen seed 42; chỉ confirm 42–44 nếu signal ổn định và degree/budget được match | Hỗ trợ claim mutuality suppress hubness; báo max/P99 indegree và edge share của top-1% hubs, không chỉ downstream |
| N1 | All-hard; all-uniform; current hard/uniform mix, giữ tổng negative quota | Screen seed 42; confirm nếu chênh lệch `Avg` hoặc `E_hat` đáng kể | Trả lời reviewer về vai trò negatives và hoàn tất khoảng trống đã ghi trong config |
| X1 | Full vs random support trên **BGE-M3 → MiniLM-H768** | 42–44, chỉ hai arm | Replicate core support claim trên teacher family khác; có giá trị hơn sweep nhiều hyperparameter trên setting chính |

Không ưu tiên trước rebuttal: sweep rộng `k`, perplexity, truncation tolerance, `lambda_row`; t-SNE/UMAP teacher-vs-student; generic loss curves. Chúng có information gain thấp hơn P0.

## Tables/figures cần thêm

### Main paper

1. **Table: fixed-budget support selection** — policy, cumulative coverage, `epsilon`, `E_hat`, teacher–student Spearman, STS Avg, Pair-cls Avg, Avg, unique texts/tokens. Đây là bằng chứng trực tiếp nhất cho causal chain.
2. **Table: full-model deletion/factorial** — Full, `R={1}`, same-nodes direct target, no ambient, no row, neither; báo `E_hat`, STS Avg, Pair-cls Avg, Avg và encoded texts. Table này trả lời component nào thực sự cần thiết.
3. **Figure 1: cumulative teacher mass ever exposed vs epochs/opportunities** — random, top-K, proportional, hybrid. Top-K nên plateau; hybrid nên có high initial coverage và tiếp tục tăng. Figure có thể tạo bằng replay sampler offline, không cần thêm training.
4. **Figure 2: coverage → geometry → downstream** — hai panel: coverage vs `E_hat`, và `E_hat` vs STS Avg; mỗi điểm là policy/budget, kèm seed error bars. Đây là figure nối theory với empirical result mạnh nhất.

### Appendix

- Efficiency table: unique texts/tokens, relation scores, auxiliary rows/update, wall-clock, peak VRAM. Với row reuse, nhấn mạnh “thêm X supervised centers với 0 extra encodes”.
- Hubness diagnostic cho directed/symmetrized/mutual kNN.
- Per-task ablation đầy đủ và seed-wise deltas; hyperparameter sensitivity nhỏ quanh setting đã chọn nếu còn compute.

## Tiêu chí ra quyết định

- Claim về causal chain chỉ được giữ mạnh nếu coverage cao hơn đi cùng `E_hat` thấp hơn trên nhiều policies/seeds, và `E_hat` có quan hệ nhất quán với STS. Nếu downstream tăng nhưng `E_hat` không giảm, chỉ claim empirical gain, không claim mechanism đã được xác nhận.
- Một component chỉ nên giữ như contribution chính nếu deletion gây regression lớn hơn seed noise hoặc tạo trade-off cơ chế rõ ràng. Nếu ambient/row gần như flat, chuyển nó thành efficiency/design detail thay vì overclaim.
- Vì Pair-cls hiện giảm so với TALAS, cần trình bày đây là trade-off thật. Nếu ambient ablation giải thích hoặc sửa được regression này, đó là run có upside lớn nhất ngoài core support experiment.
