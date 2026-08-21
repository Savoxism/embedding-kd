# RIPPLE Experiment and Figure Plan

## 1. Mục tiêu bằng chứng

Paper cần trả lời sáu câu hỏi theo đúng causal chain:

1. Mini-batch relational KD có thật sự bỏ sót teacher geometry?
2. Gain có đến từ teacher-anchored support, hay chỉ từ việc thêm nhiều comparisons?
3. Khi tăng k, alignment có chuyển từ local sang corpus-level như claim không?
4. Multi-scale diffusion có chuyển one-step geometry sang khoảng cách dài hơn không?
5. Row supervision có cải thiện task accuracy ở mức λ nào?
6. Candidate sampling với m ≪ k có giữ được tín hiệu của full neighborhood với chi phí thấp hơn không?

Mỗi claim phải có:

- một metric trực tiếp đo cơ chế;
- một downstream metric;
- một control loại trừ alternative explanation;
- mean ± standard deviation trên nhiều seeds cho kết quả được đưa vào main paper.

## 2. Trạng thái hiện tại và các điểm phải khóa trước khi chạy

### Đã có

- Main results cho ba teacher–student pairs.
- Code đã có teacher graph, diffusion targets, mixed candidate sampling và walk/row supervision.
- CLI đã expose k, perplexity, candidate size, walk weight và các walk controls cần cho protocol chính.

### Chưa đủ

- Chưa thấy artifact theo run như metrics.jsonl, config snapshot, checkpoint và final test CSV để truy ngược các số trong main table.
- Chưa có evaluator cho full-corpus alignment, rank-band JS, held-out diffusion error, scale separation và sampled-vs-full gradient fidelity.
- Chưa có CLI controls cho diffusion scales, scale weights, direct weight và shared-temperature ablation.

### Các inconsistency còn phải giải quyết

1. Paper ghi m = 64, nhưng default code là candidate_size = 66 với quota 14 direct + 26 diffusion + 26 random.
2. Exact model ID của BGE-M3 → MiniLMv2-H768 chưa được tìm thấy trong scripts/configs.
3. Tên P3-H384 giữa README và scripts chưa nhất quán.
4. λ trong paper tương ứng với walk_weight trong code.
5. λ = 0 phải vẫn giữ num_walks = 4. Đặt num_walks = 0 làm thay đổi candidate composition, nên không còn là sweep chỉ thay λ.

Không chạy full matrix trước khi năm điểm này được khóa và lưu trong một protocol manifest.

## 3. Protocol chung

### 3.1 Canonical graph regime

Main results hiện tại dùng target perplexity ρ = 30. Đây là canonical graph regime cho paper và mọi confirmation run:

- mỗi transition row tự giải bandwidth τ_i sao cho entropy bằng log ρ khi attainable;
- τ_i được lưu trong graph artifact và tái sử dụng cho student/walk readout của row đó;
- không tune hoặc report một fixed graph temperature;
- log tỷ lệ rows bị clamp vì mutual degree không đủ để đạt perplexity ρ.

### 3.2 Canonical candidate budget

Chọn một cấu hình duy nhất và đồng bộ paper với code:

- m = 64 với quota được sửa để tổng bằng 64; hoặc
- m = 66 và sửa toàn bộ paper.

Khuyến nghị dùng m = 64 vì narrative sạch hơn. Một cấu hình hợp lệ là 14 direct + 25 diffusion + 25 random.

### 3.3 Seeds và reporting

- Screening: seed 42.
- Confirmation: seeds 13, 42, 87.
- Main RIPPLE setting: thêm seeds 21 và 100 nếu variance lớn hoặc gain biên.
- Main tables: mean ± standard deviation.
- Báo cáo absolute score và paired delta so với baseline dùng cùng seed.

### 3.4 Run artifact bắt buộc

Mỗi run lưu:

- resolved config;
- git commit và dirty-state hash;
- exact teacher/student model IDs;
- graph-cache fingerprint;
- seed;
- train/validation/test metrics;
- best-checkpoint criterion;
- wall-clock time, peak VRAM và graph-cache size.

Suggested layout:

    outputs/
      <experiment>/<pair>/<variant>/seed_<seed>/
        config.yaml
        metrics.jsonl
        final_test_results.csv
        resource_metrics.json
        checkpoint/

### 3.5 Code/evaluator cần bổ sung

Trước khi chạy:

- expose diffusion scales and weights, direct weight và shared-vs-scale-specific temperature;
- expose candidate source quotas;
- lưu candidate IDs và source labels trên một fixed diagnostic subset;
- thêm offline evaluator cho các metrics ở Section 3.6;
- tách cache theo k, perplexity, teacher ID và corpus fingerprint;
- thêm graph diagnostics: mutual-edge rate, isolated nodes, component sizes, degree distribution và retained teacher mass.

### 3.6 Metrics

#### Downstream

- STS Spearman theo dataset và macro average.
- Pair classification accuracy theo dataset và macro average.
- Nếu main table hiện dùng combined Avg, giữ nguyên đúng aggregation protocol.

Không chỉ báo cáo Avg: per-task delta phải nằm trong appendix để kiểm tra gain có bị một dataset chi phối không.

#### Neighborhood mass

Với teacher distribution q_T và teacher top-k set N_k:

    Mass@k = average_i sum_{j in N_k(i)} q_T(j | i)

Đây là metric trực tiếp cho câu hỏi k lớn hơn bao phủ thêm bao nhiêu teacher mass.

#### Rank-band alignment

Chia corpus theo teacher rank bands:

- near: ranks 1–k;
- medium: k+1–4k;
- far: lớn hơn 4k.

Trên một fixed anchor set, tính JS divergence giữa teacher và student distributions trong từng band. Bands phải cố định giữa các variants để so sánh hợp lệ.

#### Full-corpus alignment

Tính JS divergence giữa teacher và student distributions trên toàn corpus cho một fixed diagnostic anchor set. Dùng cùng anchor set, chunking và numerical precision cho mọi run.

#### Held-out diffusion error

Train với một tập scales rồi evaluate:

    E_r = average_i JS(q_i,T^(r), q_i,S^(r))

ở r ∈ {1, 2, 4, 8, 16}. Giá trị quan trọng nhất là r = 8 và 16 khi các scales này không được supervise trực tiếp.

#### Scale separation

Đo khoảng cách trung bình giữa student targets ở các resolutions:

    E_sep = average_{r < s} average_i JS(q_i,S^(r), q_i,S^(s))

Shared temperature gây collapse nếu E_sep nhỏ đồng thời held-out errors lớn.

#### Gradient fidelity cho sampling

Trên cùng checkpoint và cùng anchor batch:

    G_m = cosine(gradient L_sampled(m), gradient L_reference)

L_reference dùng neighborhood lớn nhất khả thi, ví dụ m = 256 hoặc full support trên diagnostic subset.

## 4. Experiment matrix

### E0 — Reproduce main results

**Mục tiêu:** xác nhận main table có provenance và variance.

| Biến | Giá trị |
|---|---|
| Teacher–student pairs | 3 settings hiện có |
| Variants | student-only, strongest reproduced baseline, RIPPLE |
| Seeds | 13, 42, 87 |
| Metrics | per-task score, Avg, mean ± std |

Nếu số cũ không reproduce trong tolerance đã định trước, dừng confirmation runs và audit protocol trước.

### E1 — Causal anchoring ladder

**Mục tiêu:** chứng minh gain đến từ teacher-selected corpus support, không đơn thuần từ nhiều comparisons hơn.

| Variant | Support | Target | Scales | Row loss |
|---|---|---|---|---|
| A. Batch-local relational | current mini-batch | teacher relations | one-step | no |
| B. Uniform-corpus control | same number of corpus points as C | teacher relations | one-step | no |
| C. Teacher-anchored local | teacher kNN | teacher operator | r = 1 | no |
| D. Anchored multi-scale | teacher kNN | diffusion targets | canonical scales | no |
| E. Full RIPPLE | teacher kNN + walk rows | diffusion targets | canonical scales | yes |

Fairness constraints:

- A–C dùng cùng student, optimizer, epochs và số scored comparisons trên mỗi anchor.
- B và C chỉ khác cách chọn corpus candidates.
- D và C phải weight-match graph/direct objective groups.
- E và D chỉ khác row term; candidate composition phải giữ nguyên khi λ = 0.

Primary evidence:

- downstream Avg;
- near/medium/far rank-band JS;
- Mass@k.

Interpretation:

- C > B hỗ trợ claim teacher anchoring là nguồn gain.
- D > C chủ yếu ở medium/far JS hỗ trợ multi-scale transfer.
- E > D hỗ trợ row supervision, nhưng không pitch row loss như contribution độc lập.

### E2 — From local to corpus-level alignment: k-only sweep

**Mục tiêu:** Section 5.2 chỉ thay đổi k như yêu cầu.

| Variable | Values |
|---|---|
| k | 50, 100, 200, 400 |
| m | fixed canonical value |
| scales | fixed |
| λ | fixed |
| all other settings | fixed |

Đo:

- Mass@k;
- near, medium và far rank-band JS;
- full-corpus JS;
- downstream Avg;
- graph health diagnostics.

Figure phải cho thấy k tăng tạo coverage/alignment gain trước khi downstream performance bão hòa. Nếu k quá lớn làm graph kém local hoặc mutual graph phân mảnh, báo cáo đó như trade-off.

### E3 — Multi-scale mechanism

**Mục tiêu:** chứng minh multi-scale không chỉ thêm nhiều loss terms.

Variants:

1. r = 1 only.
2. r = 1 repeated with matched total graph-loss weight.
3. supervised scales r = {1, 2, 4}.
4. supervised scales r = {1, 2, 4, 8}.
5. canonical multi-scale with shared temperature.
6. canonical multi-scale with current scale-specific temperature ladder.

Đo:

- E_r tại r = 1, 2, 4, 8, 16;
- E_sep;
- downstream Avg.

Primary comparison:

- r = 1 only vs {1,2,4} tại held-out r = 8 và 16.

Control 2 là bắt buộc để loại trừ giải thích rằng multi-scale tốt hơn chỉ vì có tổng loss weight lớn hơn.

Shared-temperature variant chỉ giữ nếu code thực sự hỗ trợ một shared effective resolution khác với current ladder. Nếu implementation vốn không có degree of freedom này hoặc so sánh không meaningful, bỏ ablation và bỏ claim khỏi paper.

### E4 — Row-supervision λ sweep

**Mục tiêu:** Section 5.4 chỉ đo accuracy khi sweep λ.

| λ | 0 | 0.1 | 0.3 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|---:|
| num_walks | 4 | 4 | 4 | 4 | 4 |
| other settings | fixed | fixed | fixed | fixed | fixed |

Đo duy nhất cho main paper:

- downstream accuracy/Avg;
- mean ± standard deviation ở các λ được confirm.

Run strategy:

1. Sweep toàn bộ grid với seed 42.
2. Confirm λ = 0, best interior λ và λ = 1 với seeds 13 và 87.
3. Không chọn λ bằng test set; dùng validation Avg.

Mandatory sanity control, đặt ngoài λ-sweep:

- no injection: num_walks = 0, λ = 0;
- injected support but zero row loss: num_walks = 4, λ = 0.

Control này cho biết walk-row injection tự nó có đổi candidate mix hay không. Không đưa vào Figure 2C nếu muốn giữ Section 5.4 là λ-only; báo cáo trong appendix hoặc footnote.

### E5 — Sampling efficiency

**Mục tiêu:** chứng minh m nhỏ xấp xỉ tốt full neighborhood.

| m | 16 | 32 | 64 | 128 | 256/reference |
|---|---:|---:|---:|---:|---:|
| source proportions | fixed | fixed | fixed | fixed | fixed |

Hai tầng:

#### E5a. Same-checkpoint fidelity probe

- dùng cùng checkpoint và fixed anchor batches;
- đo retained target mass;
- đo gradient cosine G_m;
- đo batch time và peak VRAM.

#### E5b. End-to-end training

- train variants m = 16, 32, 64, 128;
- đo Avg, wall-clock per epoch và peak VRAM;
- m = 256 chỉ chạy nếu budget cho phép.

Candidate duplicates phải được resolve deterministically và actual unique candidate count phải được log.

### E6 — Optional external validity

Chỉ chạy nếu claim của paper bao gồm retrieval hoặc corpus-level semantic search:

- BEIR trên 2–3 representative datasets; hoặc
- một MTEB retrieval subset phù hợp model language/domain.

Nếu không đủ budget, thu hẹp claim về STS và pair classification thay vì thêm một bảng retrieval nông.

## 5. Figure plan

### Figure 1 — Problem and method schematic

**Vị trí:** cuối Introduction hoặc đầu Section 3.

Ba panels:

1. Batch-local view: anchor và true teacher neighbors; mini-batch chỉ quan sát một phần nhỏ.
2. Corpus-anchored view: teacher graph chọn support trực tiếp.
3. RIPPLE view: one-step operator → multi-scale diffusion → overlapping walk rows.

Thiết kế:

- cùng một anchor ở cả ba panels;
- teacher neighbors cùng màu;
- unseen relations nét đứt;
- student supervision edges nét liền;
- chỉ một legend chung.

Caption phải diễn đạt USP: teacher geometry chọn comparisons, không phải random batch composition.

### Figure 2 — Mechanism dashboard

**Vị trí:** Section 5.2–5.4.

Panel A — k sweep:

- heatmap hoặc small multiples;
- x-axis: k;
- rows: near, medium, far, full-corpus JS;
- annotation thêm Mass@k;
- lower JS is better.

Panel B — diffusion transfer:

- x-axis: evaluation scale r;
- y-axis: E_r;
- curves: r = 1 only, weight-matched repeated r = 1, multi-scale, shared-temperature nếu hợp lệ;
- supervised scales dùng filled markers, held-out scales dùng hollow markers.

Panel C — λ sweep:

- x-axis: λ;
- y-axis: downstream accuracy/Avg;
- mean ± standard deviation error bars;
- không thêm calibration metric vì Section 5.4 được giới hạn ở accuracy.

### Figure 3 — Per-task delta forest plot

**Vị trí:** Main Results hoặc appendix nếu thiếu chỗ.

- y-axis: datasets;
- x-axis: paired RIPPLE-minus-baseline delta;
- một color cho mỗi teacher–student pair;
- zero reference line;
- confidence interval hoặc seed range.

Figure này cho biết average gain có nhất quán hay bị một task chi phối.

### Figure 4 — Sampling Pareto frontier

**Vị trí:** Section 5.5.

Panel A:

- x-axis: training time hoặc peak VRAM;
- y-axis: downstream Avg;
- labels: m values.

Panel B:

- x-axis: m;
- y-axis: retained teacher mass và gradient cosine G_m;
- có thể dùng hai aligned subplots thay vì dual y-axis.

Thông điệp mong muốn: m = 64 nằm gần Pareto knee.

### Figure 5 — Graph and walk health diagnostics

**Vị trí:** Appendix.

- mutual-edge rate theo k;
- isolated-node rate theo k;
- connected-component size distribution;
- walk row-distance histogram.

Chỉ đưa vào main nếu diagnostics giải thích trực tiếp một failure mode hoặc lựa chọn hyperparameter.

## 6. Run budget theo stage

### Stage A — One-seed screening

Dùng pair rẻ nhất, đề xuất P3 teacher → H384 student:

- E1 anchoring ladder: 5 runs;
- E2 k sweep: 4 runs;
- E3 mechanism: 6 runs;
- E4 λ sweep: 5 runs;
- E5 end-to-end m sweep: 4 runs.

Tổng: 24 runs. Nếu shared-temperature comparison không hợp lệ sau code audit, còn 23 runs.

E5a gradient probes không cần retrain.

### Stage B — Confirmation

Chỉ confirm variants qua gates:

- Anchoring gate: C phải tốt hơn uniform-corpus control trên alignment và không giảm downstream quá noise.
- Multi-scale gate: giảm held-out E_8 hoặc E_16; downstream không giảm quá noise.
- Row gate: có best interior λ hoặc variance-reduction rõ; nếu flat, report sensitivity thay vì claim gain.
- Sampling gate: chọn Pareto knee có accuracy trong tolerance của reference.

Chạy thêm seeds 13 và 87 cho các variants qua gate. E0 main reproduction phải chạy trên cả ba model pairs.

### Stage C — Optional expansion

- thêm two seeds cho main RIPPLE setting nếu variance lớn;
- retrieval evaluation;
- target-perplexity robustness nếu graph diagnostics cho thấy ρ = 30 không ổn định giữa teacher models;
- extra k hoặc m points quanh vùng chuyển pha.

## 7. Stopping rules

- Không chạy thêm k nếu Mass@k và full-corpus JS đã bão hòa hai points liên tiếp và graph health xấu đi.
- Không confirm shared-temperature ablation nếu implementation audit cho thấy nó không tạo một alternative model hợp lệ.
- Không claim row calibration nếu λ sweep chỉ cho noise-level differences; mô tả row loss như regularizer và báo sensitivity.
- Không chạy m = 256 end-to-end nếu m = 128 đã match reference gradient và accuracy trong tolerance.
- Không mở rộng sang retrieval nếu paper không có đủ protocol/baseline mạnh để support claim đó.

## 8. Mapping evidence vào paper

| Paper section | Evidence |
|---|---|
| Why Mini-Batch Relations Are Insufficient | coverage theory + Figure 1 |
| Main Results | E0 + Figure 3 |
| From Local to Corpus-Level Alignment | E2 + Figure 2A |
| Multi-Scale Ablations | E3 + Figure 2B |
| Row-Supervision Ablation | E4 + Figure 2C |
| Sampling Efficiency | E5 + Figure 4 |
| Appendix diagnostics | E1 full ladder + Figure 5 + injection sanity control |

## 9. Thứ tự hành động ngay

1. Truy provenance của main table và xác nhận mọi run dùng target perplexity ρ = 30.
2. Chốt m = 64 hay 66 và exact model IDs.
3. Thêm config snapshots, diffusion CLI controls và offline evaluators.
4. Chạy graph-only diagnostics cho k ∈ {50, 100, 200, 400}; bước này không cần train student.
5. Chạy Stage A trên pair rẻ nhất.
6. Tạo Figure 2 từ screening artifacts trước khi mở rộng seeds.
7. Chỉ confirm hypotheses đã qua gates.
8. Cập nhật paper bằng số thật; giữ placeholder cho cells chưa hoàn tất.

Ưu tiên compute nếu budget hạn chế:

1. E1 causal anchoring ladder.
2. E3 held-out diffusion transfer.
3. E2 k-only sweep.
4. E4 λ accuracy sweep.
5. E5 sampling Pareto.
6. E6 retrieval.
