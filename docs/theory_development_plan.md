# HeatGeo — Plan phát triển Theory, Title/Abstract, và Ablation (v2)

> Chuỗi narrative: *mini-batch không đủ → phải align local geometry → local ⇒ global → kNN đắt → random walk/sampling xấp xỉ kNN → limitation của L_diff ⇒ L_SGC*.
>
> **Nguyên tắc viết (v2):** Section theory phát biểu hoàn toàn **symbolic** (N, k, B, m, r, ε, δ) — không nhắc tên log, không nhắc giá trị hyperparameter, không nhắc chi tiết implementation. Mọi hằng số đo được và mọi con số instantiated (0.41 neighbor/batch, 66%, truncation mass, perplexity...) chuyển xuống **§Analysis (5.3) và Appendix** dưới dạng "verifying the assumptions / instantiating the bounds". Theory nói *rate và cơ chế*; Analysis nói *con số*.

---

## 1. Survey hiện trạng paper

- `sections/`: 00_abstract, 01_introduction, 02_related_work, **04_methodology**, 05_experiments — **section 03 trống**, đúng chỗ cho Theory.
- Kết quả lý thuyết duy nhất: Proposition collapse trong §4.4 (đúng, gọn, nhưng đứng một mình).
- Các claim chưa chứng minh: "graph ≈ manifold" (§4.1); "matching diffusion ⇒ inherit geometry" (Intro/§4.2); mini-batch relational "bounded by the sampled batch" (§2.3, chỉ nói lời); sparse walk trung thành với walk thật (§4.2); candidate set nhỏ là đủ (§4.3); vai trò negatives (§4.3); SGC identifiability đầy đủ (§4.4).
- Lỗi tiện thể: abstract ghi 78.38 (bảng: 78.56) và chỉ nói "ranks second" trong khi HeatGeo **thắng overall 2/3 settings** — sửa cùng đợt abstract mới (§5 dưới).
- Implementation đo sẵn nhiều đại lượng khớp với hằng số trong bound (truncation mass, residual mass, entropy target, JS floor). **V2: các số này KHÔNG xuất hiện trong theory section**; chúng là nguyên liệu cho §5.3 Analysis + Appendix ("assumptions verified, constants measured" vẫn là selling point, nhưng sống ở phần thực nghiệm).

## 2. Khung phát biểu để mọi claim đều provable

1. **Mini-batch: không claim impossibility.** Với relational loss dạng MSE trên similarity matrix, estimator qua batch unbiased trong kỳ vọng. Cái chứng minh được: (i) tín hiệu local chiếm tỉ lệ O(k/N) trong loss; (ii) tần suất quan sát neighbor-pair thấp ⇒ sample complexity Θ((N/B)·k log k) epochs để phủ neighborhood; (iii) với in-batch softmax target: **bias thật sự**, không triệt tiêu khi tăng epoch.
2. **Local ⇒ global cần operator phía student.** L_diff chỉ khống chế phân phối restricted trên candidate set. Cầu nối: chain rule của KL + kiểm soát mass ngoài support — và đây chính là **vai trò lý thuyết của hard/random negatives** (leakage control, không phải trick).
3. **"Random walk ≈ kNN"** = truncation bound + weighted-sampling-without-replacement + coverage theo perplexity của target (không theo k).

## 3. Chuỗi kết quả (Theorem A–D)

Setup (§3.0): corpus D, N điểm; teacher embeddings t_i trên unit sphere; mutual-kNN graph, lazy transition P̃ = (I+P)/2; target p_{i,r}; student scores c_i(j)=cos(s_i,s_j); batch size B; candidate size m; T epochs. **Toàn bộ phát biểu bằng ký hiệu — không số cụ thể.**

### Theorem A — Mini-batch relational matching undersamples local geometry
- **A1 (Coverage).** E[#neighbors trong batch] = (B−1)k/(N−1); P(không có neighbor) = (1−k/(N−1))^{B−1}; coverage neighbor-pair mỗi epoch O(B/N); coupon-collector ⇒ Θ((N/B)·log k / k · k log k)-cỡ epochs để quan sát đủ neighborhood. Chứng minh: đếm xác suất + coupon collector. *Rủi ro thấp, 1 ngày.*
- **A2 (Bias của in-batch softmax).** Dưới margin assumption (cos tới neighbor ≥ cos bulk + Δ): E_B[TV(batch-target, local target)] ≥ (1−k/N)^{B−1}(1−o(1)) — bias không triệt tiêu với B cố định. Chứng minh: tách theo biến cố batch∩N_k(i)=∅ + bound softmax bằng margin. *Rủi ro thấp–vừa, 2 ngày.*
- **A3 (remark, cite-only).** Uniform vertex sampling không xấp xỉ Laplacian của kNN graph (coherence cao — Gittens & Mahoney 2013; Tropp 2015). Không chứng minh.
- Margin assumption và instantiation bằng config thật → **đưa xuống §5.3/Appendix**, không nằm trong statement.

### Theorem B — Matching local diffusion transfers global geometry
- **B0 (Assumption M1, cite-only).** kNN-graph diffusion → heat semigroup của manifold (Coifman–Lafon; von Luxburg/Hein; García Trillos–Slepčev 2018; Calder–García Trillos 2022; Bernstein 2000; Alamgir–von Luxburg 2012). Mọi theorem chính phát biểu trên discrete graph, không phụ thuộc M1.
- **B1 (Perturbation).** max_i TV(P^S_i, P^T_i) ≤ ε ⇒ max_i TV((P^S)^r_i, (P^T)^r_i) ≤ rε (telescoping; lazy walk cho hằng số rε/2); + Pinsker: KL hàng ≤ ε_KL ⇒ global diffusion geometry sai khác ≤ r√(ε_KL/2). Bản average-case qua Jensen (L_diff khống chế trung bình, không phải sup). *Rủi ro thấp, 1–2 ngày. Đối chiếu Mitrophanov 2005.*
- **B2 (Restriction — vai trò negatives).** Chain rule: KL(p∥q) = p(C)·KL(p_C∥q_C) + KL_bin(p(C)∥q(C)) + p(C̄)·KL(p_C̄∥q_C̄). L_diff khống chế term 1; negatives khống chế term 2 (ép student mass ngoài support); phần dư chặn bởi residual mass δ. ⇒ restricted matching + negatives ⇒ full-row matching sai số δ. *Điểm mới conceptual nhất. Rủi ro vừa, 2 ngày.*
- **B3 (Multi-scale tightening).** Supervise trực tiếp tại mỗi r cho sai số ε_r per scale thay vì compound r·ε₁ ⇒ justification định lượng cho scale family. *Rủi ro thấp, 1 ngày.*
- **B4.** Chuyển Prop collapse hiện tại từ §4.4 vào đây (trả lời "vì sao cần τ_r riêng").

### Theorem C — Diffusion sampling approximates the kNN neighborhood
- **C1 (Truncation).** Truncated lazy walk (top-K cột mỗi bước, renormalize) sai khác TV ≤ 1−Π_s(1−δ_s) so với walk đầy đủ. Phát biểu bằng δ_s trừu tượng; giá trị thật → §5.3. *Rủi ro thấp, 1 ngày.*
- **C2 (cite-only).** Gumbel-top-k = weighted sampling without replacement chính xác (Efraimidis–Spirakis 2006; Kool 2019).
- **C3 (Coverage theo perplexity).** C rút ∝ diffusion mass ⇒ E_C[TV(p_C, p)] ≤ E[p(C̄)] + δ_trunc, với E[p(C̄)] ≤ Σ_j p_j(1−p_j)^m kiểm soát bởi **perplexity e^{H(p)}** — *cỡ mẫu cần thiết scale theo độ sharp của target, không theo k*. Resampling mỗi epoch ⇒ kỳ vọng estimator → full objective. *Rủi ro vừa, 2–3 ngày.*
- **C4.** Chi phí: per-anchor O(m) so với O(k·d_avg^{r−1}) — một đoạn ngắn, symbolic (không đưa số k=200, m=64 vào đây).

### Theorem D — Gauge của L_diff ⇒ L_SGC (identifiability) — làm đầu tiên
- **D1.** L_diff bất biến với shift per-anchor c_i ↦ c_i + a_i; đây là toàn bộ nhóm bất biến khi support đủ giàu, ≥2 temperature.
- **D2.** L_diff = 0 ⇔ c^S_i = c^T_i + a_i trên support (softmax xác định *gaps*).
- **D3.** Global-threshold tasks không bất biến với {a_i} ⇒ L_diff để hở đúng 1 DOF/anchor mà downstream cần.
- **D4.** Thay vào SGC: sai lệch trung bình có trọng số = a_i ⇒ L_diff + λL_SGC = 0 ⇔ student cosines = teacher cosines trên pooled columns. SGC = 1 ràng buộc/anchor khớp đúng 1 chiều gauge — **minimal sufficient calibration**.
- **D5 (optional).** Stationary point = log-linear compromise giữa các scale: Σ_r (ω_r/τ_r)(softmax(c/τ_r) − p_r) = 0.
- *Thuần đại số. Rủi ro thấp, 1–2 ngày.*

## 4. Cấu trúc `sections/03_theory.tex`

```
Section 3: From Mini-Batches to Manifold Diffusion
  3.0 Setup and notation                                  (~0.3 trang, symbolic)
  3.1 Mini-batch relations undersample local geometry     [A1, A2; remark A3]
  3.2 Local matching transfers global geometry            [B1–B3 + collapse (B4); M1 nêu như assumption]
  3.3 Sampled neighborhoods suffice                       [C1–C4]
  3.4 The gauge of L_diff and calibration                 [D1–D4]
```
- Body: statement + 1–2 câu proof idea. **Không số liệu, không log, không hyperparam.** Proofs → Appendix A.
- Mỗi theorem kết bằng một forward-reference: *"Section 5.3 instantiates this bound and tests its prediction."* — đó là cách duy nhất theory chạm tới thực nghiệm.
- §4 methodology refer ngược: graph→3.2, candidates→3.3, SGC→3.4; Prop collapse rời khỏi §4.

## 5. Title & Abstract đề xuất

### 5.1 Title (hiện tại: "HeatGeo: Manifold Diffusion for Text Embedding Distillation" — generic, không nêu claim)

| # | Đề xuất | Lý do |
|---|---|---|
| 1 ⭐ | **HeatGeo: Local Diffusion Matching Transfers Global Embedding Geometry** | Statement-style, nêu đúng theorem trung tâm (local⇒global); giữ brand |
| 2 | Beyond Mini-Batch Relations: Heat-Diffusion Distillation of Teacher Embedding Manifolds | Định vị đối đầu trực tiếp với dòng relational KD — hợp narrative Theorem A |
| 3 | HeatGeo: Distilling Global Embedding Geometry from Local Diffusion Neighborhoods | Trung tính hơn #1, vẫn nêu local→global |
| 4 | From Mini-Batches to Manifolds: Provable Geometry Distillation for Compact Text Embeddings | Theory-forward; bỏ brand HeatGeo khỏi title chính |

Khuyến nghị: **#1** (hoặc #3 nếu muốn bớt tính khẳng định trước khi B1–B2 hoàn tất).

### 5.2 Abstract mới (draft, ~190 từ — sửa luôn 78.38→78.56 và "second overall"→"wins 2/3")

> Compact text embedding models power semantic search, clustering, and retrieval-augmented generation, and distillation from a strong teacher is the standard way to train them. The dominant relational objective matches teacher and student similarities inside a mini-batch, but a mini-batch is a uniform sample of the corpus: an anchor's semantic neighbors rarely appear in it, so the local geometry of the teacher space is supervised at a vanishing rate. We propose **HeatGeo**, which replaces batch relations with anchored local supervision. HeatGeo builds a mutual $k$-nearest-neighbor graph over cached teacher embeddings, treats it as a discrete semantic manifold, and trains the student to match a family of heat-diffusion distributions at scale-dependent resolutions over sampled candidate neighborhoods, using only final teacher embeddings---no hidden states, online teacher passes, or task labels. We support this design with theory: mini-batch relational objectives provably undersample local neighborhoods; matching one-step diffusion locally controls multi-step global geometry; diffusion-weighted sampling approximates full neighborhoods at a cost set by target sharpness rather than neighborhood size; and the diffusion loss leaves exactly one similarity offset per anchor undetermined, which a single calibration term fixes. Across nine datasets and three teacher--student pairs, HeatGeo attains the best overall average among compact students in two of three settings and second best in the third, with the largest gains on semantic textual similarity.

Ghi chú: 4 mệnh đề theory trong abstract map đúng A→B→C→D; nếu B2 không kịp chặt thì làm mềm "provably" thành "we show".

## 6. Ablation / Analysis bổ trợ theory (§5.3 mới + Appendix)

Mỗi thí nghiệm gắn với theorem nó kiểm chứng và có **dự đoán falsifiable** viết trước. Đây cũng là nơi duy nhất các hằng số đo được (truncation mass, residual mass, perplexity, JS floor) và các con số instantiated (0.41 neighbor/batch, 66% batch rỗng...) xuất hiện.

| # | Thí nghiệm | Theorem | Dự đoán từ theory | Chi phí |
|---|---|---|---|---|
| E1 | **HeatGeo-batch**: cùng loss nhưng candidates = thành viên batch (bỏ anchored candidate sets); sweep B ∈ {32, 128, 512} | A | Điểm tăng chậm theo neighbor-coverage (B−1)k/N; kém HeatGeo rõ ở mọi B khả thi | 3 runs/setting |
| E2 | Đo P(batch chứa ≥1 neighbor) và TV(batch-target, local target) trên teacher graph thật theo B | A1–A2 | Khớp công thức đếm; TV ≥ mức margin-bound dự đoán | Offline, rẻ |
| E3 | Sweep graph **k ∈ {50, 100, 200, 400}** | B0–B1 | Tăng rồi bão hòa/suy giảm nhẹ (noise edges vào graph); vùng ổn định rộng | 4 runs |
| E4 | Scale family: {1} / {0,1} / {0,1,2,4} (default) / {0,1,2,4,8} | B3 | Multi-scale > single-scale; lợi ích giảm dần khi r lớn (error compound bù trừ) | 4 runs |
| E5 | Tied τ (mọi scale chung 1 temperature) vs distinct τ_r | B4 (collapse) | Tied τ ≈ single-scale với mixture target — gap rõ trên STS (gaps vs ranking) | 2 runs |
| E6 | Bỏ hard negatives / bỏ random negatives / bỏ cả hai | B2 | Student mass rò ra ngoài support tăng (đo trực tiếp); giảm điểm pair-classification trước tiên | 3 runs |
| E7 | Sweep candidate size **m ∈ {16, 32, 64, 128, 256}** | C3 | Bão hòa khi m vượt perplexity của target (đo được, ≪ k) — đường cong phẳng sớm | 5 runs |
| E8 | Frozen candidate set (1 lần) vs resample mỗi epoch | C3 | Frozen kém hơn và gap tăng theo epoch (estimator refit một mẫu) | 2 runs |
| E9 | Sweep walk truncation top-K (δ_trunc từ ~0 đến lớn) | C1 | Robust cho δ nhỏ; suy giảm khi δ_trunc vượt ngưỡng vài % — vẽ điểm vs δ đo được | 3 runs |
| E10 | λ_SGC = 0 vs default; đo phân bố offset per-anchor â_i | D | Không SGC: â_i phân tán; **giảm mạnh ở threshold-based tasks (pair-classification AP, STS Spearman cross-anchor) nhưng ít ảnh hưởng ranking nội anchor** — dự đoán sắc nhất, nên làm nổi bật | 2 runs |
| E11 | Compute: wall-clock/memory build + train theo k và m | C4 | Chi phí train theo m, không theo k; build một lần offline | Đo từ runs sẵn |

- **Ưu tiên nếu thiếu compute**: E10, E5, E7, E1 (mỗi cái đóng đinh một theorem trung tâm D, B4, C, A) — chạy trên 1 setting (BGE-M3 → MiniLMv2-H768, setting HeatGeo thắng rõ), các sweep còn lại 1 setting + appendix.
- Trình bày: §5.3 "Theory-driven analysis" với mỗi đoạn mở bằng *"Theorem X predicts …"* rồi đưa figure/table — cấu trúc này biến theory thành falsifiable thay vì trang trí.
- Appendix B: "Measured constants" — bảng truncation mass, residual mass, target perplexity, JS floor, degenerate-anchor rate cho cả 3 settings.

## 7. Citations cần thêm vào `heatgeo.bib`
- Mitrophanov 2005; von Luxburg et al. (consistency); García Trillos & Slepčev 2018; Calder & García Trillos 2022; Bernstein et al. 2000; Alamgir & von Luxburg 2012.
- Efraimidis & Spirakis 2006; Kool et al. 2019.
- Chuang et al. 2020 (debiased contrastive); Chen et al. 2020 (SimCLR).
- Gittens & Mahoney 2013 / Tropp 2015 (nếu giữ remark A3); Owen (Monte Carlo) nếu cần cho C3.

## 8. Thứ tự thực hiện

| Bước | Nội dung | Effort | Rủi ro |
|---|---|---|---|
| 1 | **D** (gauge + identifiability) — nâng cấp trực tiếp §4.4 | 1–2 ngày | Thấp |
| 2 | **B1, B3** | 2 ngày | Thấp |
| 3 | **B2** (restriction + negatives) | 2 ngày | Vừa |
| 4 | **C1–C3** | 2–3 ngày | Vừa |
| 5 | **A1, A2** | 2–3 ngày | Thấp–vừa |
| 6 | Viết 03_theory.tex + Appendix A proofs | 2–3 ngày | — |
| 7 | Title + abstract chốt (theo §5) + sửa intro contributions | 0.5 ngày | — |
| 8 | Ablation runs E1–E11 (ưu tiên E10, E5, E7, E1) + §5.3 + Appendix B | song song, ~1 tuần GPU | — |

Tổng: ~2 tuần người + ~1 tuần GPU chạy song song.

## 9. Rủi ro & phòng bị
- Không claim impossibility cho mini-batch — chỉ sample-complexity/bias với margin assumption (verify trong E2).
- Student không phải Markov chain — B2 xử lý; fallback: định nghĩa student-induced graph bằng cùng operator.
- M1 chỉ là assumption có văn hiến; theorems chính đứng trên discrete graph.
- Anchors degenerate (one-hot target, component nhỏ): loại trừ bằng non-degeneracy assumption trong C; tỉ lệ báo cáo ở Appendix B.
