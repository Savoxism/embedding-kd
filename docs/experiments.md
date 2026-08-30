# Các thực nghiệm còn thiếu

## Ưu tiên bắt buộc

### 1. Motivating experiment — mini-batch relation gap

- **Chạy:** batch-local relational KD trên một teacher–student pair đại diện, 3 seeds. Theo dõi theo epoch: teacher-neighbor coverage, in-batch KL, full-corpus JS và teacher-neighbor Recall@$k$.
- **Figure 1:** sơ đồ global neighbor vs. batch-local neighbor; bên cạnh là coverage và topology distortion trong khi training loss vẫn giảm.
- **Cần thiết vì:** biến Proposition 1 từ một kết quả coverage thuần lý thuyết thành bằng chứng rằng missing support thực sự làm hỏng global geometry.

### 2. Core causal ablation

- **Chạy cùng budget:** batch-local KD; random corpus neighbors; degree-preserving random graph; RIPPLE one-step; one-step + diffusion; full RIPPLE (+ row supervision). Tối thiểu 3 seeds trên BGE-M3 → MiniLM-H768.
- **Đo:** downstream Avg., full-corpus JS và held-out diffusion error $E_8$; báo thêm $E_{16}$ trong appendix.
- **Figure 2:** ba panel tương ứng với ba metric chính, theo đúng thứ tự causal ở trên.
- **Cần thiết vì:** tách riêng tác dụng của corpus anchoring, teacher adjacency, multi-hop diffusion và row supervision; tránh kết luận rằng gain chỉ đến từ nhiều comparisons hoặc graph regularization.

### 3. Topology coverage sweep

- **Chạy:** chỉ thay $k\in\{50,100,200,400\}$; giữ nguyên candidate budget và optimizer.
- **Đo:** Mass@$k$, JS trên các teacher-rank band $1{:}50$, $51{:}200$, $201{:}800$, và full-corpus JS.
- **Figure 3a:** $k$ theo retained mass và rank-band JS.
- **Cần thiết vì:** improvement ngoài support trực tiếp là bằng chứng rõ nhất cho local-to-global propagation; Mass@$k$ tăng một mình không đủ.

### 4. Sampling fidelity và efficiency

- **Chạy:** chỉ thay candidate budget $m\in\{16,32,66,128,\text{full}\}$ trên cùng hardware.
- **Đo:** retained target mass, sampled-to-full gradient cosine, downstream Avg., step time và peak memory.
- **Figure 3b–c:** fidelity theo $m$; accuracy–time/memory trade-off.
- **Cần thiết vì:** kiểm chứng rằng mass-aware sampling giữ được tín hiệu full-corpus khi $m\ll k$, thay vì chỉ giảm compute bằng một approximation không được kiểm soát.

## Diagnostic để appendix

- **Student leakage:** đo $\delta^S$ theo candidate budget và so sánh có/không có ambient negatives. Cần để hỗ trợ Corollary về sampled geometry.
- **Row consistency:** với $\lambda=0$ và full RIPPLE, đo trung bình $|a_u-a_v|$ trên reciprocal-overlap edges. Cần để kiểm chứng trực tiếp vai trò của row supervision, thay vì chỉ dựa vào downstream Avg.
- **Cross-setting check:** lặp ba biến thể quan trọng — batch-local, RIPPLE one-step, full RIPPLE — trên hai teacher–student pairs còn lại, 3 seeds.

## Figures cần xuất

1. `latex/figures/figure1_motivation.pdf` — coverage và topology distortion của batch-local KD.
2. `latex/figures/figure2_core_ablation.pdf` — causal ablation của các thành phần RIPPLE.
3. `latex/figures/figure3_mechanisms.pdf` — topology coverage và sampling efficiency.

Không cần thêm bảng lớn trong main paper. Full sweeps, $E_{16}$, leakage và row-consistency nên để ở appendix. Main table chỉ cần chạy lại nếu chưa có raw logs đầy đủ cho 3 seeds và cùng checkpoint-selection protocol.
