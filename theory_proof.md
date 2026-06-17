# Prompt for Theoretical Analysis

You are an expert in machine learning theory, graph-based representation learning, knowledge distillation, retrieval embeddings, and AAAI/NeurIPS-style paper writing. Please write a LaTeX subsection named `\subsection{Theoretical Analysis}` for the HeatGeo paper.

Context of the paper:

- HeatGeo distills a large text embedding teacher into a compact student encoder.
- Teacher embeddings \(t_i=T(x_i)\) are cached over a corpus \(\mathcal{D}=\{x_i\}_{i=1}^N\).
- HeatGeo builds a mutual \(k\)-NN graph from teacher embeddings with adjacency:

$$
W^T_{ij}=
\begin{cases}
\exp(\cos(t_i,t_j)/\tau_g), & j\in \mathcal{M}_k^T(i),\\
0, & \text{otherwise}.
\end{cases}
$$

- The transition matrix is:
$$
P^T=(D^T)^{-1}W^T.
$$
- Multi-scale diffusion is defined as:
$$
P_r^T=(P^T)^r,\quad r\in\mathcal{R}, \quad \mathcal{R}=\{1,2,4\}.
$$
- For each anchor \(i\), the candidate set is:
$$
C_i=C_i^+\cup C_i^h\cup C_i^r,
$$
where \(C_i^+\) contains high-diffusion neighbors, \(C_i^h\) contains teacher hard negatives, and \(C_i^r\) contains random negatives.
- The teacher restricted diffusion target is:
$$
p^T_{i,r}(j)=
\frac{(P_r^T)_{ij}}
{\sum_{j'\in C_i}(P_r^T)_{ij'}},
\quad j\in C_i.
$$
- The student candidate distribution is:
$$
p_i^S(j)=
\frac{\exp(\cos(s_i,s_j)/\tau_s)}
{\sum_{j'\in C_i}\exp(\cos(s_i,s_{j'})/\tau_s)},
\quad j\in C_i.
$$
- The diffusion loss is:
$$
\mathcal{L}_{\mathrm{diff}}
=
\frac{1}{|B|}
\sum_{i\in B}
\sum_{r\in\mathcal{R}_e}
\omega_r
D_{\mathrm{KL}}(p^T_{i,r}\|p_i^S).
$$
- Spectral coordinates are computed from the teacher normalized graph Laplacian:
$$
L^T=I-(D^T)^{-1/2}W^T(D^T)^{-1/2}.
$$
Let \(U^T\in\mathbb{R}^{N\times m}\) contain the first \(m\) non-trivial eigenvectors, and let \(u_i^T\) be row \(i\). The student projection is:
$$
z_i^S=W_{\mathrm{spec}}s_i.
$$
- The spectral loss is:
$$
\mathcal{L}_{\mathrm{spec}}
=
\frac{1}{|B|}
\sum_{i\in B}
\|z_i^S-u_i^T\|_2^2.
$$
- The anchor loss is:
$$
\mathcal{L}_{\mathrm{anchor}}
=
\frac{1}{|B|}
\sum_{i\in B}
(1-\cos(W_as_i,t_i)).
$$
- The final objective contains only three losses:
$$
\mathcal{L}_{\mathrm{HeatGeo}}
=
\lambda_{\mathrm{diff}}\mathcal{L}_{\mathrm{diff}}
+
\lambda_{\mathrm{spec}}\mathcal{L}_{\mathrm{spec}}
+
\lambda_{\mathrm{anchor}}\mathcal{L}_{\mathrm{anchor}}.
$$

Important: Do not mention task loss, InfoNCE, or \(\mathcal{L}_{\mathrm{task}}\).

Writing requirements:

1. Write in LaTeX, in a serious AAAI-style paper tone, but make the theory slightly more elegant and novelty-emphasizing than a minimal proof sketch.
2. Use a style similar to the theory section of MCW-KD: formalize the new object, state theorem/lemma/proposition results, provide concise proofs, and then explain the implication for knowledge distillation. Do not copy MCW-KD. Do not discuss Wasserstein or optimal transport unless strictly necessary.
3. Include at least three theoretical results:

   - Lemma/Theorem 1: Diffusion KL implies neighborhood distribution preservation. Use Pinsker's inequality or total variation to prove that if \(D_{\mathrm{KL}}(p^T_{i,r}\|p_i^S)\le \delta\), then the student preserves semantic mass over every subset \(A\subseteq C_i\):
   $$
   |p_i^S(A)-p^T_{i,r}(A)|\le \sqrt{\delta/2}
   $$
   or an equivalent bound. Explain that HeatGeo directly preserves retrieval neighborhood mass rather than merely pointwise vectors.

   - Lemma/Theorem 2: Candidate restriction approximates full-corpus diffusion when captured mass is high. Define the full diffusion row \(q_{i,r}(j)=(P_r^T)_{ij}\), and the captured mass \(\rho_{i,r}(C_i)=\sum_{j\in C_i}q_{i,r}(j)\). If \(\rho_{i,r}(C_i)\ge 1-\epsilon\), then the restricted target \(p^T_{i,r}\) retains almost all teacher diffusion behavior. Provide a total variation or decomposition bound showing that the full-corpus discrepancy is controlled by the tail mass \(\epsilon\) plus the restricted mismatch. Prove the statement carefully and avoid overclaiming.

   - Proposition/Theorem 3: Spectral anchoring preserves low-frequency/global topology. Use the Laplacian eigenvector, Diffusion Maps, or Laplacian Eigenmaps intuition. If \(\mathcal{L}_{\mathrm{spec}}\le \eta\), then projected student coordinates \(z_i^S\) preserve teacher spectral coordinates \(u_i^T\) in mean square. Include a bound for pairwise spectral distance:
   $$
   \big|\|z_i^S-z_j^S\|_2-\|u_i^T-u_j^T\|_2\big|
   \le
   \|z_i^S-u_i^T\|_2+\|z_j^S-u_j^T\|_2.
   $$
   Then explain that low-frequency eigenvectors encode global community/manifold structure, so this loss anchors global topology.

4. Add one summary corollary: if the weighted HeatGeo objective is small, then the student simultaneously preserves local/multi-hop diffusion neighborhoods, candidate-level retrieval behavior, and low-frequency manifold topology. State this under clear assumptions and do not claim absolute preservation.
5. After each theorem/lemma/proposition, include a short paragraph titled or introduced as `Implication for HeatGeo.` explaining the meaning in paper language.
6. Use notation consistent with HeatGeo: \(p^T_{i,r}\), \(p_i^S\), \(C_i\), \(P_r^T\), \(\mathcal{L}_{\mathrm{diff}}\), and \(\mathcal{L}_{\mathrm{spec}}\).
7. Do not add new citations unless you are confident the bibliography key exists. If needed, only use likely existing keys: `\citet{belkin2003laplacian}` and `\citet{coifman2006diffusionmaps}`. Avoid citing Pinsker with a bib key; simply refer to Pinsker's inequality.
8. Output only the complete LaTeX subsection that can be pasted directly into the paper. Do not include explanations outside LaTeX.
