# Isolate Figure 2 by model pair

## Evidence and scope

The completed 43-run ablation batch contains Qwen–MiniLM S1 arms and full runs
for both Qwen–MiniLM and BGE–MiniLM. `figure_two` filters by ablation but not
pair, mixing both full models into its method point and seed error bars. The
training outputs and pair-specific tables are valid; only Figure 2 is affected.

## Decision

Infer the single model pair represented by S1 rows and include only that pair's
S1 and full rows. Reject missing or ambiguous S1 pair identity before drawing.
This preserves the existing CLI and avoids hard-coding Qwen or adding a new
option that could disagree with the coverage replay. The caller must continue
to supply coverage from the same pair, as required by the existing workflow.
The alternative of selecting a hard-coded default pair is more brittle; a
multi-pair figure redesign is outside this repair.

## Implementation and verification plan

1. Add a regression test with one S1 pair and another pair's full runs. Verify
   that the existing implementation incorrectly shifts the method point.
2. Add the pair guard and filter in `figure_two`; document the input contract.
3. Test correct method means, error bars and baseline labels, and rejection of
   missing or multiple S1 pairs in the server project virtual environment.
4. Back up the old plotting script and Figure 2 artifacts on the server. Deploy
   only the repair and test, rerun Figure 2, and compare the plotted point with
   Qwen's three seed records. Inspect the regenerated PNG.
5. Preserve all training outputs and CSV/LaTeX tables. Pause the heartbeat only
   after all outputs are verified and notify the user of completion and repair.

No training reruns, dependency installs, protocol changes, or GPU work are needed.
User instructions authorize implementing a concrete plan without another approval.

## Review

Scope, expected input identity and success checks are explicit. No outstanding
design decisions or placeholders remain.
