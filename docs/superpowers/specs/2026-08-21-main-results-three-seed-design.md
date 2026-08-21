# Three-seed main-results table update

## Scope

Update only `latex/tables/main_results.tex` using the authoritative three-seed
TALAS and RIPPLE CSV summaries. Keep all teacher and previously reported
baseline values unchanged.

## Data presentation

- Show every TALAS and RIPPLE result as `mean \pm sample standard deviation`,
  rounded to two decimal places.
- Rank methods by the displayed mean only; the standard deviation does not
  affect rank.
- Within each teacher--student setting and metric column, bold the highest
  compact-model mean and underline the second-highest compact-model mean.
- Exclude the teacher row from ranking. Include all student/baseline rows,
  including the undistilled student and SimCSE-unsup.
- Preserve the existing table layout and method ordering.

## Caption

State that both TALAS and RIPPLE are three-seed mean-plus-standard-deviation
results. Keep the attribution that the remaining baseline values come from the
TALAS paper. Remove the obsolete single-checkpoint statement for setting (c).

## Verification

1. Reconcile all 72 TALAS/RIPPLE metric cells against the two CSV summaries.
2. Recompute top-1 and top-2 marks independently for all 36 setting/metric
   columns.
3. Compile `latex/main.tex` with `latexmk -pdf -interaction=nonstopmode
   -halt-on-error` and require a zero exit status.
