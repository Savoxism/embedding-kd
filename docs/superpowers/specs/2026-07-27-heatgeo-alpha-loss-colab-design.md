# HeatGeo Alpha Loss and Colab Archive Design

## Scope

Update only the HeatGeo training path and its Colab reproduction notebook:

1. Keep the Colab training batch size at 16.
2. Rename the expected archive and extraction workspace from
   `ICLR_MDD_npa_test_1` to `ICLR_MDD_pkc_test_1`.
3. Replace the four-term HeatGeo objective with an alpha-weighted task and
   diffusion objective using `alpha = 1.0`.

Other distillation methods are out of scope.

## Loss Configuration

`HeatGeoConfig` will expose `alpha = 1.0` as the source of truth. Spectral and
anchor weights will be set to zero.

The HeatGeo criterion will compute:

\[
\mathcal{L}_{\mathrm{HeatGeo}}
=
(1-\alpha)\mathcal{L}_{\mathrm{task}}
+
\alpha\mathcal{L}_{\mathrm{diff}}
+
0\mathcal{L}_{\mathrm{spec}}
+
0\mathcal{L}_{\mathrm{anchor}}.
\]

For `alpha = 1.0`, this is:

\[
\mathcal{L}_{\mathrm{HeatGeo}}
=
\mathcal{L}_{\mathrm{diff}}.
\]

The existing loss metrics remain available so training logs retain their
current schema.

## Notebook Changes

`test_mdd.ipynb` will:

- expect
  `/content/drive/MyDrive/[ICLR] Embedding KD/ICLR_MDD_pkc_test_1.zip`;
- extract into `/content/ICLR_MDD_pkc_test_1_workspace`;
- keep `BATCH_SIZE` equal to `16`;
- verify `alpha == 1.0`, `lambda_spec == 0`, and `lambda_anchor == 0` before
  starting the expensive model-loading and training stages.

## Documentation

The HeatGeo README configuration and objective will be updated to describe the
alpha-weighted two-loss objective. The method paper is not changed because this
request configures the reproduction run rather than redefining the general
method.

## Verification

Implementation is complete when:

1. The notebook contains no `ICLR_MDD_npa_test_1` references and still sets
   batch size to 16.
2. A default `HeatGeoConfig` reports `alpha = 1.0`,
   `lambda_spec = 0`, and `lambda_anchor = 0`.
3. A synthetic HeatGeo forward pass numerically equals
   `loss_diff`.
4. HeatGeo imports and the notebook JSON structure remain valid.
