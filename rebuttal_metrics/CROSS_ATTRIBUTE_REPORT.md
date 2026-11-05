# Cross-attribute SAE probe

## Protocol

- Frozen CelebA-HQ diffusion model and paper SAE (512 to 65,536).
- Independent images 20001--24096 (4,096 total), fixed noise seed 2026 and
  timestep 509.
- The 8 by 8 spatial layout is retained and flattened, matching Table 2.
- The 679 latents with standard deviation at least 0.01 define the active pool.
- CASL uses the stored Stage-2 top-k ranking for each concept. Random samples
  the same number of active latents without replacement (100 repeats).
- For each selected concept, the identical fixed features train five
  independent LinearSVC probes. No Stage-2 mapping is applied during probing.
- One fixed 80/20 split (3,276/820 images); metrics are held-out Accuracy and
  AUROC.

## Macro results

|Selection/$k$|Target Acc. $\uparrow$|Target AUROC $\uparrow$|Non-target Acc. $\downarrow$|Non-target AUROC $\downarrow$|AUROC gap $\uparrow$|
|---|---:|---:|---:|---:|---:|
|CASL/1|0.767|0.760|0.769|0.772|-0.013|
|Random/1|0.763|0.741|0.763|0.741|0.000|
|CASL/2|0.801|0.821|0.792|0.814|0.006|
|Random/2|0.786|0.793|0.786|0.794|-0.001|
|CASL/4|0.816|0.848|0.817|0.850|-0.002|
|Random/4|0.808|0.833|0.808|0.834|0.000|
|CASL/8|0.830|0.865|0.823|0.863|0.002|
|Random/8|0.817|0.852|0.818|0.854|-0.002|
|CASL/16|0.815|0.860|0.816|0.861|-0.001|
|Random/16|0.814|0.854|0.815|0.854|0.000|

Random entries are means over 100 matched selections.

## Cross-attribute results

Each cell is Accuracy/AUROC. Rows identify the concept used to select the
fixed CASL features; columns identify the attribute predicted by the SVM.

### k=1

|Selected for|Beards|Big Nose|Blond Hair|Smiling|Young|
|---|---:|---:|---:|---:|---:|
|Beards|**0.802/0.853**|0.726/0.738|0.857/0.798|0.650/0.687|0.782/0.751|
|Big Nose|0.811/0.868|**0.750/0.758**|0.865/0.802|0.641/0.685|0.783/0.729|
|Blond Hair|0.790/0.813|0.738/0.764|**0.850/0.770**|0.715/0.796|0.789/0.765|
|Smiling|0.802/0.853|0.726/0.738|0.857/0.798|**0.650/0.687**|0.782/0.751|
|Young|0.811/0.868|0.750/0.758|0.865/0.802|0.641/0.685|**0.783/0.729**|

### k=16

|Selected for|Beards|Big Nose|Blond Hair|Smiling|Young|
|---|---:|---:|---:|---:|---:|
|Beards|**0.859/0.922**|0.698/0.751|0.905/0.936|0.794/0.878|0.810/0.828|
|Big Nose|0.841/0.910|**0.727/0.758**|0.905/0.930|0.795/0.875|0.821/0.838|
|Blond Hair|0.839/0.901|0.723/0.754|**0.902/0.933**|0.795/0.873|0.818/0.839|
|Smiling|0.849/0.918|0.728/0.752|0.905/0.919|**0.783/0.871**|0.796/0.806|
|Young|0.860/0.919|0.732/0.768|0.917/0.946|0.791/0.875|**0.806/0.814**|

The complete k=1,2,4,8,16 matrices are in
`cross_attribute_per_probe.csv`.

## Diagnosis

The experiment does not support a claim that every concept is isolated in a
separate SAE unit or subspace. Beards is selective at all budgets, Blond Hair
is selective at larger budgets, and Smiling becomes weakly selective at k=8
and k=16. Big Nose and Young remain mixed. At the macro level, target and
non-target AUROC are nearly equal.

The Stage-2 rankings also overlap substantially: the five rankings contain
only 3 unique units at k=1 and 30 unique units among 80 selections at k=16.
CelebA attribute correlations additionally raise non-target prediction for
both CASL and random active units. Therefore, max-activating-image coherence
should be described as evidence of interpretability, not proof of complete
disentanglement. The supported conclusion is that CASL aligns distributed SAE
features, with concept-dependent residual entanglement.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python rebuttal_metrics/extract_celeba_sae_spatial_active.py \
  --batch-size 32 \
  --output-dir /u/wff7ad/casl_cross_attribute_cache

python rebuttal_metrics/evaluate_cross_attribute_probe.py \
  --representation /u/wff7ad/casl_cross_attribute_cache/representations_t509_spatial_active.npy \
  --active-indices /u/wff7ad/casl_cross_attribute_cache/active_indices_threshold0.01.npy \
  --ks 1 2 4 8 16 \
  --random-repeats 100 \
  --n-jobs 28 \
  --output-dir rebuttal_metrics/results/celeba_sae_omes_independent/cross_attribute_probe_spatial
```
