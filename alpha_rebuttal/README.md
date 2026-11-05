# Minimal alpha rebuttal experiments

The experiments use CelebA-HQ IDs 29501--29508 and the five main concepts:
Smiling, Young, Big Nose, Beards, and Blond Hair.

## Reviewer 1: CASL alpha sweep

```bash
cd /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE
python alpha_rebuttal/evaluate_casl_alpha.py \
  --device cuda:0 --batch-size 16 --n-images 8
```

This evaluates `alpha={0,32,64,96,128,160,192}` with four independent
attribute classifiers, CLIP ViT-L/14, LPIPS-Alex, and the existing fixed
top-1 OS/MES/OMES results.

## Reviewer 2: native method sweeps

ASYRP grids are generated with the official implementation:

```bash
cd /p/realai/zhenghao/LoRA_SAE/Asyrp_official
python main.py --run_test --config celeba_p2.yml \
  --exp ./alpha_sweep/smiling --edit_attr smiling \
  --load_from_checkpoint smiling --n_iter 5 \
  --n_train_img 50 --n_test_img 32 --t_0 999 \
  --n_inv_step 50 --n_train_step 50 --n_test_step 100 \
  --target_image_id '0 1 2 3 4 5 6 7' \
  --sample_type ddpm --eta 0.0 --delta_interpolation \
  --min_delta 0 --max_delta 2 --num_delta 5 --ni 1
```

Repeat with the other four concept names, then run:

```bash
cd /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE
python alpha_rebuttal/extract_asyrp_alpha.py
python alpha_rebuttal/evaluate_epr_sweeps.py \
  --device cuda:0 --batch-size 32 --n-images 8
python alpha_rebuttal/summarize_results.py
```

The evaluator also reads the existing 20-setting Boundary sweeps and CASL
sweeps. All edit-induced changes are measured against the method's own
zero-intervention reconstruction (for Boundary, `recons.png`).

## Outputs

- `results/casl_alpha/per_image.csv`: Reviewer 1 per-image results.
- `results/casl_alpha/summary.csv`: Reviewer 1 concept/alpha means.
- `results/epr_sweeps/per_image.csv`: Reviewer 2 per-image classifier results.
- `results/epr_sweeps/summary.csv`: Reviewer 2 method/concept/strength means.
- `results/reviewer1_alpha_table.md`: full per-concept table.
- `results/reviewer2_epr_table.md`: compact macro table.
