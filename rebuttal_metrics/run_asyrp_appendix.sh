#!/usr/bin/env bash
set -euo pipefail

# Regenerate Appendix Table 12/13 Asyrp outputs from checkpoints that are
# present in Asyrp_official/checkpoint. Run from the Asyrp_official repository.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
official_root="$(cd "${repo_root}/../Asyrp_official" && pwd)"
device="${CUDA_DEVICE:-0}"

run_one() {
  local config="$1"
  local concept="$2"
  local checkpoint="$3"
  local output_name="$4"

  if [[ ! -f "${official_root}/checkpoint/${checkpoint}" ]]; then
    echo "MISSING CHECKPOINT: ${checkpoint}" >&2
    return 2
  fi

  (
    cd "${official_root}"
    CUDA_VISIBLE_DEVICES="${device}" python -u main.py \
      --run_test \
      --config "${config}" \
      --exp "./runs/${output_name}" \
      --edit_attr "${concept}" \
      --n_train_img 0 \
      --n_test_img 32 \
      --n_iter 1 \
      --bs_train 1 \
      --t_0 999 \
      --n_inv_step 40 \
      --n_train_step 40 \
      --n_test_step 40 \
      --get_h_num 1 \
      --user_defined_t_edit 500 \
      --user_defined_t_addnoise 200 \
      --train_delta_block \
      --use_x0_tensor \
      --save_x0 \
      --hs_coeff_delta_h 1.0 \
      --manual_checkpoint_name "${checkpoint}" \
      --sh_file_name "${repo_root}/rebuttal_metrics/run_asyrp_appendix.sh"
  )
}

case "${1:-all}" in
  church_gothic)
    run_one church.yml church_gothic \
      church_gothic_LC_church_outdoor_t999_ninv40_ngen40_0.pth \
      rebuttal_church_gothic
    ;;
  church_red_brick)
    run_one church.yml church_red_brick \
      church_red_brick_LC_church_outdoor_t999_ninv40_ngen40_0.pth \
      rebuttal_church_red_brick
    ;;
  church_snowy)
    run_one church.yml church_snow \
      church_snow_LC_church_outdoor_t999_ninv50_ngen50_4.pth \
      rebuttal_church_snowy
    ;;
  church_wooden)
    run_one church.yml church_wooden_church \
      church_wooden_church_LC_church_outdoor_t999_ninv40_ngen40_0.pth \
      rebuttal_church_wooden
    ;;
  dog_angry)
    run_one afhq.yml dog_angry \
      dog_angry_LC_dog_t999_ninv40_ngen40_0.pth \
      rebuttal_dog_angry
    ;;
  dog_sleepy)
    run_one afhq.yml dog_sleepy \
      dog_sleepy_LC_dog_t999_ninv40_ngen40_0.pth \
      rebuttal_dog_sleepy
    ;;
  dog_smiling)
    run_one afhq.yml dog_happy \
      dog_happy_LC_dog_t999_ninv40_ngen40_0.pth \
      rebuttal_dog_smiling
    ;;
  dog_puppy)
    run_one afhq.yml dog_puppy \
      rebuttal_dog_puppy_train_LC_dog_t999_ninv40_ngen40_0.pth \
      rebuttal_dog_puppy
    ;;
  all)
    for task in church_gothic church_red_brick church_snowy church_wooden \
                dog_angry dog_sleepy dog_smiling dog_puppy; do
      "$0" "${task}"
    done
    ;;
  *)
    echo "Unknown task: $1" >&2
    exit 2
    ;;
esac
