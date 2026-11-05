#!/usr/bin/env python3
"""Generate deterministic per-source visual attributes for AugCLIP."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from evaluate_omes_augclip import image_id, index_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("florence2", "qwen25vl"), default="florence2"
    )
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--domain", required=True)
    return parser.parse_args()


def parse_descriptions(text: str) -> list[str]:
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            values = json.loads(match.group(0))
            if isinstance(values, list):
                return [str(value).strip() for value in values if str(value).strip()]
        except json.JSONDecodeError:
            pass
    values = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if line:
            values.append(line.strip('"'))
    return values[:12]


def main() -> None:
    args = parse_args()
    existing = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "qwen25vl":
        model_name = args.model or "Qwen/Qwen2.5-VL-3B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )
    else:
        model_name = args.model or "microsoft/Florence-2-base-ft"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(args.device)
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
    prompt = (
        f"This image is from the {args.domain} editing dataset. "
        "List 10 distinct, factual visual attributes that are actually visible "
        "in the image. Cover the subject, appearance, pose or structure, colors, "
        "and surrounding context. Do not infer non-visual facts or mention the "
        "requested edit. Return only a valid JSON array of short strings."
    )
    indexed = index_images(args.image_dir.resolve())
    for position, (key, path) in enumerate(sorted(indexed.items(), key=lambda item: int(item[0]))):
        if key in existing and existing[key]:
            continue
        if args.backend == "florence2":
            image = Image.open(path).convert("RGB")
            captions = []
            for task in ("<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>", "<CAPTION>"):
                inputs = processor(
                    text=task, images=image, return_tensors="pt"
                ).to(args.device, torch.float16)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=256,
                        num_beams=3,
                        do_sample=False,
                    )
                decoded = processor.batch_decode(
                    generated, skip_special_tokens=False
                )[0]
                parsed = processor.post_process_generation(
                    decoded, task=task, image_size=image.size
                )[task]
                captions.append(parsed)
            descriptions = []
            for caption in captions:
                for phrase in re.split(r"(?<=[.!?;])\s+|,\s+(?=(?:and\s+)?(?:the|a|an)\s)", caption):
                    phrase = phrase.strip(" ,.;")
                    if phrase and phrase not in descriptions:
                        descriptions.append(phrase)
            if len(descriptions) < 4:
                descriptions = captions
            existing[key] = descriptions[:12]
            args.output.write_text(json.dumps(existing, indent=2))
            print(
                f"[{position + 1}/{len(indexed)}] {image_id(path)}: "
                f"{len(existing[key])} attributes"
            )
            continue
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        decoded = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        descriptions = parse_descriptions(decoded)
        if len(descriptions) < 4:
            raise RuntimeError(f"Could not parse descriptions for {path}: {decoded}")
        existing[key] = descriptions
        args.output.write_text(json.dumps(existing, indent=2))
        print(f"[{position + 1}/{len(indexed)}] {image_id(path)}: {len(descriptions)} attributes")


if __name__ == "__main__":
    main()
