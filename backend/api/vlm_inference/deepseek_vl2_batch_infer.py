#!/usr/bin/env python3
"""
DeepSeek-VL2 BATCH inference script - processes multiple images with one model load
Input: JSON on stdin with images list, classes, model
Output: JSON on stdout with detections for all images

This avoids reloading the model for each image.
"""
import sys
import os
import json
import re
from pathlib import Path

# Suppress all library output to stdout
import io
class StdoutRedirector:
    def __init__(self):
        self._original_stdout = sys.stdout
        self._buffer = io.StringIO()

    def __enter__(self):
        sys.stdout = self._buffer
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout

# Suppress warnings and logging
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def parse_detections(response_text: str, classes: list):
    """Parse VLM response text into detections"""
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not json_match:
            return []
        json_str = json_match.group()

    try:
        detections_raw = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    detections = []
    classes_lower = [c.lower() for c in classes]

    for det in detections_raw:
        if not isinstance(det, dict):
            continue

        class_name = det.get("class", "")
        if class_name.lower() not in classes_lower:
            continue

        class_idx = classes_lower.index(class_name.lower())
        class_name = classes[class_idx]

        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        try:
            confidence = float(det.get("confidence", 0.5))
            raw_values = [float(b) for b in bbox]

            if any(v > 1.0 for v in raw_values):
                x_min, y_min, x_max, y_max = [v / 100.0 for v in raw_values]
            else:
                x_min, y_min, x_max, y_max = raw_values

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            if width > 0.01 and height > 0.01:
                detections.append({
                    "class_name": class_name,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "confidence": confidence
                })
        except (ValueError, TypeError):
            continue

    return detections


def run_batch_inference(image_paths: list, classes: list, model_name: str):
    """Run DeepSeek-VL2 inference on multiple images with single model load"""
    # Suppress stdout during imports
    with StdoutRedirector():
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForCausalLM
            from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        except ImportError as e:
            pass

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    except ImportError as e:
        return {"error": f"Missing dependencies: {e}"}

    results = []

    try:
        # Load model ONCE
        with StdoutRedirector():
            vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_name)
            tokenizer = vl_chat_processor.tokenizer

            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            if torch.cuda.is_available():
                vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
            else:
                vl_gpt = vl_gpt.eval()

        class_list = ", ".join(classes)
        detection_prompt = f"""Detect all objects of these classes in the image: {class_list}

For each object found, provide its bounding box coordinates as percentages (0-100) of the image size.

Return a JSON array:
[{{"class": "classname", "bbox": [left, top, right, bottom], "confidence": 0.95}}]

If no objects found, return: []"""

        # Process each image
        for image_path in image_paths:
            try:
                with StdoutRedirector():
                    image = Image.open(image_path).convert("RGB")

                    conversation = [
                        {
                            "role": "<|User|>",
                            "content": f"<image>\n{detection_prompt}",
                            "images": [image],
                        },
                        {"role": "<|Assistant|>", "content": ""},
                    ]

                    prepare_inputs = vl_chat_processor(
                        conversations=conversation,
                        images=[image],
                        force_batchify=True,
                        system_prompt=""
                    )

                    if torch.cuda.is_available():
                        prepare_inputs = prepare_inputs.to(vl_gpt.device)

                    with torch.no_grad():
                        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                        outputs = vl_gpt.language.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=prepare_inputs.attention_mask,
                            pad_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=512,
                            do_sample=False,
                            use_cache=True,
                        )

                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                detections = parse_detections(generated_text, classes)
                results.append({
                    "image_path": image_path,
                    "detections": detections,
                    "tokens_used": len(outputs[0]),
                    "raw_response": generated_text[:500]
                })

            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "detections": []
                })

        return {"results": results}

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}


if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())

    result = run_batch_inference(
        image_paths=input_data["image_paths"],
        classes=input_data["classes"],
        model_name=input_data.get("model", "deepseek-ai/deepseek-vl2-tiny")
    )

    print(json.dumps(result))
