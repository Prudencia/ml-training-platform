#!/usr/bin/env python3
"""
DeepSeek-VL2 inference script - run via subprocess from venv
Input: JSON on stdin with image_path, classes, model
Output: JSON on stdout with detections

Requires: pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git
"""
import sys
import json
import re
from pathlib import Path

def parse_detections(response_text: str, classes: list):
    """Parse VLM response text into detections"""
    # Extract JSON from response
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

            # Handle both 0-1 and 0-100 ranges
            if any(v > 1.0 for v in raw_values):
                x_min, y_min, x_max, y_max = [v / 100.0 for v in raw_values]
            else:
                x_min, y_min, x_max, y_max = raw_values

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Clamp
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


def run_inference(image_path: str, classes: list, model_name: str):
    """Run DeepSeek-VL2 inference using official DeepSeek package"""
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    except ImportError as e:
        return {"error": f"Missing dependencies. Install with: pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git\nError: {e}"}

    try:
        # Load processor and model
        vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_name)
        tokenizer = vl_chat_processor.tokenizer

        # Load model with trust_remote_code
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        if torch.cuda.is_available():
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        else:
            vl_gpt = vl_gpt.eval()

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Build detection prompt
        class_list = ", ".join(classes)
        detection_prompt = f"""Detect all objects of these classes in the image: {class_list}

For each object found, provide its bounding box coordinates as percentages (0-100) of the image size.

Return a JSON array:
[{{"class": "classname", "bbox": [left, top, right, bottom], "confidence": 0.95}}]

If no objects found, return: []"""

        # DeepSeek-VL2 conversation format
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{detection_prompt}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Prepare inputs
        pil_images = [image]
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )

        if torch.cuda.is_available():
            prepare_inputs = prepare_inputs.to(vl_gpt.device)

        # Generate
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

        return {
            "detections": detections,
            "tokens_used": len(outputs[0]),
            "raw_response": generated_text[:500]
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}


if __name__ == "__main__":
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())

    result = run_inference(
        image_path=input_data["image_path"],
        classes=input_data["classes"],
        model_name=input_data.get("model", "deepseek-ai/deepseek-vl2-tiny")
    )

    # Output result as JSON
    print(json.dumps(result))
