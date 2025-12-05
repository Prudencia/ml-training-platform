#!/usr/bin/env python3
"""
DeepSeek-VL2 inference script - run via subprocess from venv
Input: JSON on stdin with image_path, classes, model
Output: JSON on stdout with detections
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
    """Run DeepSeek-VL2 inference"""
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        from PIL import Image
        import torch
    except ImportError as e:
        return {"error": f"Missing dependencies: {e}"}

    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )

        if not torch.cuda.is_available():
            model = model.to(device)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Build detection prompt
        class_list = ", ".join(classes)
        detection_prompt = f"""Detect all objects of these classes in the image: {class_list}

For each object found, provide its bounding box coordinates as percentages (0-100) of the image size.

Return a JSON array:
[{{"class": "classname", "bbox": [left, top, right, bottom], "confidence": 0.95}}]

If no objects found, return: []"""

        # DeepSeek-VL2 uses chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": detection_prompt}
                ]
            }
        ]

        inputs = processor(messages, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.1
            )

        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0
        generated_text = processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True
        )

        detections = parse_detections(generated_text, classes)

        return {
            "detections": detections,
            "tokens_used": len(generated_ids[0]),
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
