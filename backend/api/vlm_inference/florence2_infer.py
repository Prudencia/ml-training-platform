#!/usr/bin/env python3
"""
Florence-2 inference script - run via subprocess from venv
Input: JSON on stdin with image_path, classes, model
Output: JSON on stdout with detections
"""
import sys
import json
from pathlib import Path

def run_inference(image_path: str, classes: list, model_name: str):
    """Run Florence-2 inference"""
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        from PIL import Image
        import torch
    except ImportError as e:
        return {"error": f"Missing dependencies: {e}"}

    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)

        # Load image
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # Run detection
        task = "<OPEN_VOCABULARY_DETECTION>"
        class_prompt = ", ".join(classes)
        text_input = f"{task}{class_prompt}"

        inputs = processor(
            text=text_input,
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(img_width, img_height)
        )

        # Extract bounding boxes
        detections = []
        result = parsed.get(task, {})
        bbox_list = result.get("bboxes", [])
        label_list = result.get("bboxes_labels", result.get("labels", []))

        classes_lower = [c.lower() for c in classes]

        for bbox, label in zip(bbox_list, label_list):
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            label_lower = label.lower().strip()

            # Match to requested classes
            matched_class = None
            for j, cls in enumerate(classes_lower):
                if cls in label_lower or label_lower in cls:
                    matched_class = classes[j]
                    break

            if matched_class is None and label_lower in classes_lower:
                matched_class = classes[classes_lower.index(label_lower)]

            if matched_class is None:
                continue

            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Clamp to 0-1
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            if width > 0.01 and height > 0.01:
                detections.append({
                    "class_name": matched_class,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "confidence": 0.9  # Florence-2 doesn't output confidence
                })

        return {
            "detections": detections,
            "tokens_used": len(generated_ids[0]),
            "raw_response": str(parsed)
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
        model_name=input_data.get("model", "microsoft/Florence-2-base")
    )

    # Output result as JSON
    print(json.dumps(result))
