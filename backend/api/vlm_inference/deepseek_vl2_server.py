#!/usr/bin/env python3
"""
DeepSeek-VL2 inference SERVER - keeps model loaded in memory
Run as: python deepseek_vl2_server.py --port 8765

The model is loaded ONCE at startup and stays in memory.
Receives inference requests via HTTP POST.
"""
import sys
import os
import json
import re
import argparse
from pathlib import Path

# Suppress warnings before importing heavy libraries
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Global model variables - loaded once at startup
vl_chat_processor = None
tokenizer = None
vl_gpt = None
torch = None
Image = None


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


def load_model(model_name: str):
    """Load model into global variables"""
    global vl_chat_processor, tokenizer, vl_gpt, torch, Image

    print(f"Loading DeepSeek-VL2 model: {model_name}", file=sys.stderr)

    import torch as _torch
    from PIL import Image as _Image
    from transformers import AutoModelForCausalLM
    from deepseek_vl2.models import DeepseekVLV2Processor

    torch = _torch
    Image = _Image

    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_name)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        print(f"Model loaded on GPU", file=sys.stderr)
    else:
        vl_gpt = vl_gpt.eval()
        print(f"Model loaded on CPU", file=sys.stderr)

    print(f"DeepSeek-VL2 model ready!", file=sys.stderr)


def run_inference(image_path: str, classes: list):
    """Run inference on a single image using loaded model"""
    global vl_chat_processor, tokenizer, vl_gpt, torch, Image

    try:
        image = Image.open(image_path).convert("RGB")

        class_list = ", ".join(classes)
        detection_prompt = f"""Detect all objects of these classes in the image: {class_list}

For each object found, provide its bounding box coordinates as percentages (0-100) of the image size.

Return a JSON array:
[{{"class": "classname", "bbox": [left, top, right, bottom], "confidence": 0.95}}]

If no objects found, return: []"""

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

        return {
            "detections": detections,
            "tokens_used": len(outputs[0]),
            "raw_response": generated_text[:500]
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-VL2 Inference Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-vl2-tiny", help="Model name")
    args = parser.parse_args()

    # Load model at startup
    load_model(args.model)

    # Start simple HTTP server
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json

    class InferenceHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress default logging
            pass

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ready"}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/inference":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                try:
                    data = json.loads(body)
                    image_path = data["image_path"]
                    classes = data["classes"]

                    result = run_inference(image_path, classes)

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())

                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(("0.0.0.0", args.port), InferenceHandler)
    print(f"DeepSeek-VL2 server listening on port {args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
