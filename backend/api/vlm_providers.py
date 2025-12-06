"""
VLM (Vision Language Model) Provider Abstraction Layer

Supports multiple VLM providers for object detection:
- Anthropic Claude (claude-sonnet-4-20250514)
- OpenAI GPT-4V (gpt-4o)
- Ollama/LLaVA (local)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import base64
import json
import re
import asyncio
from datetime import datetime


@dataclass
class BoundingBox:
    """Normalized bounding box in YOLO format (0-1)"""
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float


@dataclass
class VLMResponse:
    """Response from VLM inference"""
    bboxes: List[BoundingBox]
    raw_response: str
    tokens_used: int
    cost: float
    error: Optional[str] = None


class VLMProvider(ABC):
    """Abstract base class for VLM providers"""

    @abstractmethod
    async def detect_objects(
        self,
        image_path: Path,
        classes: List[str],
        prompt: Optional[str] = None
    ) -> VLMResponse:
        """Run object detection on an image"""
        pass

    @abstractmethod
    def get_cost_estimate(self, image_count: int) -> float:
        """Estimate cost for processing N images"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that API key/endpoint is configured and working"""
        pass

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def get_media_type(self, image_path: Path) -> str:
        """Get MIME type for image"""
        suffix = image_path.suffix.lower()
        types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return types.get(suffix, "image/jpeg")

    def _build_system_prompt(self, classes: List[str]) -> str:
        """Build system prompt for object detection"""
        return f"""You are a precise object detection system. Your task is to locate objects and output exact bounding box coordinates.

IMPORTANT INSTRUCTIONS:
1. Carefully examine the ENTIRE image
2. For each object found, estimate its TIGHT bounding box
3. Coordinates are percentages (0-100) of image dimensions:
   - left: distance from LEFT edge to LEFT side of object
   - top: distance from TOP edge to TOP of object
   - right: distance from LEFT edge to RIGHT side of object
   - bottom: distance from TOP edge to BOTTOM of object

Example: An object in the center-right of the image might have bbox [50, 30, 80, 70]

Classes to detect: {', '.join(classes)}

OUTPUT FORMAT - Return ONLY a valid JSON array:
[{{"class": "classname", "bbox": [left, top, right, bottom], "confidence": 0.95}}]

If no objects found, return exactly: []
Do NOT include any text before or after the JSON array."""

    def _build_user_prompt(self, classes: List[str]) -> str:
        """Build user prompt for object detection"""
        return f"""Locate all {', '.join(classes)} in this image.

For EACH object found:
1. Identify its exact position
2. Draw a tight box around it (not too big, not too small)
3. Express coordinates as percentages (0-100) of image size

Return JSON array: [{{"class": "name", "bbox": [left%, top%, right%, bottom%], "confidence": 0.0-1.0}}]

Be precise with the bounding box - it should tightly wrap the object."""

    def _parse_response(self, response_text: str, classes: List[str]) -> List[BoundingBox]:
        """Parse VLM response text into bounding boxes"""
        # Log raw response for debugging
        print(f"VLM raw response (first 500 chars): {response_text[:500]}")

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print(f"Found JSON in code block: {json_str[:200]}")
        else:
            # Try to find raw JSON array
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                print("No JSON array found in response")
                return []
            json_str = json_match.group()
            print(f"Found raw JSON: {json_str[:200]}")

        try:
            detections = json.loads(json_str)
            print(f"Parsed {len(detections)} detections")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return []

        bboxes = []
        classes_lower = [c.lower() for c in classes]

        for det in detections:
            if not isinstance(det, dict):
                continue

            class_name = det.get("class", "")
            # Case-insensitive class matching
            if class_name.lower() not in classes_lower:
                continue

            # Use the original class name from the allowed list
            class_idx = classes_lower.index(class_name.lower())
            class_name = classes[class_idx]

            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            try:
                confidence = float(det.get("confidence", 0.5))

                # Convert from [x_min, y_min, x_max, y_max] to YOLO format
                # Handle both 0-1 range and 0-100 range (percentage)
                raw_values = [float(b) for b in bbox]
                print(f"  Raw bbox: {raw_values}")

                # If any value > 1, assume percentages (0-100), otherwise assume 0-1 range
                if any(v > 1.0 for v in raw_values):
                    x_min, y_min, x_max, y_max = [v / 100.0 for v in raw_values]
                    print(f"  Converted from 0-100 range")
                else:
                    x_min, y_min, x_max, y_max = raw_values
                    print(f"  Using 0-1 range as-is")

                # Calculate center and dimensions
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                print(f"  YOLO format: center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")

                # Clamp values to 0-1
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                if width > 0.01 and height > 0.01:  # Minimum size threshold
                    bboxes.append(BoundingBox(
                        class_name=class_name,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        confidence=confidence
                    ))
            except (ValueError, TypeError):
                continue

        return bboxes


class AnthropicProvider(VLMProvider):
    """Anthropic Claude Vision provider"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        # Pricing as of 2024 (approximate)
        self.cost_per_input_token = 0.003 / 1000  # $3 per 1M input tokens
        self.cost_per_output_token = 0.015 / 1000  # $15 per 1M output tokens

    async def detect_objects(
        self,
        image_path: Path,
        classes: List[str],
        prompt: Optional[str] = None
    ) -> VLMResponse:
        try:
            import anthropic
        except ImportError:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error="anthropic package not installed. Run: pip install anthropic"
            )

        system_prompt = self._build_system_prompt(classes)
        user_prompt = prompt or self._build_user_prompt(classes)

        image_data = self.encode_image(image_path)
        media_type = self.get_media_type(image_path)

        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }]
            )

            response_text = response.content[0].text
            bboxes = self._parse_response(response_text, classes)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            tokens_used = input_tokens + output_tokens
            cost = (input_tokens * self.cost_per_input_token +
                   output_tokens * self.cost_per_output_token)

            return VLMResponse(
                bboxes=bboxes,
                raw_response=response_text,
                tokens_used=tokens_used,
                cost=cost
            )
        except Exception as e:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error=str(e)
            )

    def get_cost_estimate(self, image_count: int) -> float:
        # Estimate ~1500 input tokens per image (image + prompt), ~300 output tokens
        return image_count * (1500 * self.cost_per_input_token + 300 * self.cost_per_output_token)

    def validate_connection(self) -> bool:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            # Simple validation - just create a minimal message
            response = client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception:
            return False


class OpenAIProvider(VLMProvider):
    """OpenAI GPT-4 Vision provider"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        # Pricing as of 2024 (approximate for gpt-4o)
        self.cost_per_input_token = 0.0025 / 1000
        self.cost_per_output_token = 0.01 / 1000

    async def detect_objects(
        self,
        image_path: Path,
        classes: List[str],
        prompt: Optional[str] = None
    ) -> VLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error="openai package not installed. Run: pip install openai"
            )

        system_prompt = self._build_system_prompt(classes)
        user_prompt = prompt or self._build_user_prompt(classes)

        image_data = self.encode_image(image_path)
        media_type = self.get_media_type(image_path)

        try:
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )

            response_text = response.choices[0].message.content
            bboxes = self._parse_response(response_text, classes)

            tokens_used = response.usage.total_tokens
            cost = (response.usage.prompt_tokens * self.cost_per_input_token +
                   response.usage.completion_tokens * self.cost_per_output_token)

            return VLMResponse(
                bboxes=bboxes,
                raw_response=response_text,
                tokens_used=tokens_used,
                cost=cost
            )
        except Exception as e:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error=str(e)
            )

    def get_cost_estimate(self, image_count: int) -> float:
        return image_count * (1500 * self.cost_per_input_token + 300 * self.cost_per_output_token)

    def validate_connection(self) -> bool:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            client.models.list()
            return True
        except Exception:
            return False


class OllamaProvider(VLMProvider):
    """Ollama local VLM provider (LLaVA, etc.)"""

    def __init__(self, endpoint: str = None, model: str = "llava:13b"):
        import os
        if endpoint is None:
            endpoint = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    async def detect_objects(
        self,
        image_path: Path,
        classes: List[str],
        prompt: Optional[str] = None
    ) -> VLMResponse:
        try:
            import httpx
        except ImportError:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error="httpx package not installed. Run: pip install httpx"
            )

        system_prompt = self._build_system_prompt(classes)
        user_prompt = prompt or self._build_user_prompt(classes)

        image_data = self.encode_image(image_path)

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                # Use /api/chat for vision models (newer Ollama API)
                response = await client.post(
                    f"{self.endpoint}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                                "images": [image_data]
                            }
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temp for consistent output
                            "num_predict": 2048
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()

            # Extract response from chat format
            response_text = data.get("message", {}).get("content", "")
            bboxes = self._parse_response(response_text, classes)

            return VLMResponse(
                bboxes=bboxes,
                raw_response=response_text,
                tokens_used=data.get("eval_count", 0),
                cost=0.0  # Local = free
            )
        except Exception as e:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error=str(e)
            )

    def get_cost_estimate(self, image_count: int) -> float:
        return 0.0  # Local inference is free

    def validate_connection(self) -> bool:
        try:
            import httpx
            response = httpx.get(f"{self.endpoint}/api/tags", timeout=10.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_available_models(self) -> List[str]:
        """List all installed models in Ollama (for auto-labeling, user selects the model)"""
        try:
            import httpx
            response = httpx.get(f"{self.endpoint}/api/tags", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                # Return all installed models - user can select which to use
                return [model.get("name", "") for model in data.get("models", []) if model.get("name")]
        except Exception:
            pass
        return []


class Florence2Provider(VLMProvider):
    """Florence-2 local VLM provider - runs via venv subprocess"""

    VENV_PATH = Path("storage/venvs/florence2")
    INFERENCE_SCRIPT = Path("api/vlm_inference/florence2_infer.py")

    def __init__(self, model: str = "microsoft/Florence-2-base", device: str = "cuda"):
        self.model = model
        self.device = device  # "cuda" or "cpu"

    def _get_python_bin(self) -> Path:
        """Get Python binary from venv"""
        return self.VENV_PATH / "bin" / "python"

    async def detect_objects(
        self,
        image_path: Path,
        classes: List[str],
        prompt: Optional[str] = None
    ) -> VLMResponse:
        import subprocess

        if not self.is_available():
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error="Florence-2 venv not installed. Install it from the Venvs page."
            )

        python_bin = self._get_python_bin()

        # Prepare input
        input_data = json.dumps({
            "image_path": str(image_path),
            "classes": classes,
            "model": self.model,
            "device": self.device
        })

        try:
            # Run inference via subprocess
            result = subprocess.run(
                [str(python_bin), str(self.INFERENCE_SCRIPT)],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )

            if result.returncode != 0:
                return VLMResponse(
                    bboxes=[],
                    raw_response="",
                    tokens_used=0,
                    cost=0,
                    error=f"Inference failed: {result.stderr or result.stdout}"
                )

            # Parse JSON output - handle empty or non-JSON responses
            stdout = result.stdout.strip()
            if not stdout:
                return VLMResponse(
                    bboxes=[],
                    raw_response="",
                    tokens_used=0,
                    cost=0,
                    error=f"Empty response from inference. stderr: {result.stderr}"
                )

            try:
                output = json.loads(stdout)
            except json.JSONDecodeError as e:
                return VLMResponse(
                    bboxes=[],
                    raw_response="",
                    tokens_used=0,
                    cost=0,
                    error=f"Invalid JSON from inference: {e}. stdout: {stdout[:500]}, stderr: {result.stderr[:500] if result.stderr else 'none'}"
                )

            if "error" in output:
                return VLMResponse(
                    bboxes=[],
                    raw_response="",
                    tokens_used=0,
                    cost=0,
                    error=output["error"]
                )

            # Convert detections to BoundingBox objects
            bboxes = [
                BoundingBox(
                    class_name=d["class_name"],
                    x_center=d["x_center"],
                    y_center=d["y_center"],
                    width=d["width"],
                    height=d["height"],
                    confidence=d["confidence"]
                )
                for d in output.get("detections", [])
            ]

            return VLMResponse(
                bboxes=bboxes,
                raw_response=output.get("raw_response", ""),
                tokens_used=output.get("tokens_used", 0),
                cost=0.0
            )

        except subprocess.TimeoutExpired:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error="Inference timed out (>5 min)"
            )
        except Exception as e:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error=str(e)
            )

    def get_cost_estimate(self, image_count: int) -> float:
        return 0.0  # Local inference is free

    def validate_connection(self) -> bool:
        """Check if venv is installed"""
        return self.is_available()

    @staticmethod
    def is_available() -> bool:
        """Check if Florence-2 venv is installed"""
        venv_python = Florence2Provider.VENV_PATH / "bin" / "python"
        return venv_python.exists()

    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        """List available Florence-2 models with detailed information"""
        return [
            {
                "name": "microsoft/Florence-2-base",
                "display_name": "Florence-2 Base",
                "params": "232M",
                "size_gb": 0.5,
                "vram_gb": 2,
                "description": "Fast and lightweight. Good for quick labeling tasks.",
                "recommended": True,
                "category": "efficient"
            },
            {
                "name": "microsoft/Florence-2-large",
                "display_name": "Florence-2 Large",
                "params": "771M",
                "size_gb": 1.5,
                "vram_gb": 4,
                "description": "Higher accuracy for complex scenes. Best detection quality.",
                "recommended": False,
                "category": "accurate"
            },
            {
                "name": "microsoft/Florence-2-base-ft",
                "display_name": "Florence-2 Base Fine-tuned",
                "params": "232M",
                "size_gb": 0.5,
                "vram_gb": 2,
                "description": "Fine-tuned on additional data. Better generalization.",
                "recommended": False,
                "category": "efficient"
            },
            {
                "name": "microsoft/Florence-2-large-ft",
                "display_name": "Florence-2 Large Fine-tuned",
                "params": "771M",
                "size_gb": 1.5,
                "vram_gb": 4,
                "description": "Best overall accuracy. Fine-tuned large model.",
                "recommended": False,
                "category": "accurate"
            },
        ]


class NVIDIANIMProvider(VLMProvider):
    """NVIDIA NIM API provider for vision models"""

    def __init__(self, api_key: str, model: str = "microsoft/phi-3.5-vision-instruct"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://integrate.api.nvidia.com/v1"
        # NVIDIA NIM pricing (approximate, varies by model)
        self.cost_per_input_token = 0.002 / 1000  # ~$2 per 1M input tokens
        self.cost_per_output_token = 0.006 / 1000  # ~$6 per 1M output tokens

    async def detect_objects(
        self,
        image_path: Path,
        classes: List[str],
        prompt: Optional[str] = None
    ) -> VLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error="openai package not installed. Run: pip install openai"
            )

        system_prompt = self._build_system_prompt(classes)
        user_prompt = prompt or self._build_user_prompt(classes)

        image_data = self.encode_image(image_path)
        media_type = self.get_media_type(image_path)

        try:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

            response = client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )

            response_text = response.choices[0].message.content
            bboxes = self._parse_response(response_text, classes)

            tokens_used = response.usage.total_tokens if response.usage else 0
            cost = 0.0
            if response.usage:
                cost = (response.usage.prompt_tokens * self.cost_per_input_token +
                       response.usage.completion_tokens * self.cost_per_output_token)

            return VLMResponse(
                bboxes=bboxes,
                raw_response=response_text,
                tokens_used=tokens_used,
                cost=cost
            )
        except Exception as e:
            return VLMResponse(
                bboxes=[],
                raw_response="",
                tokens_used=0,
                cost=0,
                error=str(e)
            )

    def get_cost_estimate(self, image_count: int) -> float:
        return image_count * (1500 * self.cost_per_input_token + 300 * self.cost_per_output_token)

    def validate_connection(self) -> bool:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            # Test with a minimal request
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            return True
        except Exception:
            return False


def get_vlm_provider(provider_type: str, settings: Dict[str, Any], model_override: Optional[str] = None) -> VLMProvider:
    """
    Factory function to get appropriate VLM provider.

    Args:
        provider_type: One of "anthropic", "openai", "ollama", "nvidia"
        settings: Dict containing API keys and endpoints
        model_override: Optional specific model to use (overrides settings)

    Returns:
        Configured VLMProvider instance
    """
    if provider_type == "anthropic" or provider_type == "claude":
        api_key = settings.get("vlm_anthropic_api_key")
        if not api_key:
            raise ValueError("Anthropic API key not configured. Set it in Settings.")
        model = model_override or settings.get("vlm_anthropic_model", "claude-sonnet-4-20250514")
        return AnthropicProvider(api_key=api_key, model=model)

    elif provider_type == "openai" or provider_type == "gpt4v":
        api_key = settings.get("vlm_openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key not configured. Set it in Settings.")
        model = model_override or settings.get("vlm_openai_model", "gpt-4o")
        return OpenAIProvider(api_key=api_key, model=model)

    elif provider_type == "nvidia" or provider_type == "nim":
        api_key = settings.get("vlm_nvidia_api_key")
        if not api_key:
            raise ValueError("NVIDIA API key not configured. Get one at https://build.nvidia.com/")
        model = model_override or settings.get("vlm_nvidia_model", "microsoft/phi-3.5-vision-instruct")
        return NVIDIANIMProvider(api_key=api_key, model=model)

    elif provider_type == "ollama":
        import os
        default_endpoint = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
        endpoint = settings.get("vlm_ollama_endpoint", default_endpoint)
        model = model_override or settings.get("vlm_ollama_model", "llava:13b")
        return OllamaProvider(endpoint=endpoint, model=model)

    elif provider_type == "florence2" or provider_type == "florence-2":
        model = model_override or settings.get("vlm_florence2_model", "microsoft/Florence-2-base")
        device = settings.get("device", "cuda")
        return Florence2Provider(model=model, device=device)

    else:
        raise ValueError(f"Unknown VLM provider: {provider_type}")


async def detect_with_retry(
    provider: VLMProvider,
    image_path: Path,
    classes: List[str],
    prompt: Optional[str] = None,
    max_retries: int = 3
) -> VLMResponse:
    """Detect objects with exponential backoff retry for rate limits"""
    last_error = None

    for attempt in range(max_retries):
        response = await provider.detect_objects(image_path, classes, prompt)

        if not response.error:
            return response

        last_error = response.error

        # Check for rate limit errors
        if "rate" in response.error.lower() or "429" in response.error:
            wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
            await asyncio.sleep(wait_time)
        else:
            # Non-rate-limit error, don't retry
            break

    return VLMResponse(
        bboxes=[],
        raw_response="",
        tokens_used=0,
        cost=0,
        error=last_error
    )
