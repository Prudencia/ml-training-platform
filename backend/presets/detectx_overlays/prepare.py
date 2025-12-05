import json
import os
import tensorflow as tf

def get_chip_name(platform):
    platform_mapping = {
        "A8": "axis-a8-dlpu-tflite",
        "A9": "a9-dlpu-tflite",
        "TPU": "google-edge-tpu-tflite"
    }
    return platform_mapping.get(platform.upper(), "axis-a8-dlpu-tflite")

def get_video_dimensions(image_size):
    # Use 16:9 resolutions that are widely supported by modern Axis cameras
    # The video capture resolution should be >= model input size
    # Common 16:9 resolutions: 640x360, 1280x720 (720p), 1920x1080 (1080p)
    video_mapping = {
        480: (640, 480),      # 4:3 - legacy, may not work on all cameras
        640: (1280, 720),     # 720p - widely supported 16:9
        768: (1280, 720),     # 720p
        960: (1920, 1080),    # 1080p
        1440: (1920, 1080),   # 1080p
    }

    # Default to 720p if size not in mapping
    video_width, video_height = video_mapping.get(image_size, (1280, 720))

    return video_width, video_height

def parse_labels_file(file_path):
    try:
        with open(file_path, 'r') as file:
            labels = [line.strip() for line in file if line.strip()]
        return labels
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default labels.")
        return ["label1", "label2"]

def generate_json(platform="A8", image_size=480, objectness=0.25, nms=0.05, confidence=0.30):
    # Get video dimensions
    video_width, video_height = get_video_dimensions(image_size)

    # Determine aspect ratio based on video dimensions
    if video_width == 1920 and video_height == 1080:
        video_aspect = "16:9"
    elif video_width == 1280 and video_height == 720:
        video_aspect = "16:9"
    elif video_width == 640 and video_height == 480:
        video_aspect = "4:3"
    else:
        video_aspect = "16:9"  # Default to 16:9 for modern cameras

    # Default values
    data = {
        "modelWidth": image_size,
        "modelHeight": image_size,
        "quant": 0,
        "zeroPoint": 0,
        "boxes": 0,
        "classes": 0,
        "objectness": objectness,
        "confidence": confidence,
        "nms": nms,
        "path": "model/model.tflite",
        "scaleMode": 0,
        "videoWidth": video_width,
        "videoHeight": video_height,
        "videoAspect": video_aspect,
        "chip": get_chip_name(platform),
        "labels": ["label1", "label2"],
        "description": ""
    }

    # Rest of the function remains the same
    labels_path = "./app/model/labels.txt"
    data["labels"] = parse_labels_file(labels_path)

    model_path = "./app/model/model.tflite"
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        data["modelWidth"] = image_size
        data["modelHeight"] = image_size

        output_details = interpreter.get_output_details()
        
        scale, zero_point = output_details[0]['quantization']
        box_number = output_details[0]['shape'][1]
        class_number = output_details[0]['shape'][2] - 5

        data["quant"] = float(scale)
        data["zeroPoint"] = int(zero_point)
        data["boxes"] = int(box_number)
        data["classes"] = int(class_number)

    except Exception as e:
        print(f"Warning: Error processing TFLite model: {e}")

    if len(data["labels"]) != data["classes"]:
        print(f"Warning: Number of labels ({len(data['labels'])}) does not match number of classes ({data['classes']}).")

    os.makedirs('./app/model', exist_ok=True)

    file_path = './app/model/model.json'
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

    print(f"JSON file has been generated and saved to {file_path}")
    print("\nGenerated JSON:")
    print(json.dumps(data, indent=2))

# Rest of the code remains the same
def generate_settings_json():
    # ... (keep the existing generate_settings_json function unchanged)
    pass

if __name__ == "__main__":
    import sys

    # Support both command-line args and interactive mode
    if len(sys.argv) >= 3:
        # Command-line mode: python prepare.py <platform> <image_size> [objectness] [nms] [confidence]
        platform = sys.argv[1]
        image_size = int(sys.argv[2])
        objectness = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
        nms = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
        confidence = float(sys.argv[5]) if len(sys.argv) > 5 else 0.30
    else:
        # Interactive mode (backward compatible)
        platform = input("Enter platform (A8/A9/TPU): ")
        image_size = int(input("Enter image size (480/640/768/960): "))
        objectness = 0.25  # Default for interactive mode
        nms = 0.05  # Default for interactive mode
        confidence = 0.30  # Default for interactive mode

    generate_json(platform, image_size, objectness, nms, confidence)

    labels_path = "./app/model/labels.txt"
    labels = parse_labels_file(labels_path)

    generate_settings_json()
    