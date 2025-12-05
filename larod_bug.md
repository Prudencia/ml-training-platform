# Larod DLPU Model Loading Bug Report

## Summary

Custom YOLOv5 TFLite models fail to load on ARTPEC-8 DLPU with "Asynchronous connection has been closed" error, while the same model works correctly on CPU backend.

## Environment

- **Camera Model:** AXIS M3215-LVE
- **Firmware Version:** 12.6.104
- **SoC:** ARTPEC-8
- **ACAP SDK Version:** 12.5.0

## Issue Description

When attempting to load a custom-trained YOLOv5s TFLite model using `axis-a8-dlpu-tflite` backend, the larod service crashes with memory corruption. The same model works perfectly when using `cpu-tflite` backend.

## Error Messages

From camera system log:

```
Model_Setup: Unable to load model: Could not load model: Asynchronous connection has been closed
Model setup failed
free(): double free detected in tcache 2
```

## Model Specifications

The model is trained specifically for ARTPEC-8 DLPU compatibility:

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv5s with Axis patches |
| Input Size | 640x640 |
| Quantization | INT8 per-tensor (not per-channel) |
| Input/Output Type | UINT8 |
| First Conv Layer | 5x5 kernel (DLPU-compatible) |
| Activation | ReLU6 (DLPU-compatible) |
| Model Size | 6.79 MB |
| FLEX Ops | None |

### model.json Configuration

```json
{
  "modelWidth": 640,
  "modelHeight": 640,
  "quant": 0.003898685798048973,
  "zeroPoint": 0,
  "boxes": 25200,
  "classes": 2,
  "objectness": 0.4,
  "confidence": 0.75,
  "nms": 0.05,
  "path": "model/model.tflite",
  "scaleMode": 0,
  "videoWidth": 1280,
  "videoHeight": 720,
  "videoAspect": "16:9",
  "chip": "axis-a8-dlpu-tflite",
  "labels": ["dog", "person"],
  "description": ""
}
```

## Steps to Reproduce

1. Train YOLOv5s model with Axis-compatible architecture:
   - ReLU6 activation function
   - 5x5 first convolution layer

2. Export to TFLite with INT8 per-tensor quantization

3. Build DetectX ACAP with platform set to "ARTPEC-8 DLPU"

4. Install ACAP on AXIS M3215-LVE (firmware 12.6.104)

5. Start the ACAP

**Result:** Model fails to load with "Asynchronous connection has been closed" error

## Workaround Verification

When the same model is deployed with `cpu-tflite` backend instead of `axis-a8-dlpu-tflite`:

```
Video 1280x720 started
Entering main loop
```

The model loads and runs correctly on CPU, confirming the model itself is valid.

## Related Issues

This issue appears to be related to:
- GitHub Discussion: https://github.com/AxisCommunications/acap-computer-vision-sdk-examples/discussions/168

That discussion reports:
- Same "Asynchronous connection has been closed" error
- Same "double free or corruption" memory error
- Issue appeared in firmware 11.x (worked in firmware 10.x)
- Affects custom TFLite models on DLPU

## Model Verification

The TFLite model has been verified to be DLPU-compatible:

```
Total tensors: 312
Non-INT8/INT32 tensors: 0 (all tensors are properly quantized)
FLEX ops: None detected
Input dtype: uint8
Output dtype: uint8
```

## Expected Behavior

The model should load successfully on `axis-a8-dlpu-tflite` backend and run inference on the DLPU hardware accelerator.

## Actual Behavior

The larod service crashes with memory corruption ("double free detected in tcache 2") when attempting to load the model, resulting in "Asynchronous connection has been closed" error.

## Request

1. Investigation into why valid INT8 TFLite models cause memory corruption in larod on firmware 12.6.104

2. Potential firmware fix or workaround for DLPU model loading

3. Confirmation if this is a known issue with specific firmware versions

## Attachments

Available upon request:
- model.tflite file
- Complete ACAP package (.eap)
- Full system logs
- Model training configuration

## Contact

[Your contact information]

---

*Report generated: 2025-12-05*
