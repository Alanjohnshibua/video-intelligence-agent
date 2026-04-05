# Models Directory

This folder is intended for model assets and configuration that should be separated from source code.

Typical contents:

- YOLO checkpoints such as `yolov8n.pt`
- face-recognition model caches
- quantized or CPU-optimized weights
- future ONNX or TensorRT exports

For lightweight local development, the project currently resolves `yolov8n.pt` from the repo root when available.
