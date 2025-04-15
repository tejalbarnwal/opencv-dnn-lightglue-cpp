# opencv-dnn-lightglue-cpp

This repository explores integrating keypoint detection and matching models—specifically [ALIKED](https://github.com/wvangansbeke/ALIKED) and [LightGlue](https://github.com/cvg/LightGlue)—with OpenCV's DNN module using C++.

The goal is to support real-time feature extraction and matching with ONNX models entirely within OpenCV DNN.

---

## Repository Structure
```bash
opencv-dnn-lightglue-cpp/
├── tutorial-1/                           # Introduction to ONNX inference using OpenCV DNN
│   ├── assets/
│   │   ├── coco.names                    # Class labels for YOLOv5
│   │   ├── sample.jpg                    # Sample image for object detection
│   │   └── YOLOv5s_torchV200.onnx        # ONNX version of YOLOv5s model
│   ├── build/                            # CMake build directory
│   ├── src/
│   │   └── try-yolov5.cpp                # Source file to run YOLOv5 with OpenCV DNN
│   └── CMakeLists.txt                    # CMake build config for tutorial-1

├── tutorial-2/                           # Attempt to implement feature matching with SuperPoint + LightGlue
│   ├── assets/
│   │   ├── DSC_0410.JPG                  # Input image 1
│   │   └── DSC_0411.JPG                  # Input image 2
│   ├── build/                            # CMake build directory
│   ├── onnx-models/                      # Various ONNX models for feature extraction + matching
│   │   ├── superpoint_2048_lightglue_end2end.onnx
│   │   ├── superpoint_lightglue_pipeline_staticinput.onnx
│   │   ├── superpoint_lightglue_pipeline.onnx
│   │   └── superpoint_lightglue_pipeline.trt.onnx
│   ├── src/
│   │   └── try-superpoint-lightglue.cpp  # Code attempting to run SuperPoint + LightGlue
│   └── CMakeLists.txt                    # CMake build config for tutorial-2

└── README.md                             # Main readme (project overview, progress, issues)
```

---

## Status

✅ YOLOv5 inference using OpenCV DNN  
✅ Initial integration of SuperPoint + LightGlue models  
⚠️ Investigation and support for ALIKED and LightGlue (independently)

---

## Issues Encountered

While working with various ONNX models, the following roadblocks were encountered with OpenCV DNN:

1. **Dynamic input size not supported**  
   - OpenCV DNN backend currently fails when loading models that expect dynamic input shapes (like some LightGlue variants).
   - Workaround: Use static input-sized ONNX models or preprocess inputs to expected shape for prototyping purposes

2. **Unsupported operations in SuperPoint-LightGlue pipeline**  
   - Models combining keypoint detection and matching often contain custom or newer ONNX ops not supported by OpenCV.
   - Errors like `Unknown layer type` or `Unsupported operation` are thrown.
   - Unsupported Operations in OpenCV DNN
        Based on reading OpenCV DNN source code and analyzing the operations used in the `superpoint-lightglue` ONNX models, the following operations are currently **not supported** by OpenCV DNN:

        - `ScatterND`: Used for dynamic tensor updates at specific indices.
        - `Erf`: Commonly used in activation functions like GELU.
        - `GridSample`: Typically used for warping or sampling input tensors, such as in spatial transformers.
        - `NonZero`: Finds indices of non-zero elements in a tensor.
        - `TopK`: Selects the top K elements; useful in attention or keypoint selection.
        - `Not`: Logical negation, often used in control flow or masking.
        - `Or`: Logical OR, used in conditional logic.
        - `And`: Logical AND, used for combining boolean conditions.

        These unsupported layers lead to errors. 


3. 🔍 Identifying the operations required to run ALIKED and standalone LightGlue and if they are supported by OpenCV

---

## References

- [OpenCV DNN Docs](https://docs.opencv.org/master/d6/d0f/group__dnn.html)
- [LightGlue GitHub](https://github.com/cvg/LightGlue)
- [ALIKE GitHub](https://github.com/wvangansbeke/ALIKED)
- [SuperPoint GitHub](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [LightGlue ONNX models](https://github.com/fabio-sim/LightGlue-ONNX)


