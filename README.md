# Two-Stage YOLOv8 License Plate Recognition (ALPR)

A robust, end-to-end Automatic License Plate Recognition (ALPR) system optimized for Vietnamese license plates, utilizes a Two-Stage YOLOv8 pipeline to detect plates and extract alphanumeric characters natively.

## Key Features

* Two-Stage Architecture: Uses a primary YOLOv8 model for precise plate localization, followed by a secondary YOLOv8 model trained exclusively on alphanumeric characters for high-accuracy reading.
* Smart Spatial Sorting: Dynamically reads both 1-line (long) and 2-line (square) plates by analyzing the Y-coordinate gaps of detected bounding boxes.
* OCR-Free Pipeline: Bypasses common OCR environment issues and relies entirely on pure object detection.
* Live Webcam Integration: Includes a real-time inference script using cvzone for stylized, high-performance tracking and reading via live video feeds.

## Pipeline Architecture

1. Stage 1: Plate Detection
   - The system receives an image or video frame and runs `plate_model.pt` to draw bounding boxes around vehicles' license plates.
   - The detected plate region is cropped, padded, and dynamically upscaled if the resolution is too low.
2. Stage 2: Character Detection
   - The cropped image is passed to `yolov8_char_model.pt`, which detects individual characters (0-9, A-Z).
   - A custom mapping dictionary fixes class index offsets natively without requiring dataset relabeling.
3. Stage 3: Spatial Sorting
   - Bounding boxes are sorted top-to-bottom.
   - The algorithm calculates the average character height to group characters into lines (handling 1-line vs. 2-line plates).
   - Characters in each line are sorted left-to-right and stitched into the final license plate string.

## Installation and Setup

1. Clone the repository
```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name
```

2. Install dependencies
Ensure you have Python 3.8+ installed. Run the following command:
```bash
pip install ultralytics opencv-python numpy cvzone
```

3. Model Weights
Place your trained YOLOv8 weights into the root directory of the project:
* `plate_model.pt` (Trained on full vehicle images)
* `yolov8_char_model.pt` (Trained on 36 alphanumeric classes)

## Webcam Inference

To run the pipeline in real-time using your webcam, simply execute the `webcam.py` script:

```bash
python webcam.py
```
Note: Press `q` to quit the live video feed.

## Acknowledgements

This project was inspired by and adapted concepts from the excellent work found in the following repository:
* [trungdinh22/License-Plate-Recognition](https://github.com/trungdinh22/License-Plate-Recognition.git) - Special thanks for the foundational logic regarding bounding box manipulation and Vietnamese license plate pipeline structure.
