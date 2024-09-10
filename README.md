
# Real-Time Traffic Sign Detection and Classification using YOLOv8

## Aim
To develop a real-time traffic sign detection and classification system using YOLOv8, enhancing road safety by accurately recognizing signs under diverse conditions for applications in autonomous vehicles and driver-assistance systems.

## Description
This project involves implementing a traffic sign detection system using YOLOv8, a state-of-the-art deep learning model known for high-speed object detection and real-time performance. The model was trained on a customized traffic signs dataset, optimizing hyperparameters such as epochs, batch sizes, learning rates, and optimizers (Adam, SGD) to achieve superior Mean Average Precision (mAP). The system aims to enhance autonomous driving and intelligent traffic systems by accurately identifying and localizing traffic signs under various conditions, contributing to improved decision-making and safety in real-world applications.

## Tools and Libraries Used
- **YOLO (Ultralytics):** Main object detection framework for training and deploying the YOLOv8 model.
- **OpenCV (cv2):** For image and video processing, including reading, displaying, and manipulating visual data.
- **Pandas (pd):** For data manipulation and handling data frames.
- **NumPy (np):** For numerical operations, array handling, and data manipulation.
- **Matplotlib (plt) and Seaborn (sns):** For data visualization and plotting graphs of training metrics.
- **PIL (Python Imaging Library):** For opening, manipulating, and saving images in various formats.
- **TQDM:** For creating progress bars during data processing and training loops.
- **IPython Display (Video):** For displaying videos within Jupyter Notebooks.

## Results
![Alt Text](https://github.com/GovindMB1/Traffic-Sign-Detection-Using-YOLOv8/blob/main/out.jpg)

## Instructions for Running the Jupyter Notebook
To run the code in the Jupyter notebook:

1. Open the notebook file in Jupyter Notebook.
2. Execute each cell in sequence by selecting the cell and pressing `Shift + Enter`.
3. Ensure you update the file paths within the notebook to reflect the correct locations for your dataset, model checkpoints, and output directories.
4. Adjust paths like `data_path`, `model_save_path`, and `output_dir` to match your file system before running the cells.
