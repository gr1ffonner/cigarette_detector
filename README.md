# Cigarette Detector

![Cigarette Detector](https://github.com/gr1ffonner/cigarette_detector/assets/112549266/f9745bb6-193d-4843-9660-ad9f1d2b7cf5)

## Description

Cigarette Detector is a deep learning project designed to detect and draw bounding boxes around cigarettes in images and videos. The project utilizes the TensorFlow and OpenCV libraries for the computer vision tasks and has been trained using Google Colab hardware. Roboflow is used for dataset preparation and augmentation.

## Tech Stack

The project is built on the following technologies:

- TensorFlow: An open-source deep learning framework used for creating, training, and deploying machine learning models.
- OpenCV: A popular computer vision library used for image and video processing.
- Roboflow: A platform for managing computer vision datasets, preprocessing, and data augmentation.

## Installation

To use the Cigarette Detector, follow these installation steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/gr1ffonner/cigarette_detector.git
   cd cigarette_detector
2. Activate venv
   
   ```bash
   python -m venv venv
   source venv/bin/activate
3. Install dependencies
   
   ```bash
   pip install -r reqs.txt

## Usage

There are three main scripts provided to use the Cigarette Detector:

1. Detect on Images:
  To detect cigarettes in a single image and generate an output image with bounding boxes, use the following command:
  
    ```bash
    python detect_on_images.py --source your_image.jpg
    ```
    The result will be saved as your_image_out.jpg with bounding boxes drawn around the detected cigarettes.
   
2. Detect on Images Directory:
  If you want to detect cigarettes in multiple images within a directory and get output images with bounding boxes, use the following command:
  
    ```bash
    python detect_on_images_dir.py --source path/to/your/images_dir/
    ```
    The result images will be saved in a directory named images_dir_out with bounding boxes drawn around the detected cigarettes.
   
3. Detect on Videos:
  To detect cigarettes on a realtime video with bounding boxes, use the following command:
  
    ```bash
    python detect_on_videos.py --source your_video.mp4
    ```
    The result will be shown as video with bounding boxes.
   
   https://github.com/gr1ffonner/cigarette_detector/assets/112549266/b4c7d126-6ff3-42fb-9ec8-b782ffb271c0




    




  
