import argparse
import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


### Define function for inference with TFLite model and displaying results
def tflite_detect_image(modelpath, imgpath, lblpath, min_conf):
    # Load the label map into memory
    with open(lblpath, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    float_input = input_details[0]["dtype"] == np.float32

    input_mean = 127.5
    input_std = 127.5

    # Load and preprocess the image
    image = cv2.imread(imgpath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]["index"])[
        0
    ]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]["index"])[
        0
    ]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]["index"])[
        0
    ]  # Confidence of detected objects

    # Loop over all detections and draw detection boxes if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > min_conf) and (scores[i] <= 1.0):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Save the output image with bounding boxes
    output_image_path = os.path.splitext(imgpath)[0] + "_out.png"
    cv2.imwrite(output_image_path, image)
    print("Output image saved at:", output_image_path)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to the input image")
    args = parser.parse_args()

    # Define paths and parameters
    MODEL_PATH = "detect.tflite"
    LABELS_PATH = "label_map.pbtxt"
    MIN_CONF_THRESHOLD = 0.2

    # Perform object detection
    if args.source:
        tflite_detect_image(MODEL_PATH, args.source, LABELS_PATH, MIN_CONF_THRESHOLD)
    else:
        print("Please provide the path to the input image using the --source argument.")
