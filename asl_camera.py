import cv2
import numpy as np
import tensorflow as tf
import time
import threading

# Load TFLite model
model_path = 'asl.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "space", "delete", "nothing"]

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Function to perform TensorFlow Lite inference
proccessing = False

def run_inference(frame):
    global proccessing
    if proccessing:
        return
    
    proccessing = True
    # Preprocess the image (resize, normalize, etc.)
    # Example: Resize the frame to match the input size expected by the model
    input_shape = input_details[0]['shape']
    image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    # image = frame.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(image, axis=0)
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5  # Normalize if needed


    # input_shape = input_details[0]['shape']
    # input_image = cv2.resize(frame, (input_shape[1], input_shape[0]))
    # input_image = np.expand_dims(input_image, axis=0)
    # # input_image = input_image.astype(np.float32) / 255.0  # Normalize if needed
    # input_image = (input_image.astype(np.float32) - 127.5) / 127.5  # Normalize if needed


    # Set the input tensor of the model
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the results (interpret the model output)
    # Example: Display the output as text
    # print(f"Prediction: {output_data[0]}")
    max_index = max(enumerate(output_data[0]),key=lambda x: x[1])[0]
    #print(max_index, output_data[0][max_index], LETTERS[max_index])
    print('Prediction:', LETTERS[max_index])

    proccessing = False




# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Set the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables for timing
start_time = time.time()
inference_interval = 1.0  # Run inference once every second


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the original frame
    cv2.imshow('Camera Feed', frame)

    # Check if it's time to run inference
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time >= inference_interval:
        # Reset the timer
        start_time = time.time()

        # Create a separate thread for TensorFlow inference
        inference_thread = threading.Thread(target=run_inference, args=(frame,))
        inference_thread.start()

    # Break the loop if the 'q' key or the 'Esc' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 is the ASCII code for 'Esc'
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
