import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from PIL import ImageFont, ImageDraw, Image

show_count_down = False
show_flash = False


# Load TFLite model
model_path = 'asl.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

fontpath = 'RubikMonoOne-Regular.ttf'
honk_font = ImageFont.truetype(fontpath, 72)


predicted_letter = ''

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "space", "delete", "nothing"]

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Function to perform TensorFlow Lite inference
proccessing = False

def add_count_down_text(frame, text):
    global honk_font

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    color = (255, 255, 255)  # White color in BGR
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position of the text at the center of the image
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    # Add the text to the image
    # cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Calculate the position of the text at the center of the image
    text_x = (frame.shape[1]) // 2
    text_y = (frame.shape[0]) // 2


    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    # Get the size of the text
    # text_width, text_height = draw.textsize(text, font=honk_font)
    # print(draw.textbbox((0, 0), text, font=honk_font))
    # text_width, text_height = draw.textbbox((0, 0), text, font=honk_font)

    # Get the bounding box of the text
    bbox = draw.textbbox((0, 0), text, font=honk_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate the position to center the text
    x = (frame.shape[1] - text_width) / 2
    y = (frame.shape[0] - text_height) / 2

    draw.text((x, y),  text, font = honk_font, fill = (255, 255, 255, 0))
    frame = np.array(img_pil)
    return frame




    

def run_inference(frame):
    global proccessing, predicted_letter
    global show_count_down, start_time_picture
    global show_flash

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
    predicted_letter = LETTERS[max_index]

    proccessing = False

    if (predicted_letter == "t" or predicted_letter == "a" or predicted_letter == "y") and show_count_down == False and show_flash == False:
        print("Take Photo in ref")
        show_count_down = True
        start_time_picture = time.time()
        predicted_letter == ""





# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Set the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

white_image = np.ones((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8) * 255


# Initialize variables for timing
start_time = time.time()
inference_interval = 1.0  # Run inference once every second


start_time_picture = time.time()
start_time_flash = time.time()

def show_flash_func(frame):
    global start_time_picture, show_count_down, start_time_flash, show_flash
    predicted_letter == ""
    current_time = time.time()
    elapsed_time = current_time - start_time_flash
    if elapsed_time <= 0.2:
        return True
    
    show_flash = False
    filename = "photobooth/" + str(round(time.time())) + '.png'
    print(filename)
    cv2.imwrite(filename, frame)

    return False


def count_down(frame):
    global start_time_picture, show_count_down, start_time_flash, show_flash
    predicted_letter == ""
    current_time = time.time()
    elapsed_time = current_time - start_time_picture

    current_time = time.time()
    if elapsed_time <= 1:
        return add_count_down_text(frame, "3")
    elif elapsed_time <= 2:
        return add_count_down_text(frame, "2")
    elif elapsed_time < 3:
        return add_count_down_text(frame, "1")      
    else:
        print('show_flash set to true')
        show_count_down = False  
        start_time_flash = time.time()
        show_flash = True
        return frame


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if show_count_down:
        frame = count_down(frame)
        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ASCII code for 'Esc'
          break

        continue

    if show_flash:
        show_flash_func(frame)
        cv2.imshow('Camera Feed', white_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ASCII code for 'Esc'
          break

        continue

    # Display the original frame
    cv2.imshow('Camera Feed', frame)


    # Check if it's time to run inference
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time >= inference_interval and show_count_down == False and show_flash == False:
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
