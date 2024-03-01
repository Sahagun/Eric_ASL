import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time
import numpy
import tensorflow as tf
import threading


class CameraApp:
    def __init__(self, window, window_title):
        # Load TFLite model
        model_path = 'asl.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "space", "delete", "nothing"]
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Function to perform TensorFlow Lite inference
        self.start_time = time.time()
        self.inference_interval = 1.0  # Run inference once every second
        self.proccessing = False

        self.window = window
        self.window.title(window_title)
        self.is_count_down = False
        self.count_down_finished = False
        self.timer_start = 0
        self.timer_text = ""
        self.flash = False

        # Open the video source (in this case, webcam)
        self.vid = cv2.VideoCapture(0)

        self.white_image = numpy.ones((int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=numpy.uint8) * 255

        
        # Create a canvas that can fit the video source
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.delay = 10
        self.update()

        self.create_button()

        self.window.mainloop()


    def create_button(self):
        self.btn_capture = tk.Button(self.window, text="Capture", width=10, command=self.capture_image)
        self.btn_capture.place(x=(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) - 80) / 2, y=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) - 40)


    def show_timer(self):
        current_time = time.time()
        if current_time - self.timer_start <= 1:
            self.timer_text = "3"
        elif current_time - self.timer_start <= 2:
            self.timer_text = "2"
        elif current_time - self.timer_start < 3:
            self.timer_text = "1"
        else:
            self.is_count_down = False
            self.timer_text = ""
            self.flash = True
            self.window.after(100, lambda: self.hide_flash()) 


    def take_photo(self):
        ret, frame = self.vid.read()
        if ret:
            filename = "photobooth/" + str(round(time.time())) + '.png'
            print(filename)
            cv2.imwrite(filename, frame)

            print("Frame captured.")


    def hide_flash(self):
        self.take_photo()
        self.flash = False
        self.create_button()


    def capture_image(self):
        self.is_count_down = True
        self.btn_capture.destroy()
        self.timer_start = time.time()


    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            if(self.flash):
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.white_image))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.run_inference(frame)


            if self.is_count_down:
              self.show_timer()
              canvas_width = self.canvas.winfo_width()
              canvas_height = self.canvas.winfo_height()
              self.canvas.create_text(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, text=self.timer_text, fill="white", font=("Arial", 72, "bold"))

        # Call the update method after delay
        self.window.after(self.delay, self.update)


    def detect_hand(self, frame):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= self.inference_interval:
            # Reset the timer
            self.start_time = time.time()

            # Create a separate thread for TensorFlow inference
            inference_thread = threading.Thread(target=self.run_inference, args=(frame,))
            inference_thread.start()

    def run_inference(self, frame):
        print('inference_thread started')
        if self.proccessing:
            return
        
        self.proccessing = True
        # Preprocess the image (resize, normalize, etc.)
        # Example: Resize the frame to match the input size expected by the model
        input_shape = self.input_details[0]['shape']
        image = cv2.resize(frame, (input_shape[1], input_shape[2]))
        # image = frame.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
        input_data = numpy.expand_dims(image, axis=0)
        input_data = (input_data.astype(numpy.float32) - 127.5) / 127.5  # Normalize if needed

        # Set the input tensor of the model
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get the output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        max_index = max(enumerate(output_data[0]),key=lambda x: x[1])[0]
        #print(max_index, output_data[0][max_index], LETTERS[max_index])
        print('Prediction:', self.LETTERS[max_index])

        self.proccessing = False

# Create a window and pass it to the CameraApp class
root = tk.Tk()
app = CameraApp(root, "Camera Feed")