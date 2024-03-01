import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time
import numpy

class CameraApp:
    def __init__(self, window, window_title):
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

        # self.btn_capture = tk.Button(window, text="Capture", width=10, command=self.capture_image)
        # self.btn_capture.place(x=(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) - 80) / 2, y=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) - 40)

        # self.btn_capture.place(x=10, y=10)
        # self.btn_capture = tk.Button(window, text="Capture", width=10, command=self.capture)
        # self.btn_capture.pack(pady=10)



        # Button to capture image
        # self.btn_capture = tk.Button(window, text="Capture", width=10, command=self.capture_image)
        # self.btn_capture.pack(anchor=tk.CENTER, expand=True)
        
        # After it is called once, the update method will be automatically called every delay milliseconds

        self.window.mainloop()

    def create_button(self):
        self.btn_capture = tk.Button(self.window, text="Capture", width=10, command=self.capture_image)
        self.btn_capture.place(x=(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) - 80) / 2, y=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) - 40)

        # self.btn_capture = tk.Button(self.window, text="Capture", width=10, command=self.capture)
        # self.btn_capture.place(x=(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) - 80) / 2, y=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) - 40)

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
            self.window.after(100, lambda: self.hide_flash())  # Remove the white image after 200 milliseconds

            # self.flash_white_image()


    def take_photo(self):
        ret, frame = self.vid.read()
        if ret:
            filename = "photobooth/" + str(round(time.time())) + '.png'
            print(filename)
            cv2.imwrite(filename, frame)
            # cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            print("Frame captured.")
            # self.is_count_down = True
            # self.btn_capture.destroy()
            # self.timer_start = time.time()
            # self.window.after(5000, self.create_button)


    def remove_flash(self, canvas):
        print('remove flash')
        canvas.delete("all")
        self.create_button()
        self.count_down_finished = True
        self.take_photo()


    def hide_flash(self):
        self.take_photo()
        self.flash = False
        self.create_button()

    def create_flash(self):
        white_image = numpy.ones((int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=numpy.uint8) * 255
        white_image = ImageTk.PhotoImage(image=Image.fromarray(white_image))
        self.canvas.create_image(0, 0, image=white_image, anchor=tk.NW)


    def flash_white_image(self):
        self.flash = True

        self.window.after(1000, lambda: self.hide_flash)  # Remove the white image after 200 milliseconds

        white_image = numpy.ones((int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=numpy.uint8) * 255
        white_image = ImageTk.PhotoImage(image=Image.fromarray(white_image))
        self.canvas.create_image(0, 0, image=white_image, anchor=tk.NW)
        print('show flash')

        # self.window.after(1000, lambda: self.remove_flash(self.canvas))  # Remove the white image after 200 milliseconds
        self.window.after(1000, lambda: self.canvas.delete("all"))  # Remove the white image after 200 milliseconds
            # self.create_button()
            # self.count_down_finished = True



    def capture_image(self):
        self.is_count_down = True
        self.btn_capture.destroy()
        self.timer_start = time.time()

        # # Capture frame by frame
        # ret, frame = self.vid.read()
        # if ret:
        #     cv2.imwrite("captured_image.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     print("Frame captured.")
        #     # self.is_count_down = True
        #     # self.btn_capture.destroy()
        #     # self.timer_start = time.time()
        #     # self.window.after(5000, self.create_button)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            # self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            # self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            if(self.flash):
                # white_image = numpy.ones((int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=numpy.uint8) * 255
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.white_image))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)



            if self.is_count_down:
              self.show_timer()
              canvas_width = self.canvas.winfo_width()
              canvas_height = self.canvas.winfo_height()
              self.canvas.create_text(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, text=self.timer_text, fill="white", font=("Arial", 72, "bold"))

              # Button to capture image
            # Calculate the center coordinates for the text
            # canvas_width = self.canvas.winfo_width()
            # canvas_height = self.canvas.winfo_height()
            # text_width = 200  # You can adjust this value as needed
            # text_height = 50   # You can adjust this value as needed
            # x_center = (canvas_width - text_width) // 2
            # y_center = (canvas_height - text_height) // 2



        # Call the update method after delay
        self.window.after(self.delay, self.update)

# Create a window and pass it to the CameraApp class
root = tk.Tk()
app = CameraApp(root, "Camera Feed")