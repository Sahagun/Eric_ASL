import tensorflow as tf
from PIL import Image
import numpy as np

'''
ref
https://www.kaggle.com/models/sayannath235/american-sign-language
'''

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "space", "delete", "nothing"]

folder = 'images'

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="asl.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for letter in LETTERS:
  path = folder + '/' + letter + '.jpg'

  # Load and preprocess image
  image_path = path
  image = Image.open(image_path)
  image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
  input_data = np.expand_dims(image, axis=0)
  input_data = (input_data.astype(np.float32) - 127.5) / 127.5  # Normalize if needed

  # Set input tensor
  interpreter.set_tensor(input_details[0]['index'], input_data)

  # Run inference
  interpreter.invoke()

  # Get output results
  output_data = interpreter.get_tensor(output_details[0]['index'])


  # Post-process output (if needed)
  # ...

  # Print or use the output data as needed
  #print(output_data)
  max_index = max(enumerate(output_data[0]),key=lambda x: x[1])[0]
  #print(max_index, output_data[0][max_index], LETTERS[max_index])
  print('expected:', letter, 'got:', LETTERS[max_index])

