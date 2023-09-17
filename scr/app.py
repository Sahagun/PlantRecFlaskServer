from flask import Flask, request, jsonify

from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time

app = Flask(__name__)
# app.secret_key = os.urandom(24)


label_path = 'src/aiy_plants_V1_labelmap.csv'
model_path = 'src/lite-model_aiy_vision_classifier_plants_V1_3.tflite'

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]

  
@app.route('/')
def index():
  return 'Hello from Flask!'

@app.route('/plant', methods=["POST"])
def plant():
  file = request.files['image']
  image = Image.open(file.stream)
  classification, acc = getImageClassification(image)
  return jsonify({'plant': classification, 'accuracy': acc})


def getImageClassification(image):
  labels = load_labels(label_path)
  interpreter = Interpreter(model_path)

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  # resize image to be classified.
  image = image.convert('RGB').resize((width, height))

  # Classify the image.
  label_id, prob = classify_image(interpreter, image)
  
  # Return the classification label of the image.
  classification_label = labels[label_id].split(',')[1]

  return classification_label, np.round(prob*100, 2)

if __name__ == '__main__':
	app.run('0.0.0.0')

# run the app.
#if __name__ == "__main__":
  # Setting debug to True enables debug output. This line should be
  # removed before deploying a production app.
#  app.debug = False
#  app.run()

    # app.run(host='0.0.0.0', port=81)
