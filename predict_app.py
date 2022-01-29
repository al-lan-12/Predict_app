import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import model_from_json
#from tensorflow.keras.models import load_model
#from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_model():
    global model
    model = load_model('png.h5')
    #model._make_predict_function()
    print(" * Model loaded!")
    
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    #image_data = re.sub('^data:image/.+;base64,', '', encoded)
    #decoded = base64.b64decode(image_data)
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    print(image)
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image).tolist()
    print(prediction)

    response = {
        'prediction': {
            'acne': prediction ,
            'eczema': prediction,
            'pigmentation': prediction
            
        }
    }
    return jsonify(response)