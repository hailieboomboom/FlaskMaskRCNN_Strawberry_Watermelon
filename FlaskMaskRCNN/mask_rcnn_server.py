# USAGE
# Start the server:
#  python run_keras_server.py
# Submit a request via cURL:
#  curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#  python simple_request.py

# import the necessary packages
import os

from flask import render_template
from keras.preprocessing.image import img_to_array
from PIL import Image
import flask
import io
import tensorflow as tf
import mrcnn
from mrcnn import utils
import mrcnn.model as modellib
import logging
from werkzeug import secure_filename
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import visualize

import json
# from model import *
# from tensorflow.compat.v1.keras.backend import set_session
import melon2class
#from melonDetect import InferenceConfig

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/strawberrymelon20200407T1224")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/strawberrymelon20200407T1224/mask_rcnn_strawberrymelon_0045.h5")


class InferenceConfig(melon2class.FruitConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    POST_NMS_ROIS_INFERENCE = 2000

    # proved->the higher the image quality, the better the detection accuracy will be increased
    # How every, the detection speed will be slowed dramatically
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 3520 # was 1024

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.6

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 200

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    global sess

    tf.compat.v1.keras.backend.set_session(sess)

    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # print(r)


def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    scaled_image = img_to_array(image)

    # scaled_image = mold_image(scaled_image, cfg)
    image = np.expand_dims(scaled_image, 0)
    # image = np.expand_dims(scaled_image, axis=0)
    # return the processed image
    return image


@app.route('/')
def index():
    return render_template('hello.html')

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view

    class_names = ['BG', 'strawberry', 'watermelon']
    if flask.request.method == "GET":
        return 'GET ERROR'
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            #im_array = skimage.io.imread(image)

            im_array = prepare_image(image)
            # preprocess the image and prepare it for classification
            # cv2.cvtColor(im_array[0],cv2.COLOR_RGB2BGR)
            # cv2.imwrite('w.jpg',im_array[0])
            # classify the input image and then initialize the list
            # of predictions to return to the client

            global sess
            global graph
            with graph.as_default():
                tf.compat.v1.keras.backend.set_session(sess)
                yhat = model.detect(im_array, verbose=0)[0]
                # yhat = json.dumps(yhat)

                result = {
                    'rois': yhat['rois'].tolist(),
                    'class_ids': yhat['class_ids'].tolist(),
                    'scores': yhat['scores'].tolist(),
                    'masks': yhat['masks'].tolist(),

                          }
            # results = imagenet_utils.decode_predictions(preds)
            # data["predictions"] = []
            #
            # # loop over the results and add them to the list of
            # # returned predictions
            # for (imagenetID, label, prob) in results[0]:
            #     r = {"label": label, "probability": float(prob)}
            #     data["predictions"].append(r)

            # indicate that the request was a success
    # return the data dictionary as a JSON response
    #visualize.display_instances(im_array, result['rois'], result['masks'], result['class_ids'], result['scores'])

    return flask.jsonify(result)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0',port=80)
