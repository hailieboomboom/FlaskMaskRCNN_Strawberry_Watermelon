'''
The detection code is partially derived and modified from mask-rcnn_server.py
in Flask Web Server Implementation of Object Detection Mask-RCNN Mode published
by The JaeLal

The source of the code is https://github.com/TheJaeLal/ObjectDetectionRestAPI .

'''


# USAGE
# Start the server:
#  python run_keras_server.py
# Submit a request via cURL:
#  curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#  python simple_request.py

# import the necessary packages
import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from keras.preprocessing.image import img_to_array
from PIL import Image
import base64
from io import BytesIO
import skimage.io

import flask
import io
import tensorflow as tf
#import mrcnn
from mrcnn import utils
import mrcnn.model as modellib
import logging
from werkzeug.utils import secure_filename
import numpy as np

import utils2

import json

import melon2class


sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()


model = None

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/strawberrymelon20200407T1224")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/strawberrymelon20200407T1224/mask_rcnn_strawberrymelon_0015.h5")


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

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_names = ['BG', 'strawberry', 'watermelon']

@app.route('/')
def index():
    return render_template('hello.html')



def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

@app.route('/upload', methods = ['POST'])
def upload():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        print("enter!!!!!!!!!!!!!!")
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        #img_string = request.form['image']
        img_name = secure_filename(img.filename)
        print(img_name)
        #print(img_string)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        print("create upload dir success")
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        image2 = skimage.io.imread(saved_path)
        print("image read successfully")
        # input_array = np.array(image)
        # input_array = np.expand_dims(input_array, axis=0)

        global sess
        global graph
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)
            yhat = model.detect([image2], verbose=0)[0]
            result = {
                'rois': yhat['rois'].tolist(),
                'class_ids': yhat['class_ids'].tolist(),
                'scores': yhat['scores'].tolist(),
                'masks': yhat['masks'].tolist()
            }
            count_num = len(yhat['scores'].tolist())
            utils2.display_instances(image2, yhat['rois'], yhat['masks'], yhat['class_ids'], class_names, yhat['scores'])



        return render_template('complete.html', count_num = count_num)
    else:
        return "Where is the image?"

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response



# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0',port=80)

