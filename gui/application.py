from flask import Flask, Response, request, jsonify, render_template
from PIL import Image, ImageDraw
from io import BytesIO
from scipy import misc
import tensorflow as tf
import random
import numpy as np
import os
import base64
import math
import sys
sys.path.insert(0, '../training')
sys.path.insert(0, '../tf_face_mtcnn/align')
import predict as pred
import detect_face


flask_app = Flask(__name__, static_url_path='')

@flask_app.route('/', methods=['GET'])
def get_index():
    return render_template("index.html",
        results = 'false',
        error = 'false')

@flask_app.route('/', methods=['POST'])
def classify_file():
    # Detect where the face is in the picture
    print(request)
    # f = request.files['file']
    face_rects, img, rgb = aligning_faces(None)
    dr = ImageDraw.Draw(img)
    for rect in face_rects:
        print(rect)
        drawThickRects(dr, rect, 5)
    cropped = crop_faces(rgb, face_rects)
    scores = recognise_faces(cropped)

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    bufferList = []
    for idx, face in enumerate(cropped):
        tmp = Image.fromarray(face)
        buf = BytesIO()
        tmp.save(buf, format="PNG")
        bufferList.append(base64.b64encode(buf.getvalue()))

    return render_template('index.html',
        results = 'true',
        error = 'false',
        image_b64 = base64.b64encode(buffer.getvalue()),
        # scores = scores,
        faces_0 = bufferList[0],
        faces_1 = bufferList[1],
        faces_2 = bufferList[2],
        faces_3 = bufferList[3],
        faces_4 = bufferList[4],
        faces_5 = bufferList[5],
        faces_6 = bufferList[6],
        faces_7 = bufferList[7],
        faces_8 = bufferList[8],
        score_0_0=scores[0][0],
        score_0_1=scores[0][1],
        score_1_0=scores[1][0],
        score_1_1=scores[1][1],
        score_2_0=scores[2][0],
        score_2_1=scores[2][1],
        score_3_0=scores[3][0],
        score_3_1=scores[3][1],
        score_4_0=scores[4][0],
        score_4_1=scores[4][1],
        score_5_0=scores[5][0],
        score_5_1=scores[5][1],
        score_6_0=scores[6][0],
        score_6_1=scores[6][1],
        score_7_0=scores[7][0],
        score_7_1=scores[7][1],
        score_8_0=scores[8][0],
        score_8_1=scores[8][1])

def drawThickRects(dr, rect, width = 1, offset = 10):
    for w in range(width - 1):
        left = rect[0] - (1 * w) - offset
        top = rect[1] - (1 * w) - offset
        right = rect[2] + (1 * w) + offset
        bottom = rect[3] + (1 * w) + offset
        dr.rectangle((left, top, right, bottom), outline="green")

def aligning_faces(image_file):
    print(image_file)
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    random_key = np.random.randint(0, high=99999)
    
    nrof_images_total = 0
    nrof_successfully_aligned = 0

    # image_content = image_file.read()
    # im = Image.open(BytesIO(image_content))
    im = Image.open('static/unkonwn.jpg')
    rgb = np.array(im.convert('RGB'))

    bounding_boxes, _ = detect_face.detect_face(rgb, minsize, pnet, rnet, onet, threshold, factor)
    return bounding_boxes, im, rgb

def crop_faces(rgb, boxes, margin=32, image_size = 200):
    crop_images =[]
    for box in boxes:
        det1 = np.squeeze(box)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = math.ceil(box[0]-margin/2)
        bb[1] = math.ceil(box[1]-margin/2)
        bb[2] = math.ceil(box[2]+margin/2)
        bb[3] = math.ceil(box[3]+margin/2)
        cropped = rgb[bb[1]:bb[3],bb[0]:bb[2]]
        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        crop_images.append(scaled)
    return crop_images

def recognise_faces(crop_images, image_size = 200):
    model_name = 'Simple'
    model_dir = '../log/' + model_name + '/train'
    results = pred.classify_images(model_name, model_dir, crop_images, image_size =image_size, labels =['Hsia Yu-chiao', 'Sung Yun-hua'])
    print(results)
    return results[0]

def main(_):
    flask_app.run(host='127.0.0.1', port=8080)


if __name__ == '__main__':
    tf.app.run()
