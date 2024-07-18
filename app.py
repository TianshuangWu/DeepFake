from keras import models
from keras.applications.xception import preprocess_input
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
import tempfile
import json
import pprint
import cv2
import dlib
import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import os
import io
import sys
import numpy as np
import base64
import werkzeug
import time

class MtcnnDetector:
    pass

face_detector = dlib.get_frontal_face_detector()


app = Flask(__name__)


app.logger.setLevel('INFO')

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')

model_Xc = models.load_model('./model/model_finetuned_xception.hdf5')


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    # Reference: https://github.com/ondyari/FaceForensics
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()  # Taking lines numbers around face
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)  # scaling size of box to 1.3
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def get_predicition(image):
    """Expects the image input, this image further cropped to face
    and the cropped face image will be sent for evalution funtion
    finally
    returns the annotated reusult with bounding box around the face.
    """
    height, width = image.shape[:2]
    try:  # If in case face is not detected at any frame
        face = face_detector(image, 1)[0]  # Face detection
        # Calling to get bound box around the face
        x, y, size = get_boundingbox(face=face, width=width, height=height)
    except IndexError:
        pass
    cropped_face = image[y:y+size, x:x+size]  # cropping the face
    # Sending the cropped face to get classifier result
    output, label = evaluate(cropped_face)
    font_face = cv2.FONT_HERSHEY_SIMPLEX  # font settings
    thickness = 2
    font_scale = 1
    if label == 'Real':
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    x = face.left()    # Setting the bounding box on uncropped image
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y
    cv2.putText(image, label+'_'+str('%.2f' % output)+'%', (x, y+h+30),
                font_face, font_scale,
                color, thickness, 2)  # Putting the label and confidence values

    # draw box over face
    return cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


def evaluate(cropped_face):
    """This function classifies the cropped  face on loading the trained model
    and
    returns the label and confidence value
    """
    img = cv2.resize(cropped_face, (299, 299))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    res = model_Xc.predict(img)[0]
    if np.argmax(res) == 1:
        label = 'Fake'
    else:
        label = 'Real'
    return res[np.argmax(res)]*100.0, label


def test_for_video(video_path):
    dirname = os.path.dirname(__file__)
    print(dirname)
    file_name = video_path.split("\\")[-1]
    print(file_name)
    output_ = os.path.join(dirname, "test_video_results", file_name)
    print(output_)

    capture = cv2.VideoCapture(video_path)
    # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    limit_frame = duration * 10
    print('duration (S) = ' + str(limit_frame))

    if capture.isOpened():
        _, image = capture.read()
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        out = cv2.VideoWriter(output_+'_output.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    else:
        _ = False
    i = 1
    while (_):
        _, image = capture.read()
        classified_img = get_predicition(image)
        out.write(classified_img)
        if i % 10 == 0:
            print("Number of frames complted:{}".format(i))
        if i > limit_frame:
            break
        i = i+1
    capture.release()
    return output_+'_output.avi'


def test_for_image(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    try:  # If in case face is not detected at any frame
        face = face_detector(image, 1)[0]  # Face detection
        # Calling to get bound box around the face
        x, y, size = get_boundingbox(face=face, width=width, height=height)
    except IndexError:
        pass
    cropped_face = image[y:y+size, x:x+size]  # cropping the face
    # Sending the cropped face to get classifier result
    output, label = evaluate(cropped_face)
    print("This image could be ", label, " and the possibility is", output)
    return output, label


def avi_to_mp4(filename, new_filename):
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(new_filename, fourcc, frames_per_second, (frameWidth, frameHeight), isColor=True)

    # Keep writing until the VideoCapture will stay open
    while cap.isOpened():
        ret, frame  = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    # Release everything
    cap.release()
    out.release()


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/images')
def images():
    return render_template('image.html')

@app.route('/images/check', methods=["GET", "POST"])
def images_check():
    if request.method == "POST":
        import PIL.Image as IImage
        file = request.files['image']
        img = IImage.open(file.stream)
        request.files['image'].seek(0)
        data = file.stream.read()

        # img_data = request.files.get('image').read()
        folder_name = "queried_images"
        timestr = time.strftime("%Y%m%d-%H%M%S.jpg")
        filename = os.path.join(folder_name, timestr)
        with open(filename, 'wb') as f:
            f.write(data)

        output, label = test_for_image(filename)

        data = base64.b64encode(data).decode()
        output = {"label": label, "score": output, "format": img.format, "img": data}
        return render_template('image_detail.html', **output)
    return render_template('image_detail.html')

@app.route('/videos')
def videos():
    return render_template('video.html')


@app.route('/show_video/<filename>')
def show_video(filename):
    return send_from_directory('./test_video_results', filename)


@app.route('/videos/check', methods=["GET", "POST"])
def videos_check():
    if request.method == "POST":
        imgdata = request.files.get('video').read()
        folder_name = "queried_videos"
        timestr = time.strftime("%Y%m%d-%H%M%S.mp4")
        filename = os.path.join(folder_name, timestr)
        with open(filename, 'wb') as f:
            f.write(imgdata)

        output_path = test_for_video(filename)

        print("videoId:" + output_path)
        # filename = os.path.basename(output_path)
        # new_filename = filename + "_new.webm"
        # avi_to_mp4(output_path, "./test_video_results/" + new_filename)
        output = {"videoId": output_path, "filename": output_path}
        return render_template('video_detail.html', **output)
    return render_template('video_detail.html')


@app.route('/old')
def home():
    return render_template('index.html')


class Image(Resource):

    def post(self):
        args = parser.parse_args()
        the_file = args['file']
        ofile, ofname = tempfile.mkstemp()
        the_file.save(ofname)
        output, label = test_for_image(ofname)
        output = {"label": label, "score": output}
        return output


class Video(Resource):

    def post(self):
        args = parser.parse_args()
        the_file = args['file']
        ofile, ofname = tempfile.mkstemp()
        the_file.save(ofname)
        output_path = test_for_video(ofname)
        output = {"videoId": output_path}
        return output


class QueryImage(Resource):

    def post(self):
        #img_data = request.form.get('image')
        img_data = request.files.get('image').read()
        # imgdata = base64.b64decode(img_data)
        folder_name = "queried_images"
        timestr = time.strftime("%Y%m%d-%H%M%S.jpg")
        filename = os.path.join(folder_name, timestr)
        with open(filename, 'wb') as f:
            f.write(img_data)

        output, label = test_for_image(filename)
        output = {"label": label, "score": output}
        return output


class QueryVideo(Resource):

    def post(self):
        # img_data = request.form.get('video')
        # imgdata = base64.b64decode(img_data)
        imgdata = request.files.get('video').read()
        folder_name = "queried_videos"
        timestr = time.strftime("%Y%m%d-%H%M%S.mp4")
        filename = os.path.join(folder_name, timestr)
        with open(filename, 'wb') as f:
            f.write(imgdata)

        output_path = test_for_video(filename)
        output = {"videoId": output_path}
        print("videoId:" + output_path)
        return output


api.add_resource(Image, '/testImage')
api.add_resource(Video, '/testVideo')
api.add_resource(QueryImage, '/queryImage')
api.add_resource(QueryVideo, '/queryVideo')

if __name__ == '__main__':
    app.run(debug=True)


def test_for_image2(image_path):
    dlib.get_frontal_face_detector(image_path)
    image = dlib.load_rgb_image(image_path)
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dlib.detector(image)
    height, width = image.shape[:2]
    try:
        face = face_detector(image, 1)[0]
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    except IndexError:
        pass
    x,y,size  = face.left(), face.top(), (face.right(), face.bottom())
    cropped_face = image[y:y+size, x:x+size]
    output, label = evaluate(cropped_face)
    print("This image could be ", label, " and the possibility is", output)
    return output, label





def test_for_image3(image_path):
    image = cv2.imread(image_path)
    detector = MtcnnDetector
    all_boxes,landmarks = detector.detect_face(image)
    height, width = image.shape[:2]
    try:
        face = face_detector(image, 1)[0]
        cv2.rectangle(image, plt.box, (0, 0, 255))
        x, y, size = get_boundingbox(face=face, width=width, height=height)
    except IndexError:
        pass
    cropped_face = image[y:y+size, x:x+size]
    output, label = evaluate(cropped_face)
    print("This image could be ", label, " and the possibility is", output)
    return output, label