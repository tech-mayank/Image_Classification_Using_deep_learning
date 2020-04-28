from __future__ import division, print_function
from flask import Flask,redirect,render_template,request,url_for
import os
from os.path import join
import tensorflow
from tensorflow.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from werkzeug.utils import secure_filename


app = Flask(__name__)

model = ResNet50(weights = 'imagenet')
def model_predict(img_path,model):
    img = load_img(img_path,target_size=(224,224))
    x =  img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x,mode='caffe')
    pred = model.predict(x)
    return pred


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # file_name = f.filename
        # f.save(file_name)
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        # file_name = f.filename
        # f.save(file_name)
        


        preds = model_predict(file_path,model)

        pred_class = decode_predictions(preds,top=1)
        result = str(pred_class[0][0][1])
        return result
    return None

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

if __name__ == '__main__':
    app.run(debug=True)
    





