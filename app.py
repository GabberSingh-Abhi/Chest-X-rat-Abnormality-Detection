from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
import pickle
from PIL import Image
import torchvision
import torch.nn as nn
import torch
from flask import Flask
from flask_ngrok import run_with_ngrok

basedir = os.path.abspath(os.path.dirname(__file__))

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "app"
        return super().find_class(module, name)
map_location=torch.device('cpu')
with open(basedir+'/chexnet.pkl','rb') as f:
    unpickler = MyCustomUnpickler(f)
    model = unpickler.load()

model=pickle.load(open(basedir+'/chexnet.pkl','rb'))
app=Flask(__name__)
run_with_ngrok(app)   

UPLOAD_FOLDER="static/image"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/overall',methods=["GET","POST"])
def overall():
    if request.method=="POST":
        fname=request.form.get("name")
        phone=request.form.get("phone")
        age=request.form.get("age")
        image_file=request.files['file']
        print("image ",image_file)
        if image_file:
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename))
            location = os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename)
            print(location)
            

            classes = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis','Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Hernia', 'Mass', 'No Finding']
            image = Image.open(location).convert('RGB')
            preprocess = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])
            image = preprocess(image)
            image = image.unsqueeze(0)
            print(image.size())
            print(image.size())
            outputs = model(image)
            index_tensor = torch.argmax(outputs)
            index = index_tensor.item()
            print(classes[index])
    return render_template('result.html',prediction=classes[index],fname=fname,phone=phone,age=age,xray_image=image_file.filename)



@app.route("/construct",methods=["GET"])
def construct():
    return render_template("construct.html")
    

print(UPLOAD_FOLDER)
if __name__=="__main__":
    app.run()

