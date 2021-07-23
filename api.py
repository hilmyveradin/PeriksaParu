import os
import numpy
from io import open
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
UPLOAD_FOLDER = "./static/images/"
DEVICE = "cuda"

mnet = mobilenet_v2(pretrained=True)
for param in mnet.parameters():
    param.requires_grad = False

class MyCustomMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=True)
        self.mnet.classifier = nn.Sequential(
            nn.Linear(1280, 5),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.mnet(x)
    
MODEL = MyCustomMobileNetV2()
MODEL.load_state_dict(torch.load("./test_model/weights_best_with random.pth", map_location=torch.device('cpu')))

Text = ""
def predict(image_path, model):

    global Text

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_path = Image.open(image_path).convert("RGB")
    test_images = image_path

    input = transform(test_images)
    input.unsqueeze_(0)

    model.eval()
    output = model(input)
    output1 = output.detach().numpy()
    result = output1[0,0]
    
    if (output.argmax(1)) == 0:
        Text = "COVID19"
        result = output1[0,0]*100
    elif (output.argmax(1)) == 1:
        Text = "NORMAL"
        result = output1[0,1]*100
    elif (output.argmax(1)) == 2:
        Text = "PNEUMONIA"
        result = output1[0,2]*100
    elif (output.argmax(1)) == 4:
        Text = "TUBERKULOSIS"
        result = output1[0,4]*100
    else:
        Text = "Maaf, gambar yang kamu masukkan tidak dapat diprediksi"
        result = 0

    result = "{0:.3f}".format(result)
    return result
    

@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def upload_predict():
    image_file = request.files["image"]
    image_location = "./static/images/" + image_file.filename
    image_file.save(image_location)
    pred = predict(image_location, MODEL)
    return render_template("index.html", prediction = pred, TextResult=Text, image_loc=image_file.filename)

@app.route("/test")
def test():
    return render_template("test.html")

if __name__ == "__main__":
    MODEL = MyCustomMobileNetV2()
    MODEL.load_state_dict(torch.load("/test_model/weights_best_with random.pth", map_location=torch.device('cpu')))
    app.run(debug=True)
