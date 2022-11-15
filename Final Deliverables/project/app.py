from flask import Flask, render_template, request
from imageio import imsave,imread
import numpy as np
from PIL import Image
import tensorflow.keras.models
import re
import base64

import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import *

app = Flask(__name__)
global model, graph
model, graph = init()
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = Image.open('output.png')
    x = x.convert('L')
    x = x.resize((28,28))
    x = np.asarray(x)
    print(x.shape)
    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    response = str(np.argmax(out, axis=1)[0])
    return response 
    
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)
