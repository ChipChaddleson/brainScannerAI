from io import BytesIO
import webview
from flask import Flask, render_template, request
from base64 import b64encode
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims, argmax, max
from os import makedirs
from os.path import join, dirname, exists

uniqueLables = ["glioma", "menengioma", "no tumor", "pituitary"]


try:
    model = load_model('./modeltst.keras')
    print("LOADED MODEL")
except Exception as e:
    print(e)





def prd(image_data):
    image = load_img(BytesIO(image_data), target_size=(512, 512))
    image = img_to_array(image)
    image = expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(f"prediction: {prediction}")
    predictedLabel = uniqueLables[argmax(prediction)]
    print(f"predictiedLable: {predictedLabel}")
    certainty = round(max(prediction) * 100)

    return predictedLabel, certainty



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def uploadFile():
    if request.method == 'POST':
        uploadedFile = request.files['file']
        if uploadedFile:
            image_data = uploadedFile.read()
            prdLabel, cert = prd(image_data)
            if cert >= 90:
                cert = f"high {cert}% certainty"
            elif cert >= 80:
                cert = f"{cert}% certainty"
            elif cert >=65:
                cert = f"low certainty of {cert}%"
            else:
                cert = f"very low certainty, only {cert}%"
            encodedString = b64encode(image_data).decode('utf-8') # turn file into base63
            return render_template('results.html', imageData=encodedString, label=prdLabel, certainty=cert)

        else:
            return 'No file uploaded'

def runFlaskServer():
    app.run(debug=False, host="0.0.0.0", port=80)
if __name__ == '__main__':
    webview.create_window('Geeks for Geeks', 'http://127.0.0.1:80') 
    webview.start(runFlaskServer)
    
