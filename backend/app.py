import os
import random
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import json
import torch.nn as nn


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SMALL_DATASET_FOLDER = 'small_dataset/test/utkcropped'

# Load the model
class AgeCNN(nn.Module):
    def __init__(self):
        super(AgeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.flatten(x)
        x = self.act4(self.fc1(x))
        x = self.fc2(x)
        return x
  
model = AgeCNN()
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to what the model expects
    transforms.ToTensor(),          # Convert the image to a tensor
])

# Path for saving uploaded files - ensure this directory exists
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def home():
    return render_template('base.html', title="Upload Image for Age Prediction")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Load the image and transform
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            predicted_age = outputs.item()

        return jsonify({'predicted_age': predicted_age})

    return jsonify({'error': 'Something went wrong'})

@app.route("/play", methods=["GET"])
def play():
    # Get a list of all images in the small dataset folder
    images = os.listdir(SMALL_DATASET_FOLDER)
    # Choose a random image from the list
    image_name = random.choice(images)
    image_path = os.path.join(SMALL_DATASET_FOLDER, image_name)
    # Predict the age of the image using the model
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        model_output = model(image_tensor)
        predicted_age = int(model_output.item())
    # Construct the image URL to pass to the template
    image_url = f"{SMALL_DATASET_FOLDER}/{image_name}"
    return render_template('play.html', title="Play Against Model", image_url=image_url, predicted_age=predicted_age)

@app.route("/play", methods=["POST"])
def submit_guess():
    global points
    user_guess = int(request.form['guess'])
    predicted_age = int(request.form['predicted_age'])
    difference = abs(user_guess - predicted_age)
    if difference <= 5:  
        points += 1
        message = "Congratulations! You guessed close enough. You get 1 point."
    else:
        message = f"Sorry, the model's prediction was {predicted_age}. You were {difference} years off."
    response = {
        "message": message,
        "image_url": request.form['image_url'],
        "predicted_age": predicted_age,
        "points": points
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
