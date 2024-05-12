import os
import random
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import base64


app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'key'

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),        
])

@app.route("/", methods=["GET"])
def home():
    return render_template('base.html', title="Upload Image for Age Prediction")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    if file:
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            predicted_age = outputs.item()
        return jsonify({'predicted_age': predicted_age})
    return 0

@app.route("/play", methods=["GET", "POST"])
def play():
    if request.method == "GET":
        if 'image_count' not in session or session['image_count'] >= 20:
            session['points'] = 0
            session['image_count'] = 0
            session['user_cumulative_diff'] = 0 
            session['model_cumulative_diff'] = 0
            all_images = os.listdir("val_20")
            random.shuffle(all_images)
            session['image_list'] = all_images[:20]   
        image_index = session['image_count']
        img = session['image_list'][image_index]
        predicted_age, actual_age, image_url = fetch_new_image_and_age(img)

        session['predicted_age'] = predicted_age
        session['actual_age'] = actual_age
        session['image_count'] += 1

        return render_template('play.html', title="Play Against Model", image_url=image_url, predicted_age=predicted_age, actual_age=actual_age)

    elif request.method == "POST":
        user_guess = int(request.form['guess'])
        predicted_age = session.get('predicted_age')
        actual_age = session.get('actual_age')

        user_difference = abs(user_guess - actual_age)
        model_difference = abs(predicted_age - actual_age)

        session['user_cumulative_diff'] = session.get('user_cumulative_diff', 0) + user_difference**2
        session['model_cumulative_diff'] = session.get('model_cumulative_diff', 0) + model_difference**2

        points = session.get('points', 0)
        if user_difference < model_difference:
            points += 1
            message = f"You win. Actual: {actual_age}, Model Guess: {predicted_age}."
        elif user_difference == model_difference:
            points +=1
            message = f"Tie. Actual: {actual_age}, Model Guess: {predicted_age}."
        else:
            message = f"Model wins. Actual: {actual_age}, Model Guess: {predicted_age}."
        
        session['points'] = points

        response = {
            "message": message,
            "points": points
        }
        return jsonify(response)
    
def fetch_new_image_and_age(image_name):
    image_path = os.path.join("val_20", image_name)
    image = Image.open(image_path).convert('RGB')
    
    with torch.no_grad():
        model_output = model(transform(image).unsqueeze(0))
        predicted_age = int(model_output.item())

    actual_age = int(os.path.basename(image_path).split("_")[0])

    with open(image_path, "rb") as f:
        image_url = f"data:image/jpg;base64,{base64.standard_b64encode(f.read()).decode('utf-8')}"

    return predicted_age, actual_age, image_url
 
@app.route("/get-new-image", methods=["GET"])
def get_new_image():
    if session['image_count'] >= 20:
        score = session.get('points', 0)
        user_rms = (session.get('user_cumulative_diff', 0) / 20)**(1/2)
        model_rms = (session.get('model_cumulative_diff', 0) / 20)**(1/2)
        return jsonify({
            'game_over': True,
            'message': f"Game over. Thanks! Your Score: {score} User: {user_rms:.2f}, Model: {model_rms:.2f}",
        })
    if 'image_list' in session and session['image_count'] < len(session['image_list']):
        predicted_age, actual_age, image_url = fetch_new_image_and_age(session['image_list'][session['image_count']])

    session['predicted_age'] = predicted_age
    session['actual_age'] = actual_age
    session['image_count'] += 1

    return jsonify({
        'image_url': image_url,
        'predicted_age': predicted_age,
        'actual_age': actual_age,
        'image_count': session['image_count']
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
