import os
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import pandas as pd

# --- 1. DEFINE VGG16 MODEL ARCHITECTURE ---
def create_vgg16_model(targets_size=38): 
    model = models.vgg16(pretrained=False)
    n_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, targets_size)
    )
    return model

# --- 2. LOAD DATA AND MODEL (WITH FIXES) ---
try:
    disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
    supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')
except FileNotFoundError:
    print("Error: Make sure 'disease_info.csv' and 'supplement_info.csv' are in the same directory.")
    exit()

# Instantiate the correct model structure
# *** IMPORTANT: Make sure 38 is the correct number of classes for your model ***
model = create_vgg16_model(38) 

# Load the model with map_location to run on a CPU
try:
    model.load_state_dict(torch.load("plant_disease_vgg16_augmented.pt", map_location=torch.device('cpu')))
except FileNotFoundError:
    print("Error: The model file 'plant_disease_vgg16_colab.pt' was not found.")
    exit()

model.eval()

# Use the exact same image transforms as in training
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# --- 3. PREDICTION FUNCTION (UPDATED) ---
def prediction(image_path):
    try:
        image = Image.open(image_path).convert('RGB') # Ensure image is in RGB
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None # Handle error gracefully

    # Apply the transformations
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    index = output.argmax().item()
    return index

# --- 4. FLASK APP SETUP ---
app = Flask(__name__)

# Ensure the 'uploads' directory exists
uploads_dir = os.path.join('static', 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def ai_engine_page():
    # Serve the main page
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return redirect(request.url) 

        image = request.files['image']
        filename = image.filename
        file_path = os.path.join(uploads_dir, filename)
        image.save(file_path)
        
        # Get prediction
        pred = prediction(file_path)

        if pred is None or pred >= len(disease_info):
            # Handle cases where prediction fails or index is out of bounds
            return "Error processing image or finding disease info. Please try again."

        # Fetch information from dataframes
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        
        # We pass the original filename to display the uploaded image
        # image_url = disease_info['image_url'][pred] # We don't need this anymore
        
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                               filename=filename, # Pass the filename of the uploaded image
                               pred=pred, sname=supplement_name, 
                               simage=supplement_image_url, buy_link=supplement_buy_link)
    
    # If it's a GET request, redirect to the main page
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)