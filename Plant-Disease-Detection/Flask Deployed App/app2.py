import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


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