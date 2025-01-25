from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# Load the model (assumes the model directly outputs disease names)
model = torch.load('skin_disease_model.pth', map_location=torch.device('cpu'))
#model.eval()  # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the input size expected by your model
    transforms.ToTensor(),         # Convert image to a PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (adjust based on your model training)
])

@app.route('/')
def home():
    return render_template("skin_disease_prediction.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        return render_template('skin_disease_prediction.html', pred='No file uploaded.', action="Please upload an image.")

    file = request.files['file']

    if file.filename == '':
        return render_template('skin_disease_prediction.html', pred='No file selected.', action="Please upload an image.")

    try:
        # Open the uploaded image
        img = Image.open(file).convert('RGB')

        # Apply the transformations
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension (1, C, H, W)

        # Predict using the model
        with torch.no_grad():
            prediction = model(img)  # Assuming the model outputs the disease name directly

        # Ensure prediction is in a readable format (e.g., string)
        disease = prediction[0] if isinstance(prediction, list) else prediction.item()

        return render_template('skin_disease_prediction.html', 
                               pred=f'The model predicts the disease as: {disease}', 
                               action="Please consult a dermatologist for further guidance.")
    except Exception as e:
        return render_template('skin_disease_prediction.html', 
                               pred=f'Error processing image: {str(e)}', 
                               action="Please try again with a valid image.")

if __name__ == '__main__':
    app.run(debug=True)
