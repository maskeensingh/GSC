import torch
import torchvision.transforms as transforms
from PIL import Image
import logging

def load_image_model(model_path):
    try:
        # Load the complete model
        image_model = torch.load(model_path, map_location=torch.device('cpu'))
        image_model.eval()
        logging.info("Image model loaded successfully.")
        return image_model
    except Exception as e:
        logging.error(f"Error loading image model: {e}")
        raise e

def map_disease_label(index):
    """
    Maps a class index to the corresponding disease name.
    Update this mapping based on your model's training labels.
    """
    disease_labels = {
        0: 'Acne',
        1: 'Eczema',
        2: 'Psoriasis',
        3: 'Rosacea',
        4: 'Vitiligo',
        5: 'Melanoma',
        6: 'Dermatitis',
        7: 'Hives',
        8: 'Warts',
        9: 'Other'
        # Add all necessary mappings
    }
    return disease_labels.get(index, 'Unknown Disease')

def predict_disease_from_image(image_model, image_file):
    try:
        image = Image.open(image_file).convert('RGB')
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size as per model's input
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # Mean values
                                 [0.229, 0.224, 0.225])  # Std deviation
        ])
        image_tensor = image_transforms(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = image_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(probabilities, 1)
            predicted_disease = map_disease_label(predicted_idx.item())
            return predicted_disease
    except Exception as e:
        logging.error(f"Error during image prediction: {e}")
        return "Prediction Failed"
