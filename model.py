import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load model once at startup
model = DigitClassifier()
model.load_state_dict(torch.load('mnist_digit_classifier.pth', map_location='cpu'))
model.eval()

def predict_digit(image):
    # Convert to grayscale and resize
    img = image.convert('L').resize((28, 28), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    
    # Invert if white background (MNIST is white-on-black)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Normalize: [0,255] → [0,1] → MNIST standard
    img_array = (img_array / 255.0 - 0.1307) / 0.3081
    
    # Predict
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item() * 100