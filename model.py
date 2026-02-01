import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

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

print("🔄 Loading model...")
model = DigitClassifier()
model.load_state_dict(torch.load('mnist_digit_classifier.pth', map_location='cpu'))
model.eval()
print("✅ Model loaded!")

# EXACT same transform as training
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(image):
    # Convert to grayscale
    img = image.convert('L')
    
    # CRITICAL: Invert if white background
    # MNIST = white digit on black (pixel values: digit=255, bg=0)
    # Canvas = black digit on white (pixel values: digit=0, bg=255)
    img = ImageOps.invert(img)
    
    # Apply same transform as training
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    print(f'Predicted is : {predicted}')    
    return predicted.item(), confidence.item() * 100