import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_model(path='mnist_cnn.pth'):
    model = Net()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_canvas_image(image):
    # Convert to grayscale
    image = image.convert("L")
    # Resize to 28x28
    image = image.resize((28, 28))
    # Invert colors (white digit on black background)
    image = ImageOps.invert(image)
    # Apply a slight blur to the image to reduce noise
    image = image.filter(ImageFilter.GaussianBlur(1))
    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)
    return image

def preprocess_uploaded_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict_digit(image, model, model_type=1):
    if model_type == 1:
        image = preprocess_canvas_image(image)
    else:
        image = preprocess_uploaded_image(image)
    
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()
