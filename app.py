import os
import glob
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

# =================== SETUP ===================
# Inisialisasi Flask
app = Flask(__name__, static_folder='web/static', template_folder='web')

# Folder dan path penting
YOLO_MODEL_PATH = "model/yolov8n.pt"
CNN_MODEL_PATH = "model/cnn_model.pth"
UPLOAD_DIR = "input"
CROPPED_DIR = "cropped"
TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_LABEL_DIR = "dataset/train/labels"
class_names = ['Elang Jawa', 'Jalak Bali', 'Kakatua Putih', 'Nuri Hitam']
# Remove this line since we set template_folder in the Flask initialization
# app.template_folder = TEMPLATE_FOLDER
TEMPLATE_FOLDER = 'web'
class_names = ['Elang Jawa', 'Jalak Bali', 'Kakatua Putih', 'Nuri Hitam']
app.template_folder = TEMPLATE_FOLDER

# Pastikan folder yang dibutuhkan ada
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =================== CLASS: Dataset ===================
class RoboflowDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        with open(label_path, "r") as f:
            label = int(f.readline().split()[0])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# =================== CLASS: CNN ===================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 62, 62]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 14, 14]
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# =================== FUNGSI: Retrain CNN ===================
def retrain_cnn(model, device, model_path, transform, epochs=20):
    print("[5] Retraining CNN with new data...")
    train_dataset = RoboflowDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform)
    
    # Skip retraining if there's no data
    if len(train_dataset) == 0:
        print("[5] No training data available. Skipping retraining.")
        return False
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"    Retrain Epoch {epoch+1} Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[5] Retrained model saved to {model_path}")
    return True

# =================== LOAD MODEL ===================
print("[1] Loading YOLOv8 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("[2] Loading CNN model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SimpleCNN().to(device)

# Create a flag to track if we need to retrain
RETRAIN_NEEDED = False

if not os.path.exists(CNN_MODEL_PATH):
    print("[2] CNN model not found. Training CNN from scratch...")
    train_dataset = RoboflowDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform)
    
    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

        for epoch in range(20):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = cnn_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"    Epoch {epoch+1} Loss: {running_loss:.4f}")

        torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)
        print(f"[2] CNN model saved at {CNN_MODEL_PATH}")
    else:
        print("[2] No training data available. Using untrained model.")
else:
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    print("[2] Loaded pre-trained CNN model.")

cnn_model.eval()

# =================== ROUTING WEB ===================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join(UPLOAD_DIR, str(file.filename))
            file.save(img_path)
            return redirect(url_for('predict', filename=file.filename))
    return render_template('index.html')

@app.route('/predict/<filename>')
def predict(filename):
    image_path = os.path.join(UPLOAD_DIR, filename)
    results = yolo_model(image_path)[0]
    image = cv2.imread(image_path)
    crops_data = []

    if not results.boxes:
        return f"Tidak ada objek terdeteksi pada {filename}"

    # Track highest confidence prediction
    highest_confidence = -1
    highest_confidence_idx = -1

    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        crop_filename = f"crop_{i}.jpg"
        crop_path = os.path.join(CROPPED_DIR, crop_filename)
        cv2.imwrite(crop_path, crop)

        pil_crop = Image.open(crop_path).convert('RGB')
        input_tensor = transform(pil_crop).unsqueeze(0).to(device)
        output = cnn_model(input_tensor)
        
        # Get prediction and confidence
        pred = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0][pred].item() * 100
        
        # Track highest confidence
        if confidence > highest_confidence:
            highest_confidence = confidence
            highest_confidence_idx = i

        crops_data.append({
            'img': crop_filename,
            'label': class_names[pred],
            'confidence': round(confidence, 2),
            'is_highest': False  # Will be updated later
        })

        # Save to dataset for future retraining (but don't retrain now)
        train_img_name = f"crop_{i}_train.jpg"
        train_img_path = os.path.join(TRAIN_IMG_DIR, train_img_name)
        pil_crop.save(train_img_path)

        label_path = os.path.join(TRAIN_LABEL_DIR, train_img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write(f"{pred} 0.5 0.5 1.0 1.0\n")  # Format label dummy (YOLO)
        
        # Set flag to indicate we have new data for retraining
        global RETRAIN_NEEDED
        RETRAIN_NEEDED = True

    # Mark the highest confidence prediction
    if highest_confidence_idx >= 0:
        crops_data[highest_confidence_idx]['is_highest'] = True

    # No retraining here - we'll do it in a separate route
    return render_template('results.html', crops_data=crops_data)

# Add a new route for manual retraining
@app.route('/retrain')
def retrain():
    global RETRAIN_NEEDED
    if RETRAIN_NEEDED:
        success = retrain_cnn(cnn_model, device, CNN_MODEL_PATH, transform, epochs=20)
        if success:
            RETRAIN_NEEDED = False
            return "Model successfully retrained with new data."
        else:
            return "No data available for retraining."
    else:
        return "No new data available. Retraining not needed."

@app.route('/cropped/<filename>')
def cropped_file(filename):
    return send_from_directory(CROPPED_DIR, filename)

@app.route('/input/<filename>')
def serve_image_input(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# =================== MAIN ===================
if __name__ == '__main__':
    app.run(debug=True)
