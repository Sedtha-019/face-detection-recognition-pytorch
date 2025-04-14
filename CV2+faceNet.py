import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
from tqdm import tqdm
import cvzone

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskModel, self).__init__()

        self.backbone = InceptionResnetV1(pretrained='vggface2')
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x) 
        class_logits = self.classifier(features)
        
        return class_logits

num_classes = 2
net = MultiTaskModel(num_classes)


def load_model(model_path, num_classes=2):
    model = MultiTaskModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_face_for_model(face_img):
    # Convert BGR to RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(face_img)
    
    # Apply the same transforms as during training
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_img).unsqueeze(0)

def main():
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model_path = r'C:\Users\sedth\PycharmProjects\pythonProject\FaceNet\best_model.pth'  # Path to your saved model
    model = load_model(model_path).to(device)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Define class names
    class_names = {
        0: "kleom",    # Replace with your actual class names
        1: "Pao"
    }
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Prepare face for model
                face_tensor = process_face_for_model(face_roi)
                face_tensor = face_tensor.to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][predicted_class].item()
                
                # Draw rectangle and prediction
                color = (190, 0, 200) if predicted_class == 0 else (190, 190, 0)  # Green for class 0, Red for class 1
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Add prediction text
                label = f"{class_names[predicted_class]}: {confidence:.2%}"
                cvzone.putTextRect(frame, label, (max(0, x - 10), max(0, y - 10)),
                               scale=1, thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=(200, 0, 0))
                # cv2.putText(frame, label, (x, y-10), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Display the frame
        cv2.imshow('Face Classification', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()