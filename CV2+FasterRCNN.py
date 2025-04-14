import cv2
import cvzone
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

# Load the model
try:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3  # Background + Kloem + Pao
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, num_classes
    )
    model.load_state_dict(torch.load(r"C:\Users\sedth\PycharmProjects\pythonProject\FaceNet\fasterrcnn_model_with_face2.pth"))
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Class labels
label_map = {"Kloem": 0, "Pao": 1}  # Mapping class indices to labels

# Preprocess a single frame
def preprocess_frame(frame):
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = F.to_tensor(image).unsqueeze(0).to(device)
        return image
    except Exception as e:
        print(f"Error in preprocessing frame: {e}")
        return None

# Draw bounding boxes on a frame
def draw_boxes(frame, boxes, labels, scores, threshold=0.45):
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 180), 2)
            text = f"{label_map.get(label)}: {score:.2f}"  # Default to 'Unknown' for invalid labels
            cvzone.putTextRect(frame, text, (max(0, x1 - 10), max(0, y1 - 10)),
                               scale=1, thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=(200, 0, 0))
    return frame

# Open webcam
# video_path = r"C:\Users\sedth\PycharmProjects\pythonProject\FaceNet\1.mp4" # Path to your video file
cap = cv2.VideoCapture(0)  # Open webcam (use video path if reading a video file)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Starting webcam. Press 'k' to quit.")

# Process each frame from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from webcam.")
        break

    # Preprocess the frame
    input_tensor = preprocess_frame(frame)
    if input_tensor is None:
        continue

    # Make predictions
    with torch.no_grad():
        predictions = model(input_tensor)

    # Parse predictions
    boxes = predictions[0]['boxes'].cpu().numpy() if 'boxes' in predictions[0] else []
    labels = predictions[0]['labels'].cpu().numpy() if 'labels' in predictions[0] else []
    scores = predictions[0]['scores'].cpu().numpy() if 'scores' in predictions[0] else []

    # Draw bounding boxes on the frame
    frame = draw_boxes(frame, boxes, labels, scores)

    # Display the frame
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('k'):  # Press 'k' to quit
        print("Webcam closed by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam processing stopped.")
