import cv2
from ultralytics import YOLO

# Load model
model = YOLO('../models/yolov8n.pt')  # Replace with your custom-trained weights

# Load and predict on image
img = cv2.imread('../data/sample_image.jpg')
results = model(img)

# Display results
results[0].plot()
cv2.imshow("Detection", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
