from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# ----------- 1. Load your trained model --------------
model = YOLO("best-5.pt")  # <- use your .pt file

# ----------- 2. Run inference ------------------------
image_path = "img_2.png"


  # model() returns a list, take first result

# ----------- 3. Load image with OpenCV --------------
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.resize(img, (640,640))

results = model(img)[0]



# ----------- 4. Draw masks and boxes ----------------
# masks (optional, only if segmentation result exists)
if results.masks:
    for mask in results.masks.data:
        # Convert mask tensor to numpy (and scale it to 0–255)
        m = mask.cpu().numpy() * 255
        m = m.astype('uint8')

        # Resize if needed
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]))

        # Create color overlay
        color = (0, 255, 0)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, thickness=2)

# boxes
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{model.names[cls_id]} {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# ----------- 5. Show the result -----------------------
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title("Segmentation + Boxes")
plt.show()
