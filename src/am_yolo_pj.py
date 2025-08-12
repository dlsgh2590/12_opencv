from ultralytics import YOLO
import cv2

# ==============================
video_path = 'walking.avi'         # 사용할 영상 경로
desired_classes = [0, 2, 7]         # 탐지할 객체 클래스 번호 (예: 사람, 자동차, 트럭)
model_path = 'yolo11n.pt'        # 변경된 모델 경로

#coco 클래스 번호
#0	person
#1	bicycle
#2	car
#3	motorcycle
#5	bus
#7	truck

# ==============================
model = YOLO(model_path)

class_names = model.model.names

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id in desired_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = class_names[cls_id]
                conf = box.conf.item()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv8n Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()