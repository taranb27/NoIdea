from ultralytics import YOLO

# model = YOLO("yolov8s.pt")
model = YOLO("runs/detect/train8/weights/best.pt")

# model.train(
#     data="/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/data.yaml",
#     epochs=25,
#     imgsz=224,
#     plots=True
# )

model.predict(
    conf=0.25,
    source="Dataset/test/images"
)