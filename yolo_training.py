from ultralytics import YOLO
# from IPython.display import display, Image

model = YOLO("yolov8s.pt")

model.train(
    data="/Users/tb/Desktop/Robotics/cw/NoIdea/Dataset/data.yaml",
    epochs=25,
    imgsz=224,
    plots=True
)

