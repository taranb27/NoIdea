from ultralytics import YOLO
# from IPython.display import display, Image

model = YOLO("yolov8s.pt")

model.train(
    data="/Users/sneha/NoIdea/new_dataset/data.yaml",
    epochs=25,
    imgsz=640,
    plots=True
)