import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/train8/weights/best.pt")

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    desired_size = (800, 800)
    image_resized = cv2.resize(image_np, desired_size, interpolation=cv2.INTER_LINEAR)

    results = model.predict(image_resized)

    for result in results:
        boxes = result.boxes.xyxy.numpy()
        confidences = result.boxes.conf.numpy()
        class_ids = result.boxes.cls.numpy()

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            label = f"{model.names[int(class_id)]}: {conf:.2f}"
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_np, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    annotated_image = Image.fromarray(image_np)
    return annotated_image

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        processed_image = process_image(file_path)
        # processed_image.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(processed_image)
        
        image_label.config(image=img_tk)
        image_label.image = img_tk

def live_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            class_ids = result.boxes.cls.numpy()

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                label = f"{model.names[int(class_id)]}: {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Live Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

window = tk.Tk()
window.title("UNO Card Recognition")
window.geometry("500x600")

label = tk.Label(window, text="Choose input method:")
label.pack(pady=10)

image_label = tk.Label(window)
image_label.pack(pady=10)

file_button = tk.Button(window, text="Upload Image", command=open_file)
file_button.pack(pady=10)

stream_button = tk.Button(window, text="Start Live Stream", command=live_stream)
stream_button.pack(pady=10)

window.mainloop()
