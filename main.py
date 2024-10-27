# import numpy as np
# import cv2

# card_cascade=cv2.CascadeClassifier("classifier/one_green.xml")
# img = cv2.imread("images/one_green.jpg")
# resized = cv2.resize(img, (400, 200))
# gray=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# card=card_cascade.detectMultiScale(gray, 6.5, 17)

# for(x,y,w,h) in card:
#     resized=cv2.rectangle(resized, (x,y),(x+w,y+h),(0,255,0),2)

# cv2.imshow('img', resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import pandas as pd
import cv2

# Load CSV annotations
annotations = pd.read_csv('classifier/labels_my-project-name_2024-10-27-07-05-25.csv')  # Replace with your CSV file path

def detect_in_image(image_path, annotations):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Filter annotations for the specific image
    img_annotations = annotations[annotations['filename'] == image_path]

    # Draw bounding boxes for each annotation
    for _, row in img_annotations.iterrows():
        x_min = int(row['x_min'])
        y_min = int(row['y_min'])
        x_max = int(row['x_max'])
        y_max = int(row['y_max'])
        label = row['label']

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        # Put label above the bounding box
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with detections
    cv2.imshow('Detected Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to run the detection
if __name__ == "__main__":
    # Replace with the path to the image you want to test
    image_path = 'images/one_green.jpg'  # Update this to your image file path

    detect_in_image(image_path, annotations)
