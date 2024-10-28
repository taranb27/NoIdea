import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifer10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Dog', 'Frog', 'Truck']
