import os
import numpy as np
from PIL import Image
import pickle
directory = "E:/Datasets/tiny-imagenet-200/tiny-imagenet-200/train"
list_classes_folders = os.listdir(directory)
image_counter = 0
class_counter = 0
images = None
labels = None
for class_folder in list_classes_folders:
    list_image_dirs = os.listdir(directory+"/"+class_folder+"/images")
    print(directory+"/"+class_folder+"/images")
    print("counter", image_counter)
    for image_dir in list_image_dirs:
        image = Image.open(directory+"/"+class_folder+"/images/"+image_dir)
        image_array = np.asarray(image)
        if len(image_array.shape) != 3:
            stacked = np.stack((image_array,image_array), axis = 2)
            stacked_twice = np.concatenate((stacked, image_array.reshape(64,64,1)), axis = 2)
            image_array = stacked_twice
        image_array = image_array.reshape(1,64,64,3)
        if image_counter == 0:
            images = image_array
            labels = class_folder
        else:
            images = np.vstack((images, image_array))
            labels = np.vstack((labels, class_folder))
        image_counter += 1
    class_counter += 1
print("counter", counter)
print("class_counter", class_counter)
print(images.shape)
print(labels.shape)

pickle.dump(images, open("training_data.p", "wb"))
pickle.dump(images, open("training_labels.p", "wb"))
