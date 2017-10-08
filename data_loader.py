import os
import numpy as np
import pickle
from PIL import Image
class data_loader(object):
    def __init__(self):
        self.create_lookup_tables()

    def load_data_arrays(self):
        self.data = pickle.load(open("training_data.p"), "rb")
        self.labels = pickle.load(open("training_labels.p"), "rb")

    def create_lookup_tables(self):
        self.str_to_class_lookup = {}
        self.class_to_str_lookup = {}
        list_str = os.listdir("E:/Datasets/tiny-imagenet-200/tiny-imagenet-200/train")
        counter = 0
        for string in list_str:
            self.str_to_class_lookup[string] = counter
            self.class_to_str_lookup[counter] = string
            counter += 1

    def shuffle_data(self):
        #new random permutation
        perm = np.arange(100000)
        np.random.shuffle(perm)

        #shuffle images
        shuffled_images = np.zeros(self.data.shape)
        shuffled_images[range(perm.size)] = self.data[perm]

        #shuffle labels ((((((((((IN THE SAME ORDER)))))))))))))
        shuffled_lables = np.zeros(self.labels.shape, dtype = np.uint8)
        shuffled_lables[range(perm.size)] = self.labels[perm]

        #results
        self.data = shuffled_images
        self.labels = shuffled_lables

    def extract_tinyimagenet_from_JPEGs(self, data_type = "training", directory = "E:/Datasets/tiny-imagenet-200/tiny-imagenet-200/train"):
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

        pickle.dump(images, open(data_type+"_data.p", "wb"))
        pickle.dump(images, open(data_type+"_labels.p", "wb"))
