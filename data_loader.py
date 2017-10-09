import os
import numpy as np
import pickle
from PIL import Image
class data_loader(object):
    def __init__(self):
        self.create_lookup_tables()
        self.counter = 0

    def load_data_arrays(self):
        self.data = pickle.load(open("training_data.p", "rb"))
        index = []
        for i in range(200):
            for j in range(500):
                index.append(i)
        self.labels = np.zeros((100000, 200), dtype = np.int64)
        self.labels[range(100000), index] = 1

    def create_lookup_tables(self):
        self.str_to_class_lookup = {}
        self.class_to_str_lookup = {}
        list_str = os.listdir("E:/Datasets/tiny-imagenet-200/tiny-imagenet-200/train")
        ctr = 0
        for string in list_str:
            self.str_to_class_lookup[string] = ctr
            self.class_to_str_lookup[ctr] = string
            ctr += 1

    def next_batch(self, batch_size):
        i = self.counter
        data = self.data[i * batch_size : (i+1) * batch_size]
        labels = self.labels[i * batch_size : (i+1) * batch_size]

        self.counter += 1
        return [data, labels]

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
        self.counter = 0

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
        pickle.dump(images, open(data_type+"_data.p", "wb"))
        pickle.dump(labels, open(data_type+"_labels.p", "wb"))

        print("image_counter", image_counter)
        print("class_counter", class_counter)
        print(images.shape)
        print(labels.shape)
