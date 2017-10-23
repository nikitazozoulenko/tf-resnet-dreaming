from data_loader import *
from PIL import Image

data_loader = data_loader()
data_loader.load_data_arrays()
data_loader.data = np.vstack((data_loader.data, np.flip(data_loader.data, axis = 2)))
data_loader.labels = np.vstack((data_loader.labels, data_loader.labels))

#new random permutation
perm = np.arange(200000)
np.random.shuffle(perm)

#shuffle images
shuffled_images = np.zeros(data_loader.data.shape)
shuffled_images[range(perm.size)] = data_loader.data[perm]

#shuffle labels ((((((((((IN THE SAME ORDER)))))))))))))
shuffled_lables = np.zeros(data_loader.labels.shape, dtype = np.uint8)
shuffled_lables[range(perm.size)] = data_loader.labels[perm]

pickle.dump(shuffled_images, open("shuffled_training_data.p", "wb"))
pickle.dump(shuffled_lables, open("shuffled_training_labels.p", "wb"))

# image = Image.fromarray(shuffled_images[560])
# image2 = Image.fromarray(shuffled_images[720])
# print(shuffled_lables[560])
# print(shuffled_lables[720])
# image.show()
# image2.show()

print("done")

#self.data = pickle.load(open("shuffled_training_data.p", "rb"))
#self.labels = pickle.load(open("shuffled_training_labels.p", "rb"))
