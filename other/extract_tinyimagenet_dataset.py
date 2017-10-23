from data_loader import *

data_loader = data_loader()
data_loader.extract_tinyimagenet_from_JPEGs()
print(data_loader.str_to_class_lookup)
print(len(data_loader.str_to_class_lookup))
