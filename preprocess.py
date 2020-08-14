import os
import pickle
import numpy as np

data_dir = "/data/unagi0/ktokitake"
birds_dir = os.path.join(data_dir, "birds_attngan")
xlsa_dir = os.path.join(data_dir, "xlsa17")

images={}

images_dir= birds_dir+"/CUB_200_2011/images"
for dir in os.listdir(images_dir):
    images[dir] = os.listdir(images_dir+"/%s" % dir)


