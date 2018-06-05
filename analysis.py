from constant import train_dir
import os
from collections import Counter
from utils import get_all_cell_mask
from utils import get_all_train_image
from matplotlib import pyplot as plt

# {(520, 696, 4): 92, (1040, 1388, 4): 1, (360, 360, 4): 91, (260, 347, 4): 5, (256, 256, 4): 334, (603, 1272, 4): 6, (1024, 1024, 4): 16, (512, 640, 4): 13, (256, 320, 4): 112}
def cell_shape_dict():
  train_images = get_all_train_image()
  image_shapes = [image.shape for image in train_images]
  shape_dict = dict(Counter(image_shapes))
  return shape_dict

# {(512, 640): 411, (256, 256): 9538, (603, 1272): 1377, (1024, 1024): 1345, (520, 696): 9552, (256, 320): 4682, (260, 347): 408, (360, 360): 2134, (1040, 1388): 14}
def cell_mask_shape_dict():
  mask_dict = get_all_cell_mask()
  mask_shapes = [mask.shape for masks in mask_dict.values() for mask in masks]
  cell_mask_shape_dict = dict(Counter(mask_shapes))
  return cell_mask_shape_dict

def cell_mask_num_dict():
  train_image_ids = os.listdir(train_dir)
  mask_nums = [len(os.listdir(train_dir + '\\' + image_id + '\\masks')) for image_id in train_image_ids]
  mask_num_dict = dict(Counter(mask_nums))
  return mask_num_dict

def show_dict(dicts):
  x = list(dicts.keys())
  y = list(dicts.values())
  plt.plot(x,y)
  plt.show()

def show_cell_mask_num():
  mask_num_dict = cell_mask_num_dict()
  show_dict(mask_num_dict)