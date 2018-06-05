import os

from constant import logger
from constant import test_dir
from constant import target_shape
from utils import get_cell_image
from utils import get_all_image_ids
from utils import image2gray
from utils import split_mask

from skimage.transform import resize
import numpy as np
import 

assert len(target_shape) = 3

def get_predict_cells():
  ids = get_all_image_ids(test_dir)
  cells = [get_cell_image(image_id,train_dir=test_dir) for image_id in ids]
  cells = [image2gray(cell) for cell in cells]
  cells = [ resize(cell,output_shape=target_shape) for cell in cells]
  cells = np.array(cells)
  assert cells.shape == ((len(ids)),target_shape[0],target_shape[1],target_shape[2])
  return cells


def to_origin_shape(image,image_id,directory=test_dir):
  origin_image = get_cell_image(image_id,train_dir=directory)
  origin_shape = origin_image.shape
  converted_image = resize(origin_image,output_shape=origin_shape)
  return converted_image

def save_masks(masks,image_id,directory=test_dir):
  path = directory + '\\' + image_id + '\\' + 'mask'
  if not os.path.exists(path):
    os.makedirs(path)
  lens = len(masks)
  for i in range(lens):
