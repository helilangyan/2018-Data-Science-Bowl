# dicts = {
#   (1,2):1,
#   (1,3):2,
# }
# print(dicts)
# import os
# masks_dir = "E:\\data\\kaggle\\2018_Data_Science_Bowl\\stage1_train\\0b0d577159f0d6c266f360f7b8dfde46e16fa665138bf577ec3c6f9c70c0cd1e\\masks"
# print(os.listdir(masks_dir))

from skimage.morphology import label
from utils import batch_data_generator
from skimage.io import imsave,imread
from collections import Counter


def test_set_equal():
  print({1,0}=={1.,0})

import numpy as np
def test_reduce(start=3,end=6,size=10):
  x1 = np.ones((size,size))
  x2 = np.ones((size,size))
  x2[:start,:start] = 0
  x3 = np.ones((size,size))
  x3[end:,end:] = 0
  x = [x1,x2,x3]
  return np.multiply.reduce(x)

def test_numpy_log():
  print(np.log2(256))

def test_in():
  print(8.0 in [8,9,10,11])

def test_skimage_label(mask=None):
  mask = test_reduce()
  labeled_mask= label(mask,connectivity=2,background=-1)
  # 连通图的数量
  num_graph = np.max(labeled_mask.flatten())
  single_masks = []
  for i in range(1,num_graph+1):
    indexes = labeled_mask == i
    print(mask[indexes].flatten()[0])
    if mask[indexes].flatten()[0] == 0:
      single_mask = np.ones_like(mask)
      single_mask[indexes] = 0
      single_masks.append(single_mask)
      assert single_mask.shape == mask.shape
  single_masks = np.array(single_masks)
  assert single_masks.shape == (num_graph - 1,mask.shape[0],mask.shape[1])
  return single_masks

def test_isinstance_array():
  arr = np.array([[1,2],[3,4]])
  assert isinstance(arr,np.ndarray)
  assert isinstance(arr == 1,np.ndarray)
  print(arr == 1)


# 和python 的append 完全不一样
def test_numpy_append():
  arr = np.array([])
  arr1 = np.array([[1,2],[3,4]])
  arr2 = np.array([[5,6],[7,8]])
  arr = np.append(arr,arr1)
  arr = np.append(arr,arr2)
  return arr

def test_array_split():
  arr1 = np.array(range(10))
  return np.split(arr1,4)

def num_generator():
  return ( num for num in range(10))

def test_generator():
  gen = num_generator()
  for num in gen:
    print(num)

def test_size():
  arr1 = np.array([[1,2],[3,4]])
  return np.size(arr1)

# print(test_size())


def test_batch():
  gen = batch_data_generator(batch_size=3)
  item = next(gen)
  print(len(item[0][0]))

def test_save():
  image = np.ones(shape=(256,256))
  image[:64,:64] = 0
  imsave('test.png',image,)

# test_save()

def test_imread():
  image = imread('test.png',as_grey=True)
  print(image.shape)
  return dict(Counter(image.flatten()))

# print(test_imread()

def test_abs():
  arr = np.array([1,-2,3,-4])
  return np.sum(np.abs(arr)) / len(arr)

# print(test_abs())

def test_while():
  x = 1
  string = str(x) + 'file'
  while x < 5:
    x +=1
    print(string)
  print(string)  
test_while()