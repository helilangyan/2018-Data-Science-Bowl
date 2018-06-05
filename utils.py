import time
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import itertools
from glob import glob
import os
from skimage.io import imread, imread_collection
from skimage.morphology import label
from skimage.color import rgb2gray
from skimage.transform import resize
from constant import train_dir, logger
from constant import split_ratio
from constant import target_shape
from skimage.color import rgb2gray
from skimage.io import imshow, imsave

# 获取一张 cell 图片


def get_cell_image(image_id, train_dir=train_dir):
    image_path = train_dir + '\\' + image_id + '\\images\\' + image_id + '.png'
    image = imread(image_path)
    if len(image.shape) == 2:
        h, w = image.shape
        logger.debug('read cell image shape: ( %d , %d),id: %s',
                     h, w, image_id)
    if len(image.shape) == 3:
        h, w, c = image.shape
        logger.debug(
            'read cell image shape: ( %d , %d, %d),id: %s', h, w, c, image_id)
    if len(image.shape) == 4:
        h, w, c, alpha = image.shape
        logger.debug(
            'read cell image shape: ( %d , %d, %d, %d),id: %s', h, w, c, alpha, image_id)
    return image

# 数组归一化


def array_union(array, max=255):
    return array / 255

# 获取一个cell 的所有mask图片，归一化之后的


def get_cell_mask(image_id, train_dir=train_dir):
    masks_dir = train_dir + '\\' + image_id + '\\masks'
    masks_pathes = [masks_dir + '\\' +
                    file_name for file_name in os.listdir(masks_dir)]
    lens = len(masks_pathes)
    masks = np.array(imread_collection(masks_pathes))
    assert len(masks.shape) == 3
    masks = array_union(masks)
    logger.debug('read %d mask images for cell %s ', lens, image_id)
    return masks

# 返回所有样本的id


def get_all_image_ids(train_dir=train_dir):
    return os.listdir(train_dir)

# 图片转变会灰度图


def image2gray(image):
    if len(image.shape) == 3 and image.shape[2] != 4:
        image = rgb2gray(image)
        assert len(image.shape) == 2
    elif len(image.shape) == 3 and image.shape[2] != 3:
        image = rgb2gray(image)
        assert len(image.shape) == 2
    return image


def add1dimosionTogray(gray_image):
    if not isinstance(gray_image, np.ndarray):
        gray_image = np.array(gray_image)
    if len(gray_image.shape) == 3 and gray_image.shape[2] == 1:
        return gray_image
    elif len(gray_image.shape) == 2:
        gray_image = gray_image[:, :, np.newaxis]
        return gray_image

# 获取一张黑白cell和对应的mask_combined


def get_cell_and_mask_by_id(image_id):
    # 处理cell
    cell = get_cell_image(image_id)
    cell = image2gray(cell)
    pix_max_value = np.max(cell.flatten())
    cell = cell / pix_max_value
    cell = add1dimosionTogray(cell)
    cell = resize(cell, output_shape=target_shape)
    cell = normalize_image(cell)
    # 处理mask
    masks = get_cell_mask(image_id)
    mask = mask_combine(masks)
    mask = add1dimosionTogray(mask)
    mask = resize(mask, output_shape=target_shape)
    mask = normalize_image(mask)
    assert cell.shape == mask.shape
    return (cell, mask)

# 获取一批 cell 和mask 的组合


def get_cell_and_mask_by_ids(image_ids):
    return [get_cell_and_mask_by_id(image_id) for image_id in image_ids]

# 批次数据生成器
# 丢掉最后一部分小于 batch_size 的 数据


def batch_data_generator(batch_size, use_shuffle=False, split_ratio=split_ratio):
    assert split_ratio < 1
    image_ids = get_all_image_ids()
    if not isinstance(image_ids, np.ndarray):
        image_ids = np.array(image_ids)
    image_ids = image_ids[:int(len(image_ids) * split_ratio)]
    if use_shuffle:
        np.random.shuffle(image_ids)
    n_ids = len(image_ids)
    assert n_ids > batch_size
    image_ids = image_ids[:-(n_ids % batch_size)]
    n_ids = len(image_ids)
    assert n_ids % batch_size == 0
    split_ids = np.split(image_ids, int(n_ids/batch_size))
    return (get_cell_and_mask_by_ids(ids) for ids in split_ids)

# 获取测试数据


def get_test_data(split_ratio=split_ratio):
    image_ids = get_all_image_ids()
    assert split_ratio < 1
    test_image_ids = image_ids[int(len(image_ids) * split_ratio):]
    cell_and_masks = get_cell_and_mask_by_ids(test_image_ids)
    cells = np.array([item[0] for item in cell_and_masks])
    masks = np.array([item[1] for item in cell_and_masks])
    return cells, masks

# 获取所有的mask 图片


def get_all_cell_mask(train_dir=train_dir):
    train_image_ids = os.listdir(train_dir)
    cell_mask_dict = {image_id: get_cell_mask(
        image_id, train_dir=train_dir) for image_id in train_image_ids}
    return cell_mask_dict

# 获取所有的训练 cell 图片


def get_all_train_image(train_dir=train_dir):
    train_image_ids = os.listdir(train_dir)
    logger.info('read all train images, total: %d', len(train_image_ids))
    train_image_pathes = [train_dir + '\\' + image_id +
                          '\\images\\' + image_id + '.png' for image_id in train_image_ids]
    return array_union(np.array(imread_collection(train_image_pathes)))

# 切分mask 图片


def split_mask(mask):
    labeled_mask = label(mask, connectivity=2, background=-1)
    # 连通图的数量
    num_graph = np.max(labeled_mask.flatten())
    # 掩码列表
    single_masks = []
    for i in range(1, num_graph+1):
        indexes = labeled_mask == i
        # 如果连通图元素是1
        if mask[indexes].flatten()[0] == 1:
            single_mask = np.ones_like(mask)
            single_mask[indexes] = 0
            single_masks.append(single_mask)
            assert single_mask.shape == mask.shape
    single_masks = np.array(single_masks)
    # 连通图中有一个图元素都是0，是背景
    assert single_masks.shape == (num_graph - 1, mask.shape[0], mask.shape[1])
    return single_masks



# 列表里的值变成0或者1 (归一化)
def array_to_0_1(array, boundary=0.5):
    assert isinstance(array, np.ndarray)
    new_array = np.ones_like(array)
    new_array[array <= boundary] = 0
    return new_array


def test_error(real_masks, predict_masks):
    if not isinstance(real_masks, np.ndarray):
        real_masks = np.array(real_masks)
    if not isinstance(predict_masks, np.ndarray):
        predict_masks = np.array(predict_masks)
    assert real_masks.shape == predict_masks.shape
    total_size = np.size(real_masks)
    error = np.sum(np.abs(real_masks - predict_masks)) / total_size
    return error


def reshape(image, target_shape=target_shape, gray=True):
    if gray:
        assert len(image.shape) >= 3
        image = rgb2gray(image)
    resized_image = resize(image, output_shape=target_shape)
    return resized_image


def same_shape(images):
    shape_set = set([image.shape for image in images])
    if len(shape_set) != 1:
        logger.error('images has %d different shape', len(shape_set))
        return False
    else:
        return True


def array_0_1(array):
    flatten_array = array.flatten()
    array_set = set(flatten_array)
    if array_set == {0., 1.}:
        # {1,0}=={1.,0} True
        return True
    else:
        return False


def mask_combine(masks):
    assert same_shape(masks)
    assert array_0_1(masks)
    return np.add.reduce(masks)


# 指定字符串显示当前时间
def now_str(format="%Y-%m-%d %X"):
    return time.strftime(format, time.localtime())


def normalize_image(x):
        # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def denormalize_image(x):
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) * 255
    return x_norm
