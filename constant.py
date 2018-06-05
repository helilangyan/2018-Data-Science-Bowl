# ====== logging ==========
import logging
logging.basicConfig(filename='./net_without_sigmoid.log', level=logging.INFO)
logger = logging.getLogger(name="data_bow")

# =========path==============
data_root = 'E:\\data\\kaggle\\2018_Data_Science_Bowl'
test_dir = data_root + '\\' + 'stage1_test'
train_dir = data_root + '\\' + 'stage1_train'
trian_label_filepath = data_root + '\\' + 'stage1_train_labels.csv'

# ========shape===============

target_shape = (128,128,1)
cell_shape_dict = {(520, 696, 4): 92, (1040, 1388, 4): 1, (360, 360, 4): 91, (260, 347, 4): 5, (256, 256, 4): 334, (603, 1272, 4): 6, (1024, 1024, 4): 16, (512, 640, 4): 13, (256, 320, 4): 112}
mask_cell_dict = {(512, 640): 411, (256, 256): 9538, (603, 1272): 1377, (1024, 1024): 1345, (520, 696): 9552, (256, 320): 4682, (260, 347): 408, (360, 360): 2134, (1040, 1388): 14}

# =========paramter=============
leaky_alpha = 0.1# leaky relu 的 参数（斜率）
split_ratio = 0.8

# ========== constant =============
pix_max = 255