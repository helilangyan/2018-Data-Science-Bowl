import numpy as np
from constant import logger
import tensorflow as tf

from keras.models import load_model
from model import resnet_encoder as net
from utils import batch_data_generator
from utils import get_test_data
from utils import test_error
from utils import now_str
from constant import logger

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


model_ext = '.h5'


def train():
    # ======= paramter ==============
    use_local_net = False
    epoch = 20
    num_epoches = 40
    batch_size = 32

    # =======net ================
    model_path = 'model_without_sigmoid\\' + \
        'model' + '_' + str(epoch - 1) + model_ext
    if use_local_net:
        model = load_model(model_path)
        logger.info('use local model from %d epoch', epoch)
    else:
        model = net()
    logger.info('model inited')
    test_cells, test_masks = get_test_data()

    # ===========train =============
    while epoch < num_epoches:
        logger.info('%s epoch %d started', now_str(), epoch)
        generator = batch_data_generator(
            batch_size=batch_size, use_shuffle=True)
        batch_idx = 0
        for cell_mask in generator:
            cell_list = np.array([item[0] for item in cell_mask])
            mask_list = np.array([item[1] for item in cell_mask])
            model.train_on_batch(cell_list, mask_list)
            batch_idx += 1
            if batch_idx % 20:
                logger.info('%s epoch %d batch_index %d',
                            now_str(), epoch, batch_idx)
        batch_idx = 0

        # ===========test==================
        predict_masks = model.predict(test_cells, batch_size=64)
        error = test_error(test_masks, predict_masks)
        logger.info('predict error %f in epoch %d', error, epoch)
        model.save('model_without_sigmoid\\' + 'model' + '_' + str(epoch) + model_ext)
        logger.info('%s epoch %d model saved', now_str(), epoch)
        epoch += 1


train()
