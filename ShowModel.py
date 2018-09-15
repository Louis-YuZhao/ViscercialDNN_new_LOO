# -*- coding: utf-8 -*-
from keras.utils import plot_model
from UnetTrainTwoLoss import get_unet_short
model = get_unet_short()
plot_model(model, to_file='model.png')