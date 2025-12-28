# @title Autoload all modules

import matplotlib
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import importlib
import os
import functools
import itertools
import torch

from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint

sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling2
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling2 import (ReverseDiffusionPredictor,
                       LangevinCorrector,
                       EulerMaruyamaPredictor,
                       AncestralSamplingPredictor,
                       NoneCorrector,
                       NonePredictor,
                       AnnealedLangevinDynamics)
import datasets_wavelet
import datasets_spatial
import os.path as osp
import time
import scipy.io as sio
import imageio.v2 as imageio
import cv2
from packages.ffdnet.models import FFDNet
import scipy.io as sio
from packages.ffdnet.test_ffdnet_ipol import ffdnet_vdenoiser
from tvdenoise import tvdenoise

net = FFDNet(num_input_channels=1).cuda()
model_fn = 'packages/ffdnet/models/net_gray.pth'
state_dict = torch.load(model_fn)
net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
net.load_state_dict(state_dict)

##########################################################################################################
sde = 'VESDE'  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
    from configs.ve import SIAT_kdata_spatial_ncsnpp as configs_spatial, \
        SIAT_kdata_wavelet_ncsnpp as configs_wavelet

    model_num = 'checkpoint.pth'
    ckpt_filename = '/home/e/桌面/TCWD/Iterative_Reconstruction/exp_wavelet/checkpoints/checkpoint_40.pth'
    config = configs_wavelet.get_config()
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                N=config.model.num_scales)  ###################################  sde
    sampling_eps = 1e-5

batch_size = 1  # @param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0  # @param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets_wavelet.get_data_scaler(config)
inverse_scaler = datasets_wavelet.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

# @title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)

##########################################################################################################
sde_space = 'VESDE'  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde_space.lower() == 'vesde':
    model_num = 'checkpoint.pth'
    ckpt_filename_space = '/home/e/桌面/TCWD/Iterative_Reconstruction/exp_spatial/checkpoints/checkpoint_60.pth'
    config_space = configs_spatial.get_config()
    sde_space = VESDE(sigma_min=config_space.model.sigma_min, sigma_max=config_space.model.sigma_max,
                      N=config_space.model.num_scales)  ###################################  sde
    sampling_eps = 1e-5

batch_size = 1  # @param {"type":"integer"}
config_space.training.batch_size = batch_size
config_space.eval.batch_size = batch_size

sigmas = mutils.get_sigmas(config_space)
scaler = datasets_spatial.get_data_scaler(config_space)
inverse_scaler = datasets_spatial.get_data_inverse_scaler(config_space)
score_model_space = mutils.create_model(config_space)

optimizer = get_optimizer(config_space, score_model_space.parameters())
ema = ExponentialMovingAverage(score_model_space.parameters(),
                               decay=config_space.model.ema_rate)
state_space = dict(step=0, optimizer=optimizer, model=score_model_space, ema=ema)

state_space = restore_checkpoint(ckpt_filename_space, state_space, config_space.device)
ema.copy_to(score_model_space.parameters())

img_size = config_space.data.image_size
channels = config_space.data.num_channels
shape_space = (batch_size, channels, img_size, img_size)
##########################################################################################################
predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector  # @param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}

snr = 0.075  # 0.16 #@param {"type": "number"}
n_steps = 1  # @param {"type": "integer"}
probability_flow = False  # @param {"type": "boolean"}

trial = 1
# percent_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
percent_list = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006]
list_img = []

for i in range(20):
    img_test = imageio.imread(
        r'/home/e/桌面/TCWD/Iterative_Reconstruction/measurement/result_{}.png'.format(i))
    list_img.append(img_test)

list_weight = [[0.5, 0.8]]

for iii in range(len(list_weight)):
    weight_a = list_weight[iii][0]
    weight_b = list_weight[iii][1]
    for jjj in range(len(list_img)):
        img_input = list_img[jjj]
        for k in range(trial):
            steps = 600
            for percent in percent_list:
                start = time.time()
                sampling_fn = sampling2.get_pc_sampler(sde, sde_space, shape, shape_space, predictor, corrector,
                                                       inverse_scaler, snr, n_steps=n_steps,
                                                       probability_flow=probability_flow,
                                                       continuous=config.training.continuous,
                                                       eps=sampling_eps,
                                                       device=config.device,
                                                       )
                Reconstruct_Field, a, b, c = sampling_fn(score_model, score_model_space,
                                                         np.fft.ifftshift(img_input), 40, -4, 0,
                                                         steps,
                                                         np.random.rand(*img_input.shape) + 1j * np.random.rand(
                                                             *img_input.shape),
                                                         percent, net, weight_a, weight_b)
                Reconstruct_Image_Inten = np.power(np.abs(Reconstruct_Field), 2)
                Reconstruct_Image_Amp = np.abs(Reconstruct_Field)
                end = time.time()
                print("Saving_{}_time:{}s".format(k, end - start))
                max = np.max(Reconstruct_Image_Amp)
                min = np.min(Reconstruct_Image_Amp)
                Reconstruct_Image_Amp = (Reconstruct_Image_Amp - min) / (max - min) * 255
                cv2.imwrite(
                    r'{}/{}_{}.png'.format(
                        '/home/e/桌面/TCWD/Iterative_Reconstruction/test_20', jjj,
                        percent),Reconstruct_Image_Amp)