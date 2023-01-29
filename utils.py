#### general librairies
from __future__ import division, print_function, absolute_import
import os, sys, time, h5py, gc, csv, cv2 , itertools, multiprocessing,random, json
from itertools import islice, chain, repeat, zip_longest
import numpy as np
from collections import Counter
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from scipy.stats.mstats import pearsonr
from scipy.io import wavfile
#import opensmile
#import librosa, librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial import distance as dist
from sklearn.metrics.pairwise import euclidean_distances
import warnings
from statsmodels.tsa.stattools import adfuller
import torchaudio
from os import listdir
from os.path import isfile, join
from scipy.stats import wasserstein_distance as dst
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.preprocess.preprocessors import BasePreprocessor, AudioToSpectrogramPreprocessor, CnnPreprocessor
from video_dataset import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
sys.setrecursionlimit(10000)


warnings.filterwarnings("ignore")

#### torch modules
import torch
import torchvision
#torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets
from torch.utils.data import IterableDataset, DataLoader, TensorDataset, Dataset
from torch.utils.checkpoint import checkpoint_sequential
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
##### tensorflow/keras modules
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, VGG19
#from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg19 import preprocess_input

#from tensorflow.keras.applications import ResNet50
#from tensorflow.python.client import device_lib
#import pretrainedmodels
##import pretrainedmodels.utils as utils
import keract
from keract import get_activations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras import layers
#### sickit-learn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity

##### files to include
from timesformer.models.vit import TimeSformer
from models_to_train.STAM import STAM
from models_to_train.video_swin_transformer import SwinTransformer3D
#from models_to_train.vivit import ViViT
from models_to_train.train_video_features_att import *
from models_to_train.train_model import *
from models_to_train.vivit_keras import *
#from models_to_pretrain.C3D import *
#from models.extract_c3d_feat_tf import *
from models_to_pretrain.c3d import C3D
#from PyTorch_ViT.pytorch_pretrained_vit import ViT
from swin_transformer import *
from vit_pytorch import ViT
from ast_models import *
from liris_data_helper import *
from eev_data_helper import *
from experiments_manager import *
from data_manager import *





