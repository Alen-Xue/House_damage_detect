# House_damage_detect

the datasets download address is: https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage

Just use your account of Kaggle to download

This code use Pytorch program library to complete this project, which include:
--
import torch

from torchvision import transforms, datasets, models

import torch.nn as nn

import numpy as np

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset, DataLoader


Prepare for train and val datasets
--

In order to make the model more generalized, we need to preprocess the image, which includes random cropping, horizontal flipping, and rotation.

Use the ResNet50 as the model
--
We use the pretained model, and take careful for freeze the model weight.

Futhermore, the target of this project is to detect if 'House' in the picture is damaged, it is essentially a binary classfication model. Thus we should modify the number of feature in "fn" layer 2048->512->2(dim = 1), and to avoid overfitting set drop node rate is 20%.

Use the Grad-CAM to create damage heatmap
--
![image](https://github.com/Alen-Xue/House_damage_detect/assets/126217366/9d84d4ca-2f72-41fe-8987-e9c2ca44b72f)




