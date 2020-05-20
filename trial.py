import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
#
# model = models.resnext101_32x8d(pretrained = True)
# print('model: ',model )
# #model = list(model.children())[:-1]
# #model.append(nn.Linear(512, 2))
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)
# model = list(model.children())[:-2]
# #model.append(nn.Linear(512, 2))
# mymodel = nn.Sequential(*model)
# print(mymodel)

mse = np.zeros(4)
for i in range(4):
    mse[i] = (i / 4)

print(mse)

