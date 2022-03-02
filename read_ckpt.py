import pickle as pkl
import numpy as np
import torch
with open('./results/lanegcn/5.000.ckpt','rb') as f:
    file = torch.load(f)
print(file)
import ipdb;ipdb.set_trace()

