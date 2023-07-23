import numpy as np
TIMESTEP_IN = 12
TIMESTEP_OUT = 12
CHANNEL = 1
BATCHSIZE = 64
LEARN = 0.001
EPOCH = 200
PATIENCE = 10
OPTIMIZER = 'Adam'
LOSS = 'MAE'
ADJTYPE = 'doubletransition'
CALSPLIT = 0.5
# calibration error quantile rate

cal_list = np.arange(0,21)/21

# for adaptive
lambda_list = np.arange(0,41)/41

