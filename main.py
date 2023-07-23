import sys
import os
import shutil

import torch
from datetime import datetime
import time
from lib.Utils import load_data, inverse_transform, transform
from src.GraphWaveNet import *
from Param import *

torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAVAL)
    CAL_NUM = int(data.shape[0] * TRAVALCAL)
    XS, YS = [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)

    elif mode == 'CAL':
        for i in range(TRAIN_NUM - TIMESTEP_IN, CAL_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1 ):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)

    elif mode == 'TEST':
        for i in range(CAL_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def getModel(name):
    # way1: only adaptive graph.
    # model = gwnet(device, num_nodes = N_NODE, in_dim=CHANNEL).to(device)
    # return model

    # way2: adjacent graph + adaptive graph

    adj_mx = load_adj(ADJPATH, ADJTYPE, DATANAME)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    model = gwnet(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=supports, dropout=0.0).to(device)
    return model

def evaluateModel(model, data_iter,quantiles_list):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            #y_pred = model(x)
            y_pred = model(x)
            l = 0.0
            for i in range(len(quantiles_list)):
                y_pred_ = y_pred[..., i]
                loss_ = quantile_loss(y_pred_, y, quantiles_list[i])
                l += loss_
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def quantile_loss( y_pred, y, q):
    error = (y_pred-y)
    single_loss = torch.max(q*error, (q-1)*error)
    loss = torch.mean(single_loss)
    return loss

def trainModel(name, mode, XS, YS, quantiles_list, scaler):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)

    min_val_loss = np.inf
    wait = 0

    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)

    for epoch in range(EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = 0.0
            for i in range(len(quantiles_list)):
                y_pred_ = y_pred[..., i]
                loss_ = quantile_loss(y_pred_, y, quantiles_list[i])
                loss += loss_
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, val_iter, quantiles_list)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", train_loss, "validation loss:",
              val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % (
                "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:",
                val_loss))

    torch_score = evaluateModel(model, train_iter, quantiles_list)
    YS_pred_ = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    YS = inverse_transform(YS, scaler['mean'], scaler['std'])

    for i in range(len(quantiles_list)):
        YS_pred = inverse_transform(YS_pred_[..., i], scaler['mean'], scaler['std'])
        MSE, RMSE, MAE, MAPE = evaluate(YS, YS_pred)

        with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
            f.write("%s, %s, quantile, MSE, RMSE, MAE, MAPE, %.2f, %.10f, %.10f, %.10f, %.10f\n" % (
                name, mode, quantiles_list[i], MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e\n" % (name, mode, torch_score))
    print('Model Training Ended ...', time.ctime())

def calModel(name, mode, XS, YS,  scaler, lambda_list, exp_q, n_grid):
    '''
    this version is to compute our methods
    '''
    print('MODEL CALIBRATION STATE:')
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    cal_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    cal_iter = torch.utils.data.DataLoader(cal_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS_pred = predictModel(model, cal_iter)
    YS_pred_0 = YS_pred[..., 0]
    YS_pred_1 = YS_pred[..., 1]
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred_0.shape)
    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(YS_pred_0), np.squeeze(YS_pred_1)
    YS, YS_pred_0, YS_pred_1 = inverse_transform(YS, scaler['mean'], scaler['std']), \
                               inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                               inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])

    YS_pred_0 = torch.Tensor(YS_pred_0).to(device)
    YS_pred_1 = torch.Tensor(YS_pred_1).to(device)
    YS = torch.Tensor(YS).to(device)

     # split some data
    split = int(YS.shape[0]* CALSPLIT)

    YS_train = YS[:split]
    YS_0_train = YS_pred_0[:split]
    YS_1_train = YS_pred_1[:split]

    YS_val = YS[split:]
    YS_0_val = YS_pred_0[split:]
    YS_1_val = YS_pred_1[split:]

    YS_1_train = YS_1_train * (YS_1_train>0)
    error_low =  YS_train - YS_1_train
    error_high =  YS_0_train - YS_train

    err_dis = torch.cat([error_low,error_high])
    error_quantile = n_grid

    node = YS.size(2)
    seq = YS.size(1)

    # return [error_q, seq, node]
    corr_err_list = []
    for q in range(0, error_quantile+1):
        q_n = []
        for n in range(node):
            q_t = []
            for t in range(seq):
                corr_err = torch.quantile(err_dis[:, t, n], q/error_quantile)
                q_t.append(corr_err)
            q_t = torch.stack(q_t)
            q_n.append(q_t)
        q_n = torch.stack(q_n).T
        corr_err_list.append(q_n)
    corr_err_list = torch.stack(corr_err_list)

    coverage_list = []
    interval_list = []
    for m in range(0, error_quantile+1):
        y_u_pred = YS_pred_0 + corr_err_list[m]
        y_l_pred = YS_pred_1 - corr_err_list[m]
        y_l_pred = y_l_pred * (y_l_pred >=0)
        mask = YS > 0

        coverage = torch.logical_and(y_u_pred >= YS, y_l_pred <= YS)
        coverage_a = torch.mean(coverage.float(),axis=0)
        coverage_list.append(coverage_a)
        interval_a = torch.mean(torch.abs(y_u_pred - y_l_pred),axis=0)       
        interval_list.append(interval_a)

    # torch.Size([sampe, seq, node])
    coverage_list = torch.stack(coverage_list)
    interval_list = torch.stack(interval_list)


    # 归一化interval_list
    interval_nor = []
    for n in range(node):
        interval_n = []
        for t in range(seq):
            interval_ = (interval_list[:,t, n] - torch.min(interval_list[:,t, n])) /\
                        (torch.max(interval_list[:,t, n]) - torch.min(interval_list[:,t, n]))
            interval_n.append(interval_)
        interval_n = torch.stack(interval_n)
        interval_nor.append(interval_n)
    interval_nor = torch.stack(interval_nor).T

    coverage_nor = []
    for n in range(node):
        coverage_n = []
        for t in range(seq):
            coverage_ = (coverage_list[:,t, n] - torch.min(coverage_list[:,t, n])) /(
                    torch.max(coverage_list[:,t, n]) - torch.min(coverage_list[:,t, n]) + 1e-5)
            coverage_n.append(coverage_)
        coverage_n = torch.stack(coverage_n)
        coverage_nor.append(coverage_n)
    coverage_nor = torch.stack(coverage_nor).T 

    cor_err = []
    lambda_list = torch.Tensor(lambda_list).to(device)
    for i in lambda_list:
        loss = - i * coverage_nor + (1 - i) * interval_nor
        index = torch.argmin(loss,axis=0).cpu().numpy()
        err_t = []
        for t in range(seq):
            err_n = []
            for n in range(node):
                corr_err = corr_err_list[index[t,n], t, n]
                err_n.append(corr_err)
            err_n = torch.stack(err_n)
            err_t.append(err_n)
        err_i = torch.stack(err_t)
        cor_err.append(err_i)
    cor_err = torch.stack(cor_err)

    # 选择期望的cor_err
    independent_coverage_l =[]
    for i in range(len(lambda_list)):
        y_u_pred = YS_0_val + cor_err[i]
        y_l_pred = YS_1_val - cor_err[i]

        y_l_pred = y_l_pred * (y_l_pred>0)
        mask = YS_val>0
        independent_coverage = torch.logical_and(torch.logical_and(y_u_pred >= YS_val, 
        y_l_pred <= YS_val), mask)
        m_coverage = torch.sum(independent_coverage)/torch.sum(mask)
        independent_coverage_l.append(m_coverage)
    m_coverage = torch.stack(independent_coverage_l)
    index = torch.argmin(torch.abs(m_coverage - exp_q))

    return [cor_err[index].cpu().numpy(), cor_err[index].cpu().numpy()]

def testunModel(name, mode, XS, YS, err, scaler):
    print('model uncertainty test')
    print('timestep_in, timestep_out', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    YS_pred = predictModel(model, test_iter)
    YS_pred_0 = YS_pred[...,0]
    YS_pred_1 = YS_pred[...,1]

    YS, YS_pred_0, YS_pred_1 = np.squeeze(YS), np.squeeze(YS_pred_0), \
                               np.squeeze(YS_pred_1)
    YS, YS_pred_0, YS_pred_1 = inverse_transform(YS, scaler['mean'], scaler['std']), \
                               inverse_transform(YS_pred_0, scaler['mean'], scaler['std']), \
                              inverse_transform(YS_pred_1, scaler['mean'], scaler['std'])

    y_u_pred = YS_pred_0 + err[0]
    y_l_pred = YS_pred_1 - err[1]

    y_l_pred = y_l_pred * (y_l_pred > 0)
    mask = YS > 0
    independent_coverage = np.logical_and(np.logical_and(y_u_pred >= YS, y_l_pred <= YS), YS > 0)

    # compute the coverage and interval width
    results = {}
    results["Upper limit"] = np.array(y_l_pred)
    results["Lower limit"] = np.array(y_u_pred)
    results["Confidence interval widths"] = np.abs(y_u_pred - y_l_pred) * mask
    results["Mean confidence interval widths"] = np.sum(results["Confidence interval widths"]) / \
                                                 np.sum(mask)
    results["Independent coverage indicators"] = independent_coverage
    results["Mean independent coverage"] = np.sum(independent_coverage.astype(float)) / np.sum(mask)
    results["Calbration error"] = np.mean(err)

    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("calibration error,  %.4f\n "
                % results["Calbration error"])
        f.write("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
                % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

    print('*' * 40)
    print("Calibration error, %.4f\n" % np.mean(err[0] + err[1]))
    print("Mean independent coverage, Mean confidence interval widths, %.4f, %.4f\n "
          % (results["Mean independent coverage"], results["Mean confidence interval widths"]))

################# Parameter Setting #######################
MODELNAME = 'GraphWaveNet'

import argparse
import configparser
parser = argparse.ArgumentParser()
parser.add_argument('--config',default='../configuration/PEMS08.conf',type = str, help = 'configuration file path')
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--uncer_m', type=str, default='adaptive') # quantile/quantile_conformal/adaptive/grid_search
parser.add_argument('--q', type=float, default=0.9) 
parser.add_argument('--TRAINRATIO', type=float, default=3.5) 
parser.add_argument('--n_grid', type=int, default=100) 
parser.add_argument('--GPU', type=str, default='0') 

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
config = configparser.ConfigParser()
config.read(args.config)
config_data = config['Data']
DATANAME = config_data['DATANAME']
FLOWPATH = config_data['FLOWPATH']
N_NODE = int(config_data['N_NODE'])
ADJPATH = config_data['ADJPATH']
UNCER_M = args.uncer_m
q = args.q
TRAINRATIO = args.TRAINRATIO # cal占8份的比例，1-6
TRAVAL = 0.8* (1-TRAINRATIO/8)
TRAVALCAL = 0.8 # Calibration train+val+cal 总共8份
TRAINVALSPLIT = 1/(8-TRAINRATIO)# val占总体一份, val 占总体比例

n_grid = args.n_grid

quantiles_list = [1-q, q]

KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + UNCER_M + '_' + str(q) + '_' + str(TRAINRATIO) + '_' + datetime.now().strftime(
    "%y%m%d%H%M")
print(KEYWORD)
PATH = '../save/' + KEYWORD

device = torch.device("cuda:{}".format(args.GPU)) if torch.cuda.is_available() else torch.device("cpu")

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('GraphWaveNet.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_GraphWaveNet.py', PATH)

    data = load_data(DATANAME, FLOWPATH)
    print('data.shape', data.shape)

    trainx, trainy = getXSYS(data, 'TRAIN')
    # transform
    mean = trainx.mean()
    std = trainy.std()
    scaler = {'mean': mean, 'std': std}
    data = transform(data, scaler['mean'], scaler['std'])

    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS, quantiles_list, scaler)
    print('training ended')

    print(KEYWORD, 'cal started', time.ctime())
    calXS, calYS = getXSYS(data, 'CAL')
    print('CAl XS.shape YS,shape', calXS.shape, calYS.shape)
    err = calModel(MODELNAME, 'cal', calXS, calYS, scaler, lambda_list, q, n_grid)
    print(KEYWORD, 'cal ended', time.ctime())

    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testunModel(MODELNAME, 'test', testXS, testYS, err, scaler)
    
    print(KEYWORD, 'testing ended', time.ctime())

if __name__ == '__main__':
    main()

