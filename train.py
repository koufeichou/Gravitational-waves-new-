import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# data = torch.randn(1, 1, 3, 4096)
data_dir='./data/'
test_data_dir='./testdata/'

def get_params_wave(batch_size, step):          #（50， 0）
    params_dir = 'wave_bns_%s.txt' % (step * batch_size + 1)
    params = np.loadtxt(data_dir + params_dir)
    params = params.transpose()
    params = params[np.newaxis, :]
    params = params[np.newaxis, :]
    i = 1
    while i < batch_size:
        params_dir_append = 'wave_bns_%s.txt' % (step * batch_size + i + 1)
        params_append = np.loadtxt(data_dir + params_dir_append)
        params_append = params_append.transpose()
        params_append = params_append[np.newaxis, :]
        params_append = params_append[np.newaxis, :]
        params = np.concatenate((params, params_append), axis=0)

        i = i + 1

    return params

# def get_params_wave_test(batch_size, step):          #（50， 0）
#     params_dir = 'wave_bns_%s.txt' % (step * batch_size + 1)
#     params = np.loadtxt(test_data_dir + params_dir)
#     params = params.transpose()
#     params = params[np.newaxis, :]
#     params = params[np.newaxis, :]
#     i = 1
#     while i < batch_size:
#         params_dir_append = 'wave_bns_%s.txt' % (step * batch_size + i + 1)
#         params_append = np.loadtxt(test_data_dir + params_dir_append)
#         params_append = params_append.transpose()
#         params_append = params_append[np.newaxis, :]
#         params_append = params_append[np.newaxis, :]
#         params = np.concatenate((params, params_append), axis=0)
#
#         i = i + 1
#
#     return params

def get_inj_time(batch_size, step):
    params_dir = 'inj_time_%s.txt' % (step * batch_size + 1)
    params = np.loadtxt(data_dir + params_dir)
    params = params.reshape(1,1)
    i = 1
    while i < batch_size:
        params_dir_append = 'inj_time_%s.txt' % (step * batch_size + i + 1)
        params_append = np.loadtxt(data_dir + params_dir_append)
        params_append = params_append.reshape(1,1)
        params = np.concatenate((params, params_append), axis=0)
        i = i + 1

    return params

# def get_inj_time_test(batch_size, step):
#     params_dir = 'inj_time_%s.txt' % (step * batch_size + 1)
#     params = np.loadtxt(test_data_dir + params_dir)
#     params = params.reshape(1,1)
#     i = 1
#     while i < batch_size:
#         params_dir_append = 'inj_time_%s.txt' % (step * batch_size + i + 1)
#         params_append = np.loadtxt(test_data_dir + params_dir_append)
#         params_append = params_append.reshape(1,1)
#         params = np.concatenate((params, params_append), axis=0)
#         i = i + 1
#
#     return params

# get_params_wave(5,0)
# get_inj_time(5,0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,16), stride=(3,1), padding=(0,0), bias=False)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1,16), stride=(1,1), padding=(0,0), bias=False)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1,16), stride=(1,1), padding=(0,0), bias=False)
        self.fc1 = nn.Linear(1888, 64)
        self.fc2 = nn.Linear(64, 1)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3,16), stride=(3,1)),
            nn.MaxPool2d((1,4), (1,4)),
            nn.ReLU(False),
            nn.Conv2d(8, 10, kernel_size=(1,16), stride=(1,1)),
            nn.MaxPool2d((1,4), (1,4)),
            nn.ReLU(False),
            nn.Conv2d(10, 12, kernel_size=(1, 16), stride=(1, 1)),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.ReLU(False)
        )


        self.fc_loc = nn.Sequential(
            nn.Linear(12 * 1 * 59, 64),
            nn.ReLU(False),
            nn.Linear(64, 2),
            nn.ReLU(False)
        )

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 12 * 1 * 59)
        st = self.fc_loc(xs)

        if(st[0][0] < st[0][1]):
            wave_start = st[0][0]
            wave_end = st[0][1]
        else:
            wave_start = st[0][1]
            wave_end = st[0][0]

        a = wave_start.round().int()
        b = wave_end.round().int()

        x[:,:,:, 0:a-1] = 0
        x[:,:,:, b+1:4096] = 0

        # c = (wave_start - a) * x[:,:,:,a+1] + (1 - wave_start + a) * x[:,:,:,a]
        # d = (wave_end - b) * x[:,:,:,b+1] + (1 - wave_end + b) * x[:,:,:,b]

        # x[:,:,:,a] = (100.5 - wave_start) * x[:,:,:,a] + (1 - 100.5 + wave_end) * x[:,:,:,a+1]



        # x[:,:,:,a] = (wave_start - a) * x[:,:,:,a+1] + (1 - wave_start + a) * x[:,:,:,a]
        # x[:,:,:,b] = (wave_end - b) * x[:,:,:,b+1] + (1 - wave_end + b) * x[:,:,:,b]


        # c = 0  还需调整

        return x

    def forward(self, x):
        x = self.stn(x)

        x = F.relu(F.max_pool2d(self.conv1(x), (1,4), (1,4)))
        x = F.relu(F.max_pool2d(self.conv2(x), (1,4), (1,4)))
        x = F.relu(F.max_pool2d(self.conv3(x), (1,4), (1,4)))
        x = x.view(-1, 1888)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01)

Nreal = 1900
batch_size = 10

def train():
    model.train()
    loss_fn = nn.L1Loss()
    for step in range(int(Nreal / batch_size)):
        data = get_params_wave(batch_size,step)
        target = get_inj_time(batch_size,step)

        data = torch.from_numpy(data)
        target = torch.from_numpy(target)

        data = data.float()
        target = target.float()

        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_l_sum = loss.cpu().item()

        print('step %d, loss %.4f' % (step+1, train_l_sum))

train()

# Nreal_test=500
#
# def test():
#     model.eval()
#
#     for step in range(int(Nreal_test / batch_size)):
#         data = get_params_wave_test(batch_size,step)
#         target = get_inj_time_test(batch_size,step)
#
#         data = torch.from_numpy(data)
#         target = torch.from_numpy(target)
#
#         data = data.float()
#         target = target.float()
#
#         data, target = Variable(data), Variable(target)
#
#         output = model(data)
#
#         pred = output.data
#
#         correct = pred.eq(target.data)
#
#         print('step %d, loss %.4f' % (step+1, correct))
#
# test()




#
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0