import matplotlib.pyplot as plt

import torch
import numpy as np
import os
import time
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.autograd import grad

torch.autograd.set_detect_anomaly(True)
print(torch.cuda.is_available())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.il = nn.Linear(1, 80)
        self.mi = nn.Linear(80, 80)
        self.mi1 = nn.Linear(80, 80)
        self.mi2 = nn.Linear(80, 40)
        self.ol = nn.Linear(40, 1)
        # self.tn = nn.ReLU()
        self.tn = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, x):
        u = x
        # print(u)
        hidden1 = self.il(u)
        hidden2 = self.mi(self.tn(hidden1))
        hidden2a = self.mi1(self.tn(hidden2))
        hidden3 = self.mi2(self.tn(hidden2a))
        out = self.ol(self.tn(hidden3))
        return out


def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]


vxn = 50
vx = np.linspace(0, np.pi, vxn)
ix = torch.FloatTensor(vx).reshape(vxn, 1)


def u_true(x):
    u = np.sin(x) ** 2
    return u


def f_true(x):
    f = np.sin(2.0 * x) + 2.0 * x * np.cos(2 * x)
    # f = 4.0*np.cos(2.0*x)
    return f


plt.plot(ix, ix)
plt.plot(ix, f_true(ix))

bndry = torch.FloatTensor([vx[0], vx[vxn - 1]]).reshape(2, 1)
Truebndry = torch.FloatTensor([0., 0.]).reshape(2, 1)
mynet = Net()
mynet(bndry)


def Du(x):
    u = mynet(x)
    u_x = grad(flat(u), x, create_graph=True, allow_unused=True)[
        0]  # nth_derivative(flat(u), wrt=x, n=1)
    z = u_x * x
    u_xx = grad(flat(z), x, create_graph=True, allow_unused=True)[
        0]  # nth_derivative(flat(u), wrt=x, n=1)
    f = u_xx
    return f


mynet = Net()

import random

batches = []
fbatches = []
for i in range(20):
    b = random.choices(vx, k=10)
    bar = np.array(b)
    fb = f_true(bar)
    fb0 = torch.FloatTensor(fb).reshape(10, 1)
    fb0.requires_grad = True
    ib = torch.FloatTensor(b).reshape(10, 1)
    ib.requires_grad = True
    batches.append(ib)
    fbatches.append(fb0)
Du(batches[13])

mynet = Net()
epocs = 200
m = len(batches)
loss_fn = nn.MSELoss()
optimizerf = optim.SGD(mynet.parameters(), lr=0.001)
optimizerbdry = optim.SGD(mynet.parameters(), lr=0.001)

loss_fn = nn.MSELoss()
for epoc in range(1, epocs):
    print(epoc)
    loss2tot = 0.0
    for i in range(len(batches)):
        batch_index = np.random.randint(0, m)
        b = batches[batch_index]
        fb = fbatches[batch_index]

        outputsf = Du(b)
        output_bndry = mynet(bndry)
        loss_bndry = loss_fn(output_bndry, Truebndry)
        optimizerf.zero_grad()

        lossf = loss_fn(outputsf, fb)
        loss2tot += lossf
        optimizerbdry.zero_grad()

        lossf.backward(retain_graph=True)
        loss_bndry.backward(retain_graph=True)

        optimizerf.step()
        optimizerbdry.step()
    if epoc % 50 == 0:
        loss = 0.0
        print('epoc %d bndry loss %f, f loss %f' % (epoc, float(loss_bndry),
                                                    float(
                                                        loss2tot)))  # file=open('./elliptic_out.txt','a'))
    if epoc % 1000 == 0:
        torch.save(mynet.state_dict(), 'nonlinmodel')

# mynet.load_state_dict(torch.load('nonlinmodel'))
ix = torch.FloatTensor(vx).reshape(vxn, 1)
u = mynet(ix)
plt.plot(ix, u_true(ix), linewidth=6)
plt.plot(ix, u.detach().numpy(), linewidth=2)

ix = torch.FloatTensor(vx).reshape(vxn, 1)
plt.plot(ix, f_true(ix), linewidth=6)
ix.requires_grad = True
du = Du(ix)
du = du.detach()
plt.plot(ix.detach(), du, linewidth=2)
plt.show()


# def u_true(x):
#     u = np.sin(2 * x) ** 2
#     return u / 8.0
#
#
# def f_true(x):
#     f = np.sin(4.0 * x) / 4.0 + x * np.cos(4 * x)
#     # f = 4.0*np.cos(2.0*x)
#     return f
#
#
# ix = torch.FloatTensor(vx).reshape(vxn, 1)
# plt.plot(ix, ix / 8.0)
# plt.plot(ix, f_true(ix))
#
# mynet = Net()
#
#
# def Du(x):
#     u = mynet(x)
#     u_x = grad(flat(u), x, create_graph=True, allow_unused=True)[
#         0]  # nth_derivative(flat(u), wrt=x, n=1)
#     z = u_x * x
#     u_xx = grad(flat(z), x, create_graph=True, allow_unused=True)[
#         0]  # nth_derivative(flat(u), wrt=x, n=1)
#     f = u_xx
#     return f
#

# import random
#
# batches = []
# fbatches = []
# for i in range(20):
#     b = random.choices(vx, k=10)
#     bar = np.array(b)
#     fb = f_true(bar)
#     fb0 = torch.FloatTensor(fb).reshape(10, 1)
#     fb0.requires_grad = True
#     ib = torch.FloatTensor(b).reshape(10, 1)
#     ib.requires_grad = True
#     batches.append(ib)
#     fbatches.append(fb0)
# Du(batches[13])
#
# # mynet = Net()
# epocs = 3003
# m = len(batches)
# loss_fn = nn.MSELoss()
# optimizerf = optim.SGD(mynet.parameters(), lr=0.0005)
# optimizerbdry = optim.SGD(mynet.parameters(), lr=0.001)
#
# loss_fn = nn.MSELoss()
# for epoc in range(1, epocs):
#     print(epoc)
#     loss2tot = 0.0
#     for i in range(len(batches)):
#         batch_index = np.random.randint(0, m)
#         b = batches[batch_index]
#         fb = fbatches[batch_index]
#         optimizerf.zero_grad()
#         optimizerbdry.zero_grad()
#         output_bndry = mynet(bndry)
#         outputsf = Du(b)
#         loss_bndry = loss_fn(output_bndry, Truebndry)
#         lossf = loss_fn(outputsf, fb)
#         lossf.backward(retain_graph=True)
#         optimizerf.step()
#         loss2tot += lossf
#         loss_bndry.backward(retain_graph=True)
#         optimizerbdry.step()
#     if epoc % 50 == 0:
#         loss = 0.0
#         print('epoc %d bndry loss %f, f loss %f' % (epoc, float(loss_bndry),
#                                                     float(
#                                                         loss2tot)))  # file=open('./elliptic_out.txt','a'))
#     if epoc % 3000 == 0:
#         torch.save(mynet.state_dict(), 'nonlinmodel')
#
# ix = torch.FloatTensor(vx).reshape(vxn, 1)
# u = mynet(ix)
# plt.plot(ix, u_true(ix), linewidth=6)
# plt.plot(ix, u.detach().numpy(), linewidth=2)
#
# ix = torch.FloatTensor(vx).reshape(vxn, 1)
# plt.plot(ix, f_true(ix), linewidth=6)
# ix.requires_grad = True
# du = Du(ix)
# du = du.detach()
# plt.plot(ix.detach(), du, linewidth=2)
