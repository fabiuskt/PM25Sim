# This is a basic PINN for solving burgers equation.
# it uses an exact solution  from https://people.sc.fsu.edu/~jburkardt/py_src/burgers_solution/burgers_solution.html

import torch
import numpy as np
import os
import time
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt

from Data.Generators.firstGenerator import readCSV, generateNrandom, \
    trueSolution

l = 1.0
n = 100
m = 100
epocs = 3000


generateNrandom(n, m, 0.0, np.pi, 0.0, 1.0, l)
innerData, boundData = readCSV()


def showSlice(t, n=10):
    with torch.no_grad():
        x = np.linspace(0.0, np.pi, n)
        xNet = torch.from_numpy(np.array([x]).reshape(n, 1)).float()
        tNet = np.zeros(n)
        rTrue = []
        for i in range(n):
            tNet[i] = t
            rTrue.append(trueSolution(x[i], t,l))
        tNet = tNet.reshape(n, 1)
        tNet = torch.from_numpy(tNet).float()
        plt.plot(x, mynet(xNet,tNet))
        plt.plot(x, rTrue)
        plt.show()


# a very simple torch method to compute derivatives.
def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True, allow_unused=True)[0]
        f = grads
        if grads is None:
            print('bad grad')
            return torch.tensor(0.)
    return grads


# no attempt here to optimize the number or size of layers.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.il = nn.Linear(2, 80)
        self.mi = nn.Linear(80, 80)
        self.mi1 = nn.Linear(80, 80)
        self.mi2 = nn.Linear(80, 40)
        self.ol = nn.Linear(40, 1)
        self.tn = nn.Tanh()
        self.layers = nn.ModuleList(
            [self.il, self.mi, self.mi1, self.mi2, self.ol])

    def forward(self, x, t):
        u = torch.cat((x, t), 1)
        u = self.il(u)
        for x in self.layers[1:]:
            u = x(self.tn(u))
        return u


def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]


def f(x, t, l):
    u = mynet(x, t)
    # u = [v[0], v[1]]
    u_t = nth_derivative(flat(u), wrt=t, n=1)
    u_x = nth_derivative(flat(u), wrt=x, n=1)
    f = u_t + l * u_x
    return f


# bvx (=boundary values x) is the set of x axis points as a tensor
# bvt (=boundary values t) is the set of t axis points as a tensor
# bvu (=boundary values u) is the set of values u(x,t) as a tensor
bvx = torch.from_numpy(boundData[:, 1]).float()
bvx = bvx.reshape(m, 1)
bvx.requires_grad = True

bvt = torch.from_numpy(boundData[:, 2]).float()
bvt = bvt.reshape(m, 1)
bvt.requires_grad = True

bvu = torch.from_numpy(boundData[:, 0]).float()  # boundary for t = 0
bvu = bvu.reshape(m, 1)

# ix (=inner x) is the set of x axis points as a tensor
# it (=inner t) is the set of t axis points as a tensor
# iu (=inner u) is the set of values u(x,t) as a tensor
ix = torch.from_numpy(innerData[:, 1]).float()
ix = ix.reshape(n, 1)
ix.requires_grad = True

it = torch.from_numpy(innerData[:, 2]).float()
it = it.reshape(n, 1)
it.requires_grad = True

iu = torch.from_numpy(innerData[:, 0]).float()  # boundary for t = 0
iu = iu.reshape(n, 1)

# divide values into four batches of 10.
mx = bvx.reshape(4, int(m/4), 1)  # this is all zeros
mt = bvt.reshape(4, int(m/4), 1)  # this is the full x-axis
mu = bvu.reshape(4, int(m/4), 1)  # this is the u values on t=0
nx = ix.reshape(4, int(n/4), 1)  # this is all zeros
nt = it.reshape(4, int(n/4), 1)  # this is the full x-axis
nu = iu.reshape(4, int(n/4), 1)  # this is the u values on t=0
# now (mx, mt, mu) is the initial condition

boundaryTriple = list(zip(mx, mt, mu))
innerTriple = list(zip(nx, nt, nu))

allTriple = boundaryTriple + innerTriple
# boundaryTriple is a list of all the boundary tuples

zeros = torch.zeros(m, dtype=torch.float, requires_grad=True).reshape(4, int(n/4), 1)
bb = []
ze = zeros[0].reshape(1, int(n/4), 1)
for i in range(len(bvt)):
    ts = zeros.clone()
    ts[:][:] = bvt[i]
    for j in range(4):
        xb = mx[j].reshape(1, int(n/4), 1)
        tb = ts[0].reshape(1, int(n/4), 1)
        bb.append(list(zip(xb, tb, ze)))

    # bb is a list of all triplex (x, t, 0.0) for all x and t
# so it describes the interior

mynet = Net()
# if you want to train for more iterations after the first 20000 epochs load the
# model and go from there
# mynet.load_state_dict(torch.load('burgmodel'))
print(mynet.parameters())
btch = allTriple
btch2 = bb
losstot = np.zeros(len(allTriple))
losstot2 = np.zeros(len(bb))
loss_fn = nn.MSELoss()
optimizer = optim.SGD(mynet.parameters(), lr=0.1)
alsf = list(mynet.parameters())
alsf.append(m)
loss_fn = nn.MSELoss()
for epoc in range(1, epocs + 1):
    loss2tot = 0.0
    for i in range(len(btch)):
        # pick a random boundary batch
        b = btch[np.random.randint(0, len(allTriple))]
        # pick a random interior batch
        bf = btch2[np.random.randint(0, len(bb))]
        optimizer.zero_grad()
        outputs = mynet(b[0], b[1])
        outputsf = f(bf[0][0], bf[0][1], l)
        loss = loss_fn(outputs, b[2])
        loss2 = loss_fn(outputsf, bf[0][2])
        loss2tot += loss2
        losstot[i] = loss
        losst = loss + loss2
        losst.backward(retain_graph=True)
        optimizer.step()
    if epoc % 500 == 0:
        loss = 0.0
        #showSlice(0.5, n=30)
        for i in range(len(allTriple)):
            loss += losstot[i]
        print('epoc %d bndry loss %f, f loss %f' % (
            epoc, float(loss), float(loss2tot)))

for i in range(10):
    showSlice(i/10,100)
    # if epoc % 50 == 0:
    #    torch.save(mynet.state_dict(), 'first')


