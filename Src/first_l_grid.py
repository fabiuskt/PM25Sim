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
from torch.autograd import grad, variable
import matplotlib.pyplot as plt

from Data.Generators.firstGenerator import readCSV, generateNrandom, \
    trueSolution

ltrue = 1.0
l = 0.0
n = 100
m = 100
g = 100
epocs = 5000


generateNrandom(n, m, 0.0, np.pi, 0.0, 1.0, ltrue)
innerData, boundData = readCSV()


def showSlice(t, l, n=10):
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

def showErrorMap(l, n=10):
    with torch.no_grad():
        result = []
        allX = []
        allT = []
        for i in range(n+1):
            x = np.linspace(0.0, np.pi, n)
            allX.append(x)
            xNet = torch.from_numpy(np.array([x]).reshape(n, 1)).float()
            tNet = np.zeros(n)
            rTrue = []
            t = i/n
            allT.append(np.ones(x.shape)*t)
            for j in range(n):
                tNet[j] = t
                rTrue.append(trueSolution(x[j], t,l))
            tNet = tNet.reshape(n, 1)
            tNet = torch.from_numpy(tNet).float()
            a = mynet(xNet, tNet).numpy()[:,0]
            result.append(a - np.array(rTrue))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.pcolor(allX, allT, result, shading='auto')
        ax.axis(xmin=0.0, xmax=np.pi, ymin=0.0, ymax=1.0)
        ax.scatter(boundData[:, 1],boundData[:, 2],marker='x', color='red')
        ax.scatter(innerData[:, 1],innerData[:, 2],marker='x', color='red')

        xspace = np.linspace(0.0, np.pi, 10)
        yspace = np.linspace(0.0, 1.0, 10)
        gridX, gridY = np.meshgrid(xspace, yspace)
        plot = ax.scatter(gridX, gridY, marker='x', color='green')

        plt.xlabel("x", axes=ax)
        plt.ylabel("t", axes=ax)

        fig.colorbar(plot)
        fig.show()
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
mx = bvx.reshape(int(m/10), 10, 1)  # this is all zeros
mt = bvt.reshape(int(m/10), 10, 1)  # this is the full x-axis
mu = bvu.reshape(int(m/10), 10, 1)  # this is the u values on t=0
nx = ix.reshape(int(n/10), 10, 1) # this is all zeros
nt = it.reshape(int(n/10), 10, 1)  # this is the full x-axis
nu = iu.reshape(int(n/10), 10, 1)  # this is the u values on t=0
# now (mx, mt, mu) is the initial condition

boundaryTriple = list(zip(mx, mt, mu))
innerTriple = list(zip(nx, nt, nu))

allTriple = boundaryTriple + innerTriple
# boundaryTriple is a list of all the boundary tuples

zeros = torch.zeros(g*g, dtype=torch.float, requires_grad=True).reshape(int(g*g/10), 10, 1)
xspace = torch.from_numpy(np.linspace(0.0, np.pi,g)).float()
yspace = torch.from_numpy(np.linspace(0.0, 1.0, g)).float()
xspace.requires_grad = True
yspace.requires_grad = True
gridX, gridY = torch.meshgrid([xspace,yspace])
gridX = gridX.reshape(int(g*g/10), 10, 1)
gridY = gridY.reshape(int(g*g/10), 10, 1)

bb = []
ze = zeros[0].reshape(1, 10, 1)
for i in range(10):
    for j in range(int(g*g/10)):
        xb = gridX[j].reshape(1, 10, 1)
        tb = gridY[j].reshape(1, 10, 1)
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

variable=Variable(torch.from_numpy(np.array(l)),requires_grad=True)

optimizer = optim.SGD([
                {'params': mynet.parameters()},
                {'params': variable}],lr=0.01, momentum=0.9)
showErrorMap(variable, n=100)
loss_fn = nn.MSELoss()

print(len(btch))
for epoc in range(1, epocs + 1):
    loss2tot = 0.0
    for i in range(len(btch)):
        # pick a random boundary batch
        b = btch[np.random.randint(0, len(allTriple))]
        # pick a random interior batch
        bf = btch2[np.random.randint(0, len(bb))]
        optimizer.zero_grad()
        outputs = mynet(b[0], b[1])
        outputsf = f(bf[0][0], bf[0][1], variable)
        loss = loss_fn(outputs, b[2])
        loss2 = loss_fn(outputsf, bf[0][2])
        loss2tot += loss2
        losstot[i] = loss
        losst = loss + loss2
        losst.backward(retain_graph=True)
        optimizer.step()
    if epoc % 1 == 0:
        print("lambda = ", variable)
        loss = 0.0
        #showErrorMap(variable, n=100)
        #showSlice(0.5, n=30)
        for i in range(len(allTriple)):
            loss += losstot[i]
        print('epoc %d bndry loss %f, f loss %f' % (
            epoc, float(loss), float(loss2tot)))
    #if epoc == 2500:
    #    optimizer.param_groups[0]['lr'] = 0.001

for i in range(10):
    print(variable)
    showSlice(i/10, variable, 100)
    # if epoc % 50 == 0:
    #    torch.save(mynet.state_dict(), 'first')


