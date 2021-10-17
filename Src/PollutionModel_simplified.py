# This is a basic PINN for solving burgers equation.
# it uses an exact solution  from https://people.sc.fsu.edu/~jburkardt/py_src/burgers_solution/burgers_solution.html

import torch
import numpy as np

from torch import nn
from torch import optim

from torch.autograd import grad, variable
import matplotlib.pyplot as plt

from Data.Generators.AdDifGenerator import generateEqual, trueSolution,  source
from Src.exact import vx, vx_x, vy, vy_y

n = 1000

a = 10
b = 10

g = 20

timeStart = 0.0
timeEnd = 3.0

xStart = -5.0
xEnd = 5.0

yStart = -5.0
yEnd = 5.0

batchSize = 10
epocs = 5000

T0 = 0.2

data = generateEqual(a, b, -5.0, 5.0, -5.0, 5.0, 0.0, 3.0)





def showErrorMap(t, outputNumber=0, n=10):
    with torch.no_grad():
        result = []
        allT = []
        x = np.linspace(-5.0, 5.0, n)
        y = np.linspace(-5.0, 5.0, n)
        gridX, gridY = np.meshgrid(x, y)
        for i in range(n):

            xNet = torch.from_numpy(gridX[:, i]).reshape(n, 1).float()
            yNet = torch.from_numpy(gridY[:, i]).reshape(n, 1).float()
            tNet = np.zeros(n)
            rTrue = []
            allT.append(np.ones(x.shape) * t)
            for j in range(n):
                tNet[j] = t
                rTrue.append(source(gridX[i, j], gridY[i, j], t))
            tNet = tNet.reshape(n, 1)
            tNet = torch.from_numpy(tNet).float()
            b = mynet(xNet, yNet, tNet).numpy()
            a = b[:, outputNumber]
            result.append(rTrue)  # - np.array(rTrue))

    plt.pcolor(gridX, gridY, result, shading='auto')
    plt.colorbar()
    plt.axis(xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0)
    # ax.scatter(boundData[:, 1], boundData[:, 2], marker='x', color='red')
    plt.scatter(data[:, 1], data[:, 2], marker='x', color='red')

    # xspace = np.linspace(-5.0, 5.0, 10)
    # yspace = np.linspace(0.0, 1.0, 10)
    # gridX, gridY = np.meshgrid(xspace, yspace)
    # plot = ax.scatter(gridX, gridY, marker='x', color='green')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("exact"+["contamination","sourceterm","diffusion coefficient 1","diffusion coefficient 2"][outputNumber] + " - t = " + str(t))

    plt.show()


# a very simple torch method to compute derivatives.
def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True, allow_unused=True)[0]
        f = flat(grads)
        if grads is None:
            print('bad grad')
            return torch.tensor(0.)
    return grads


# no attempt here to optimize the number or size of layers.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.il = nn.Linear(3, 80)
        self.mi = nn.Linear(80, 80)
        self.mi1 = nn.Linear(80, 80)
        self.mi2 = nn.Linear(80, 40)
        self.ol = nn.Linear(40, 4)
        self.tn = nn.Tanh()
        self.layers = nn.ModuleList(
            [self.il, self.mi, self.mi1, self.mi2, self.ol])

    def forward(self, x, y, t):
        u = torch.cat((x, y, t), 1)
        u = self.il(u)
        for x in self.layers[1:]:
            u = x(self.tn(u))
        # u = u.reshape((10,5,1))
        return u


def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]


def f(x, y, t):
    netOutput = mynet(x, y, t)
    # netOutput = [u, k, Q, d1, d2]
    u_t = nth_derivative(flat(netOutput[:, 0]), wrt=t, n=1)
    u_x = nth_derivative(flat(netOutput[:, 0]), wrt=x, n=1)
    u_y = nth_derivative(flat(netOutput[:, 0]), wrt=y, n=1)

    u_xx = nth_derivative(flat(netOutput[:, 0]), wrt=x, n=2)
    u_yy = nth_derivative(flat(netOutput[:, 0]), wrt=y, n=2)

    d1_x = nth_derivative(flat(netOutput[:, 2]), wrt=x, n=1)
    d2_y = nth_derivative(flat(netOutput[:, 3]), wrt=y, n=1)

    f = u_t + vx(x, y, t, T0) * u_x + vx_x(x, y, t, T0) * netOutput[:,
                                                          0].reshape(1,
                                                                     -1).t() + vy(
        x,
        y,
        t,
        T0) * u_y + vy_y(
        x, y, t, T0) * netOutput[:, 0].reshape(1,
                                               -1).t() - d1_x * u_x - netOutput[:,2].reshape(1, -1).t() * u_xx - d2_y * u_y - netOutput[:, 3].reshape(1,
                                                                 -1).t() * u_yy - netOutput[
                                                                                  :,
                                                                                  1].reshape(
        1, -1).t()

    return f


# ix (=inner x) is the set of x axis points as a tensor
# iy (=inner y) is the set of y axis points as a tensor
# it (=inner t) is the set of t axis points as a tensor
# iu (=inner u) is the set of values u(x,t) as a tensor
ix = torch.from_numpy(data[:, 1]).float()
ix = ix.reshape(n, 1)
ix.requires_grad = True

iy = torch.from_numpy(data[:, 2]).float()
iy = iy.reshape(n, 1)
iy.requires_grad = True

it = torch.from_numpy(data[:, 3]).float()
it = it.reshape(n, 1)
it.requires_grad = True

iu = torch.from_numpy(data[:, 0]).float()  # boundary for t = 0
iu = iu.reshape(n, 1)

# divide values into four batches of batchSize.
nx = ix.reshape(int(n / batchSize), batchSize, 1)
ny = iy.reshape(int(n / batchSize), batchSize, 1)
nt = it.reshape(int(n / batchSize), batchSize, 1)
nu = iu.reshape(int(n / batchSize), batchSize, 1)

allQuadruples = list(zip(nx, ny, nt, nu))
# allTriple is a list of all the inner tripples

zeros = torch.zeros(g * g, dtype=torch.float, requires_grad=True).reshape(
    int(g * g / batchSize), batchSize, 1)
xspace = torch.from_numpy(np.linspace(xStart, xEnd, g)).float()
yspace = torch.from_numpy(np.linspace(yStart, yEnd, g)).float()
tspace = torch.from_numpy(np.linspace(timeStart, timeEnd, g)).float()
xspace.requires_grad = True
yspace.requires_grad = True
tspace.requires_grad = True
gridX, gridY, gridT = torch.meshgrid([xspace, yspace, tspace])
gridX = gridX.reshape(int(g * g * g / batchSize), batchSize, 1)
gridY = gridY.reshape(int(g * g * g / batchSize), batchSize, 1)
gridT = gridT.reshape(int(g * g * g / batchSize), batchSize, 1)

bb = []
ze = zeros[0].reshape(1, batchSize, 1)
for i in range(batchSize):
    for j in range(int(g * g * g / batchSize)):
        xb = gridX[j].reshape(1, batchSize, 1)
        yb = gridY[j].reshape(1, batchSize, 1)
        tb = gridT[j].reshape(1, batchSize, 1)
        bb.append(list(zip(xb, yb, tb, ze)))

    # bb is a list of all quadruples (x, y, t, 0.0) for all x and t
# so it describes the interior

mynet = Net()
# if you want to train for more iterations after the first 20000 epochs load the
# model and go from there
# mynet.load_state_dict(torch.load('burgmodel'))
print(mynet.parameters())
btch = allQuadruples
btch2 = bb
losstot = np.zeros(len(allQuadruples))
losstot2 = np.zeros(len(bb))
loss_fn = nn.MSELoss()

optimizer = optim.SGD(mynet.parameters(), lr=0.01, momentum=0.9)
#showErrorMap(0.0, n=100)
loss_fn = nn.MSELoss()

print(len(btch))
for epoc in range(1, epocs + 1):
    loss2tot = 0.0
    np.random.shuffle(btch)
    np.random.shuffle(btch2)
    for i in range(len(btch)):
        # pick a random boundary batch
        b = btch[i]
        # pick a random interior batch
        bf = btch2[i]
        optimizer.zero_grad()
        outputs = mynet(b[0], b[1], b[2])
        outputsf = f(bf[0][0], bf[0][1], bf[0][2])
        loss = loss_fn(outputs[:, 0].reshape(1, -1).t(), b[3])
        loss2 = loss_fn(outputsf, bf[0][3])
        loss2tot += loss2
        losstot[i] = loss
        losst = loss + loss2
        losst.backward(retain_graph=True)
        optimizer.step()
    if epoc % 1 == 0:
        loss = 0.0
        if epoc %1 == 0:
            for t in np.linspace(0.0, 3.0, 5):
                for i in range(4):
                    showErrorMap(t, i, n=100)
        # showSlice(0.5, n=30)
        for i in range(len(allQuadruples)):
            loss += losstot[i]
        print('epoc %d data loss %f, f PDE loss %f' % (
            epoc, float(loss), float(loss2tot)))
    # if epoc == 2500:
    #    optimizer.param_groups[0]['lr'] = 0.001


