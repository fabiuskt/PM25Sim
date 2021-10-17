import numpy as np


s = 2.0
cx = 1.5
cy = 1.5
xc = 0.0
yc = 0.0
la = 0.1
T0 = 0.22
dC = 0.001
alpha = 1.0

r=0.001
x0 = 2.5
x1 = 0.0
y0 = 2.5
y1 = 0.0
tau0=0.0
tau1=1.0

k = 0.01


def vx(x, y, t):
    vel = cx + la * (x - y) * np.sin(t / T0)
    return vel


def vx_x(x, y, t):
    vel = la * np.sin(t / T0)
    return vel


def vy(x, y, t):
    vel = cy + la * (x + y) * np.sin(t / T0)
    return vel


def vy_y(x, y, t):
    vel = la * np.sin(t / T0)
    return vel


def delta(x):
    delt = (1 / (alpha * np.sqrt(np.pi))) * np.exp(-(x ** 2) / (alpha ** 2))
    return delt


def trueSolution(x, y, t):
    result = np.exp(-s * (
                (-t * (np.sin(t / T0) * (y + x) * la + cy) - yc + y) ** 2 + (
                    -t * (np.sin(t / T0) * (x - y) * la + cx) - xc + x) ** 2))
    #result += r*t*delta(x-x0)*delta(y-y0)*delta(t-tau0)
    #result += r*t*delta(x-x1)*delta(y-y1)*delta(t-tau1)
    return result


def source(x, y, t):
    result = np.exp(-s * (
            (x - xc - t*(cx + la * (x - y) * np.sin(t / T0))) ** 2 + (
            y - yc - t * (
            cy + la * (x + y) * np.sin(t / T0))) ** 2)) * vx_x(x, y,
                                                               t) + np.exp(
        -s * ((x - xc - t * (cx + la * (x - y) * np.sin(t / T0))) ** 2 + (
                y - yc - t * (
                cy + la * (x + y) * np.sin(t / T0))) ** 2)) * vy_y(x, y,
                                                                   t) - np.exp(
        -s * ((x - xc - t * (cx + la * (x - y) * np.sin(t / T0))) ** 2 + (
                y - yc - t * (
                cy + la * (x + y) * np.sin(t / T0))) ** 2)) * s * vx(x,
                                                                     y,
                                                                     t)*(
        2 * (1 - la * t * np.sin(t / T0))*(x - xc - t * (
                cx + la * (x - y) * np.sin(t / T0))) - 2 * la * t * np.sin(
            t / T0) * (y - yc - t * (
                cy + la * (x + y) * np.sin(t / T0)))) - np.exp(-s * (
            (x - xc - t * (cx + la * (x - y) * np.sin(t / T0))) ** 2 + (
            y - yc - t * (
            cy + la * (x + y) * np.sin(t / T0))) ** 2)) * s * vy(x,
                                                                 y,
                                                                 t) * (
                     2 * la * t * np.sin(t / T0) * (x - xc - t * (
                     cx + la * (x - y) * np.sin(t / T0))) + 2 * (
                             1 - la * t * np.sin(t / T0)) * (y - yc - t*(
                 cy + la * (x + y) * np.sin(t / T0)))) - np.exp(-s * (
            (x - xc - t * (cx + la * (x - y) * np.sin(t / T0))) ** 2 + (
            y - yc - t * (
            cy + la * (x + y) * np.sin(t / T0))) ** 2)) * s * (2 * (
            -cx - (la * t * (x - y) * np.cos(t / T0)) / T0 - la * (
            x - y) * np.sin(t / T0)) * (x - xc - t*(
        cx + la * (x - y) * np.sin(t / T0))) + 2 * (-cy - (
            la * t * (x + y) * np.cos(t / T0)) / T0 - la * (x + y) * np.sin(
        t / T0)) * (y - yc - t * (cy + la * (x + y) * np.sin(t / T0)))) - dC * (
                     -2 * np.exp(-s * ((x - xc - t * (
                     cx + la * (x - y) * np.sin(t / T0))) ** 2 + (
                                               y - yc - t * (cy + la * (
                                               x + y) * np.sin(
                                           t / T0))) ** 2)) * s * (
                             2 * la ** 2 * t ** 2 * np.sin(
                         t / T0) ** 2 + 2 * (1 - la * t * np.sin(
                         t / T0)) ** 2) + np.exp(-s * ((x - xc - t * (
                     cx + la * (x - y) * np.sin(t / T0))) ** 2 + (
                                                               y - yc - t * (
                                                               cy + la * (
                                                               x + y) * np.sin(
                                                           t / T0))) ** 2)) * s ** 2 * (
                             2 * (1 - la * t * np.sin(t / T0)) * (
                             x - xc - t*(cx + la * (x - y) * np.sin(
                         t / T0))) - 2 * la * t * np.sin(t / T0) * (
                                     y - yc - t * (
                                     cy + la * (x + y) * np.sin(
                                 t / T0)))) ** 2 + np.exp(-s * (
                     (x - xc - t * (cx + la * (x - y) * np.sin(
                         t / T0))) ** 2 + (y - yc - t * (
                     cy + la * (x + y) + np.sin(
                 t / T0))) ** 2)) * s ** 2 * (
                             2 * la * t * np.sin(t / T0) * (
                             x - xc - t * (
                             cx + la * (x - y) * np.sin(
                         t / T0))) + 2 * (1 - la * t * np.sin(
                         t / T0)) * (y - yc - t * (
                             cy + la * (x + y) * np.sin(
                         t / T0)))) ** 2)
    #result -= r * t * delta(x - x0) * delta(y - y0) * delta(t - tau0)
    #result -= r * t * delta(x - x1) * delta(y - y1) * delta(t - tau1)
    result -= 0.01 * trueSolution(x,y,t)
    return result + k*trueSolution(x,y,t)


def generateEqual(n, m, xBegin, xEnd, yBegin, yEnd, tBegin, tEnd):
    xSteps = np.linspace(xBegin, xEnd, n)
    ySteps = np.linspace(yBegin, yEnd, n)
    tSteps = np.linspace(tBegin, tEnd, m)

    data = []
    gridX, gridY, gridT = np.meshgrid(xSteps, ySteps, tSteps, indexing='ij')

    for i in range(n):
        for j in range(n):
            for k in range(m):
                data.append([trueSolution(gridX[i, j, k], gridY[i, j, k],
                                             gridT[i, j, k]), gridX[i,j,k], gridY[i,j,k], gridT[i,j,k]])

    #for i in range(m):
    #    plt.pcolor(data[:,:,i])
    #    plt.show()

    return np.array(data)




#a, b = readCSV()
#print(a)
#generateEqual(10, 10, -5.0, 5.0, -5.0, 5.0, 0.0, 3.0)
