import numpy as np
import matplotlib.pyplot as plt
import torch

a0 = 1
sigma = 1
lamb = 0.1
cx = 1.5
cy = 1.5
xc = -2
yc = -4
s = 0

alpha = 0.4

d = 0.001
kappa = 0.01


def vx(x, y, t, T0):
    vel = cx + lamb * (x - y) * torch.sin(t / T0)
    return vel


def vx_x(x, y, t, T0):
    vel = lamb*torch.sin(t /T0)
    return vel


def vy(x, y, t, T0):
    vel = cy + lamb * (x + y) * torch.sin(t / T0)
    return vel


def vy_y(x, y ,t, T0):
    vel = lamb*torch.sin(t /T0)
    return vel


def delta(x):
    delt = (1 / (alpha * np.sqrt(np.pi))) * np.exp(-(x ** 2) / (alpha ** 2))
    return delt


def exacta2(xi, yi, t, T0, x0, x1, y0, y1, tau0, tau1, hora):
    exac = a0 + np.exp(-sigma * ((xi - vx(xi, yi, t, T0) * t - xc) ** 2 + (
            yi - vy(xi, yi, t, T0) * t - yc) ** 2)) + s * (
                   delta(xi - hora - x0) * delta(yi - hora - y0) * delta(
               hora - tau0) + delta(xi - hora - x1) * delta(
               yi - hora - y1) * delta(hora - tau1))
    return exac


def exacta(xi, yi, t, T0, x0, x1, y0, y1, tau0, tau1):
    exac = a0 + np.exp(-sigma * ((xi - vx(xi, yi, t, T0) * t - xc) ** 2 + (
            yi - vy(xi, yi, t, T0) * t - yc) ** 2)) + s * t * (
                   delta(xi - t - x0) * delta(yi - t - y0) * delta(
               t - tau0) + delta(xi - t - x1) * delta(yi - t - y1) * delta(
               t - tau1))
    return exac


def source(x, y, t, T0, x0, x1, y0, y1, tau0, tau1, kappa):
    src = -d * ((sigma ** 2) * (2 * (1 - t * np.sin(t / T0) * lamb) * (-t * (
                np.sin(t / T0) * (
                    y + x) * lamb + cy) - yc + y) + 2 * t * np.sin(
        t / T0) * lamb * (-t * (
                np.sin(t / T0) * (x - y) * lamb + cx) - xc + x)) ** 2 * np.exp(
        -sigma * ((-t * (
                    np.sin(t / T0) * (y + x) * lamb + cy) - yc + y) ** 2 + (
                              -t * (np.sin(t / T0) * (
                                  x - y) * lamb + cx) - xc + x) ** 2)) + (
                            sigma ** 2) * (
                            2 * (1 - t * np.sin(t / T0) * lamb) * (-t * (
                                np.sin(t / T0) * (
                                    x - y) * lamb + cx) - xc + x) - 2 * t * np.sin(
                        t / T0) * lamb * (-t * (np.sin(t / T0) * (
                                y + x) * lamb + cy) - yc + y)) ** 2 * np.exp(
        -sigma * ((-t * (
                    np.sin(t / T0) * (y + x) * lamb + cy) - yc + y) ** 2 + (
                              -t * (np.sin(t / T0) * (
                                  x - y) * lamb + cx) - xc + x) ** 2)) - 2 * sigma * (
                            2 * ((1 - t * np.sin(t / T0) * lamb) ** 2) + 2 * (
                                t ** 2) * (
                                np.sin(t / T0)) ** 2 * lamb ** 2) * np.exp(
        -sigma * ((-t * (
                    np.sin(t / T0) * (y + x) * lamb + cy) - yc + y) ** 2 + (
                              -t * (np.sin(t / T0) * (
                                  x - y) * lamb + cx) - xc + x) ** 2)) + (
                            4 * s * t * ((-y1 + y - t) ** 2) * np.exp(
                        -(-y1 + y - t) ** 2 / (alpha ** 2) - (
                                    -x1 + x - t) ** 2 / (alpha ** 2) - (
                                    t - tau1) ** 2 / (alpha ** 2))) / (
                            (alpha ** 7) * np.pi ** (3 / 2)) + (
                            4 * s * t * ((-x1 + x - t) ** 2) * np.exp(
                        -(-y1 + y - t) ** 2 / (alpha ** 2) - (
                                    -x1 + x - t) ** 2 / (alpha ** 2) - (
                                    t - tau1) ** 2 / (alpha ** 2))) / (
                            (alpha ** 7) * np.pi ** (3 / 2)) - (
                            4 * s * t * np.exp(
                        -(-y1 + y - t) ** 2 / (alpha ** 2) - (
                                    -x1 + x - t) ** 2 / (alpha ** 2) - (
                                    t - tau1) ** 2 / (alpha ** 2))) / (
                            (alpha ** 5) * np.pi ** (3 / 2)) + (
                            4 * s * t * ((-y0 + y - t) ** 2) * np.exp(
                        -(-y0 + y - t) ** 2 / (alpha ** 2) - (
                                    -x0 + x - t) ** 2 / (alpha ** 2) - (
                                    t - tau0) ** 2 / (alpha ** 2))) / (
                            (alpha ** 7) * np.pi ** (3 / 2)) + (
                            4 * s * t * (-x0 + x - t) ** 2 * np.exp(
                        -(-y0 + y - t) ** 2 / (alpha ** 2) - (
                                    -x0 + x - t) ** 2 / (alpha ** 2) - (
                                    t - tau0) ** 2 / (alpha ** 2))) / (
                            (alpha ** 7) * np.pi ** (3 / 2)) - (
                            4 * s * t * np.exp(
                        -(-y0 + y - t) ** 2 / (alpha ** 2) - (
                                    -x0 + x - t) ** 2 / (alpha ** 2) - (
                                    t - tau0) ** 2 / (alpha ** 2))) / (
                            (alpha ** 5) * np.pi ** (3 / 2))) + (
                      np.sin(t / T0) * (y + x) * lamb + cy) * (-sigma * (
                2 * (1 - t * np.sin(t / T0) * lamb) * (-t * (np.sin(t / T0) * (
                    y + x) * lamb + cy) - yc + y) + 2 * t * np.sin(
            t / T0) * lamb * (-t * (
                    np.sin(t / T0) * (x - y) * lamb + cx) - xc + x)) * np.exp(
        -sigma * ((-t * (
                    np.sin(t / T0) * (y + x) * lamb + cy) - yc + y) ** 2 + (
                              -t * (np.sin(t / T0) * (
                                  x - y) * lamb + cx) - xc + x) ** 2)) - (
                                                                           2 * s * t * (
                                                                               -y1 + y - t) * np.exp(
                                                                       -(
                                                                                    -y1 + y - t) ** 2 / (
                                                                                   alpha ** 2) - (
                                                                                   -x1 + x - t) ** 2 / (
                                                                                   alpha ** 2) - (
                                                                                   t - tau1) ** 2 / (
                                                                                   alpha ** 2))) / (
                                                                           (
                                                                                       alpha ** 5) * np.pi ** (
                                                                                       3 / 2)) - (
                                                                           2 * s * t * (
                                                                               -y0 + y - t) * np.exp(
                                                                       -(
                                                                                    -y0 + y - t) ** 2 / (
                                                                                   alpha ** 2) - (
                                                                                   -x0 + x - t) ** 2 / (
                                                                                   alpha ** 2) - (
                                                                                   t - tau0) ** 2 / (
                                                                                   alpha ** 2))) / (
                                                                           (
                                                                                       alpha ** 5) * np.pi ** (
                                                                                       3 / 2))) + (
                      np.sin(t / T0) * (x - y) * lamb + cx) * (-sigma * (
                2 * (1 - t * np.sin(t / T0) * lamb) * (-t * (np.sin(t / T0) * (
                    x - y) * lamb + cx) - xc + x) - 2 * t * np.sin(
            t / T0) * lamb * (-t * (
                    np.sin(t / T0) * (y + x) * lamb + cy) - yc + y)) * np.exp(
        -sigma * ((-t * (
                    np.sin(t / T0) * (y + x) * lamb + cy) - yc + y) ** 2 + (
                              -t * (np.sin(t / T0) * (
                                  x - y) * lamb + cx) - xc + x) ** 2)) - (
                                                                           2 * s * t * (
                                                                               -x1 + x - t) * np.exp(
                                                                       -(
                                                                                    -y1 + y - t) ** 2 / (
                                                                                   alpha ** 2) - (
                                                                                   -x1 + x - t) ** 2 / (
                                                                                   alpha ** 2) - (
                                                                                   t - tau1) ** 2 / (
                                                                                   alpha ** 2))) / (
                                                                           (
                                                                                       alpha ** 5) * np.pi ** (
                                                                                       3 / 2)) - (
                                                                           2 * s * t * (
                                                                               -x0 + x - t) * np.exp(
                                                                       -(
                                                                                    -y0 + y - t) ** 2 / (
                                                                                   alpha ** 2) - (
                                                                                   -x0 + x - t) ** 2 / (
                                                                                   alpha ** 2) - (
                                                                                   t - tau0) ** 2 / (
                                                                                   alpha ** 2))) / (
                                                                           (
                                                                                       alpha ** 5) * np.pi ** (
                                                                                       3 / 2))) + kappa * (
                      np.exp(-sigma * ((-t * (np.sin(t / T0) * (
                                  y + x) * lamb + cy) - yc + y) ** 2 + (-t * (
                                  np.sin(t / T0) * (
                                      x - y) * lamb + cx) - xc + x) ** 2)) + (
                                  s * t * np.exp(
                              -(-y1 + y - t) ** 2 / (alpha ** 2) - (
                                          -x1 + x - t) ** 2 / (alpha ** 2) - (
                                          t - tau1) ** 2 / (alpha ** 2))) / (
                                  (alpha ** 3) * np.pi ** (3 / 2)) + (
                                  s * t * np.exp(
                              -(-y0 + y - t) ** 2 / (alpha ** 2) - (
                                          -x0 + x - t) ** 2 / alpha ** 2 - (
                                          t - tau0) ** 2 / alpha ** 2)) / (
                                  (alpha ** 3) * np.pi ** (
                                      3 / 2)) + a0) - sigma * (2 * (
                -np.sin(t / T0) * (y + x) * lamb - (
                    t * np.cos(t / T0) * (y + x) * lamb) / T0 - cy) * (-t * (
                np.sin(t / T0) * (y + x) * lamb + cy) - yc + y) + 2 * (-np.sin(
        t / T0) * (x - y) * lamb - (t * np.cos(t / T0) * (
                x - y) * lamb) / T0 - cx) * (-t * (
                np.sin(t / T0) * (x - y) * lamb + cx) - xc + x)) * np.exp(
        -sigma * ((-t * (
                    np.sin(t / T0) * (y + x) * lamb + cy) - yc + y) ** 2 + (
                              -t * (np.sin(t / T0) * (
                                  x - y) * lamb + cx) - xc + x) ** 2)) + (
                      s * t * ((2 * (-y1 + y - t)) / (alpha ** 2) + (
                          2 * (-x1 + x - t)) / (alpha ** 2) - (
                                           2 * (t - tau1)) / (
                                           alpha ** 2)) * np.exp(
                  -(-y1 + y - t) ** 2 / (alpha ** 2) - (-x1 + x - t) ** 2 / (
                              alpha ** 2) - (t - tau1) ** 2 / (alpha ** 2))) / (
                      (alpha ** 3) * np.pi ** (3 / 2)) + (s * np.exp(
        -(-y1 + y - t) ** 2 / (alpha ** 2) - (-x1 + x - t) ** 2 / (
                    alpha ** 2) - (t - tau1) ** 2 / (alpha ** 2))) / (
                      (alpha ** 3) * np.pi ** (3 / 2)) - (s * np.exp(
        -(y - y1) ** 2 / (alpha ** 2) - (x - x1) ** 2 / (alpha ** 2) - (
                    t - tau1) ** 2 / (alpha ** 2))) / (
                      (alpha ** 3) * np.pi ** (3 / 2)) + (s * t * (
                (2 * (-y0 + y - t)) / (alpha ** 2) + (2 * (-x0 + x - t)) / (
                    alpha ** 2) - (2 * (t - tau0)) / (alpha ** 2)) * np.exp(
        -(-y0 + y - t) ** 2 / (alpha ** 2) - (-x0 + x - t) ** 2 / (
                    alpha ** 2) - (t - tau0) ** 2 / (alpha ** 2))) / (
                      (alpha ** 3) * np.pi ** (3 / 2)) + (s * np.exp(
        -(-y0 + y - t) ** 2 / (alpha ** 2) - (-x0 + x - t) ** 2 / (
                    alpha ** 2) - (t - tau0) ** 2 / (alpha ** 2))) / (
                      (alpha ** 3) * np.pi ** (3 / 2)) - (s * np.exp(
        -(y - y0) ** 2 / (alpha ** 2) - (x - x0) ** 2 / (alpha ** 2) - (
                    t - tau0) ** 2 / (alpha ** 2))) / (
                      (alpha ** 3) * np.pi ** (3 / 2))
    # src = vy* (-d)* vx *kappa *(-sigma)* (-2*vy*(y-t*vy)-2*vx*(x-t*vx))*np.exp(-sigma*((y-t*vy)**2+(x-t*vx)**2)) + (s*t*((2*(-y1+y-t))/alpha**2+(2*(-x1+x-t))/alpha**2-(2*(t-tau1))/alpha**2)*np.exp(-(-y1+y-t)**2/alpha**2-(-x1+x-t)**2/alpha**2-(t-tau1)**2/alpha**2))/(alpha**3*np.pi**(3/2)) + (s*np.exp(-(-y1+y-t)**2/alpha**2-(-x1+x-t)**2/alpha**2-(t-tau1)**2/alpha**2))/(alpha**3*np.pi**(3/2))-(s*np.exp(-(y-y1)**2/alpha**2-(x-x1)**2/alpha**2-(t-tau1)**2/alpha**2))/(alpha**3*np.pi**(3/2)) + (s*t*((2*(-y0+y-t))/alpha**2+(2*(-x0+x-t))/alpha**2-(2*(t-tau0))/alpha**2)*np.exp(-(-y0+y-t)**2/alpha**2-(-x0+x-t)**2/alpha**2-(t-tau0)**2/alpha**2))/(alpha**3*np.pi**(3/2)) + (s*np.exp(-(-y0+y-t)**2/alpha**2-(-x0+x-t)**2/alpha**2-(t-tau0)**2/alpha**2))/(alpha**3*np.pi**(3/2))-(s*np.exp(-(y-y0)**2/alpha**2-(x-x0)**2/alpha**2-(t-tau0)**2/alpha**2))/(alpha**3*np.pi**(3/2))
    # src = vy*-d*+vx*+kappa*-(s*np.exp(-(y-y1)**2/(1000*alpha**2)-(x-x1)**2/(1000*alpha**2)-(t-tau1)**2/(1000*alpha**2)))/(alpha**3*(np.pi)**(3/2))-(s*np.exp(-(y-y0)**2/(1000*alpha**2)-(x-x0)**2/(1000*alpha**2)-(t-tau0)**2/(1000*alpha**2)))/(alpha**3*(np.pi)**(3/2))-sigma*(-2*vy*(y-t*vy)-2*vx*(x-t*vx))*np.exp(-sigma*((y-t*vy)**2+(x-t*vx)**2))
    return src


x = np.linspace(-4.0, 4.0, 80)
y = np.linspace(-4.0, 4.0, 80)

sigma = 2
xc = -2
yc = -4
x0 = 2.5
y0 = 2.5
x1 = 0
y1 = 0
# tau0 = 7.0
# tau1 = 0.22
tau0 = 0.08
tau1 = 2.18
a0 = 1
cx = 1.5
cy = 1.5
lamb = 0.1
s = 0
epsilon = 0
alpha = 0.4
T0 = 0.2

coeff_D = 0.001
kappa = 0.01

tEnd = 3.0


# xv, yv = np.meshgrid(x, y)
# out = np.zeros((len(xv), len(yv)))
#
# for t in np.linspace(0,3,10):
#     for i in range(len(xv)):
#         for j in range(len(yv)):
#             # out[i,j] = exacta2(array[])
#             out[i, j] = source(xv[i, j], yv[i, j], t, T0, x0, x1, y0, y1, tau0,
#                                tau1,kappa)
#
#     plt.imshow(out, 'hot')
#     plt.show()

def exact_ready(x, y, t):
    result = source(x, y, t, T0, x0, x1, y0, y1, tau0, tau1)
    return result
