import numpy as np
import random
import matplotlib.pyplot as plt


def trueSolution(x,t,l):
    return np.sin(2*np.pi*(x-l*t))


def boundary(x):
    return np.sin(2*np.pi*x)


def generateNrandom(n, m, xBegin, xEnd, tBegin, tEnd,l):
    xSteps = np.linspace(xBegin, xEnd, n)
    tSteps = np.linspace(tBegin, tEnd, m)

    deltaX = xSteps[1]-xSteps[0]
    deltaT = tSteps[1]-tSteps[0]

    data = np.zeros((n,m))
    data2 = np.zeros((n,m))
    for i in range(n):
        data[0,i] = trueSolution(xSteps[0],tSteps[i],l)
        data[i,0] = trueSolution(xSteps[i],tSteps[0],l)

        data2[0, i] = trueSolution(xSteps[0], tSteps[i], l)
        data2[i, 0] = trueSolution(xSteps[i], tSteps[0], l)

    gridX, gridT =np.meshgrid(xSteps,tSteps)

    for i in range(1,n):
        for j in range(1,m):
            data[i,j]=data[i,j-1] *(1-deltaT/deltaX*l)+ data[i-1,j]*deltaT/deltaX*l
            data2[i,j] = trueSolution(xSteps[i], tSteps[j], l)

    s = data-data2
    plt.pcolor(data)
    plt.show()
    gridX, gridT =np.meshgrid(xSteps,tSteps)
    innerData = [data.reshape(n*m),gridX.reshape(n*m),gridT.reshape(n*m)]

    np.savetxt("C:\\Users\\Fabius\\Documents\\Python\\PM25Sim\\Data\\firstData\\innerData.csv", innerData, delimiter=';')


def readCSV():
    innerData = np.loadtxt("C:\\Users\\Fabius\\Documents\\Python\\PM25Sim\\Data\\firstData\\innerData.csv", delimiter=';')
    boundData = np.loadtxt("C:\\Users\\Fabius\\Documents\\Python\\PM25Sim\\Data\\firstData\\boundaryData.csv", delimiter=';')
    return innerData, boundData

a,b = readCSV()
print(a)
generateNrandom(10,100,0.0,np.pi,0.0,1.0,1.0)
