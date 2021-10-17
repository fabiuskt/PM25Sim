import numpy as np
import random

def trueSolution(x,t,l):
    return np.sin(2*np.pi*(x-l*t))


def boundary(x):
    return np.sin(2*np.pi*x)


def generateNrandom(n, m, xBegin, xEnd, tBegin, tEnd,l):
    randX = [random.uniform(xBegin,xEnd) for i in range(n)]
    randT = [random.uniform(tBegin,tEnd) for i in range(n)]
    randBound = [random.uniform(xBegin,xEnd) for i in range(m)]

    innerData = [(trueSolution(randX[i], randT[i], l),randX[i],randT[i]) for i in range(n)]
    boundaryData = [(boundary(randBound[i]),randBound[i],0.0) for i in range(m)]

    np.savetxt("C:\\Users\\Fabius\\Documents\\Python\\PM25Sim\\Data\\firstData\\innerData.csv", innerData, delimiter=';')
    np.savetxt("C:\\Users\\Fabius\\Documents\\Python\\PM25Sim\\Data\\firstData\\boundaryData.csv", boundaryData, delimiter=';')


def readCSV():
    innerData = np.loadtxt("C:\\Users\\Fabius\\Documents\\Python\\PM25Sim\\Data\\firstData\\innerData.csv", delimiter=';')
    boundData = np.loadtxt("C:\\Users\\Fabius\\Documents\\Python\\PM25Sim\\Data\\firstData\\boundaryData.csv", delimiter=';')
    return innerData, boundData

