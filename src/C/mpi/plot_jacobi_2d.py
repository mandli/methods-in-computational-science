#!/usr/bin/env python

import os
import sys
import glob

import numpy
import matplotlib.pyplot as plt

def load_data(path):

    # Estimate number of processors
    num_procs = len(glob.glob(os.path.join(path, "jacobi_*.txt")))

    # Load all data
    data = []
    num_points = [0, 0]
    for i in range(num_procs):
        data.append(numpy.loadtxt(os.path.join(path, "jacobi_%s.txt" % i)))
        num_points[0] += data[-1].shape[0]
        num_points[1] = data[-1].shape[1]
        print(num_points)
    
    # Create data arrays
    x = numpy.linspace(0, numpy.pi, num_points[1])
    y = numpy.linspace(0, numpy.pi, num_points[0])
    X, Y = numpy.meshgrid(x,y)
    
    U = numpy.empty(num_points)
    print(U.shape)
    index = 0
    for i in range(num_procs):
        print(index, data[i].shape[0])
        print(U[:, index:index + data[i].shape[0]].shape)
        print(data[i].transpose().shape)
        print("---")
        U[:, index:index + data[i].shape[0]] = data[i].transpose()
        index += data[i].shape[0]

    return X, Y, U

def plot_solution(x, y, u):
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth() * 3)
    
    axes = fig.add_subplot(1, 3, 1)
    plot = axes.pcolor(X, Y, U)
    fig.colorbar(plot)
    axes.set_title("Computed Solution")
    axes.set_xlabel("x")
    axes.set_ylabel("y")

    axes = fig.add_subplot(1, 3, 2)
    plot = axes.pcolor(X, Y, true_solution(u.shape[0] - 1))
    fig.colorbar(plot)
    axes.set_title("True Solution")
    axes.set_xlabel("x")
    axes.set_ylabel("y")

    axes = fig.add_subplot(1, 3, 3)
    plot = axes.pcolor(X, Y, numpy.abs(U - true_solution(u.shape[0] - 1)))
    fig.colorbar(plot)
    axes.set_title("Error")
    axes.set_xlabel("x")
    axes.set_ylabel("y")

    return None

def true_solution(N):

    x = numpy.linspace(0, numpy.pi, N + 1)
    y = numpy.linspace(0, numpy.pi, N + 1)
    X, Y = numpy.meshgrid(x,y)

    U_true = 2 * numpy.sin(X) * numpy.cos(3 * Y)

    return U_true

if __name__ == '__main__':
    path = os.getcwd()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    X, Y, U = load_data(path)

    fig = plot_solution(X, Y, U)
    plt.show()
