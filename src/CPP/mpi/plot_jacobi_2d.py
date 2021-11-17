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
    rank_N = numpy.empty(num_procs, dtype=int)
    for i in range(num_procs):
        data.append(numpy.loadtxt(os.path.join(path, "jacobi_%s.txt" % i)))
        N = data[-1].shape[1]
        rank_N[i] = data[-1].shape[0]
    
    print("Grids: N = %s, rank_N = %s" % (N, rank_N))
    assert(N == rank_N.sum())
    
    # Create data arrays
    x = numpy.linspace(0, numpy.pi, N)
    y = numpy.linspace(0, numpy.pi, rank_N.sum())
    X, Y = numpy.meshgrid(x,y)
    
    U = numpy.empty((N, rank_N.sum()))
    index = 0
    for i in range(num_procs):
        U[:, index:index + data[i].shape[0]] = data[i].transpose()
        index += data[i].shape[0]

    return X, Y, U.transpose()

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
