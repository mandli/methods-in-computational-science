#!/usr/bin/env python

import os

import numpy
import matplotlib.pyplot as plt

def load_data(path):

    U = numpy.loadtxt(os.path.join(path, "jacobi_0.txt")).transpose()
    
    x = numpy.linspace(0, numpy.pi, U.shape[0])
    y = numpy.linspace(0, numpy.pi, U.shape[0])

    X, Y = numpy.meshgrid(x,y)
    
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
    X, Y, U = load_data(os.getcwd())
    fig = plot_solution(X, Y, U)
    plt.show()