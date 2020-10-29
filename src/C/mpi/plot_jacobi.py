#!/usr/bin/env python

import os

import numpy
import matplotlib.pyplot as plt

def load_data(path, serial=True, num_procs=1):
    
    if serial:
        data = numpy.loadtxt(os.path.join(path, "jacobi_mpi.txt"))
        x = data[:, 0]
        U = data[:, 1]

    else:
        raise NotImplemented("Need to do this...")

    return x, U

def plot_solution(x, U):

    x_true, u_true = true_solution()

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(x, U, 'x')
    axes.plot(x_true, u_true, 'k')

    axes.set_xlim([0.0, 1.0])
    axes.set_xlabel(r"$x$")
    axes.set_ylim([0.0, 3.0])
    axes.set_ylabel(r"$u(x)$")
    axes.set_title(r"Solution to $u_{xx} = f(x)$")

    return fig

def true_solution():

    x = numpy.linspace(0.0, 1.0, 1000)
    U = (4.0 - numpy.exp(1.0)) * x - 1.0 + numpy.exp(x)
    return x, U

if __name__ == '__main__':
    x, U = load_data(os.getcwd())
    fig = plot_solution(x, U)
    fig.savefig("jacobi.png")