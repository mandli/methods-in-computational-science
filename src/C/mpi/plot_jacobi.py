#!/usr/bin/env python

import sys
import os
import glob
import argparse

import numpy
import matplotlib.pyplot as plt

def load_data(path):

    # Estimate number of processors
    num_procs = len(glob.glob(os.path.join(path, "jacobi_*.txt")))

    # Load all data
    data = []
    num_points = 0
    for i in range(num_procs):
        data.append(numpy.loadtxt(os.path.join(path, "jacobi_%s.txt" % i)))
        num_points += data[-1].shape[0]
    
    # Create data arrays
    x = numpy.empty(num_points)
    U = numpy.empty(num_points)
    index = 0
    for i in range(num_procs):
        x[index:index + data[i].shape[0]] = data[i][:, 0]
        U[index:index + data[i].shape[0]] = data[i][:, 1]
        index += data[i].shape[0]

    return x, U

def plot_solution(x, U):

    x_true, u_true = true_solution()

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(x, U, 'ro')
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
    path = os.getcwd()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    x, U = load_data(path)
    fig = plot_solution(x, U)
    fig.savefig("jacobi.png")
    plt.show()