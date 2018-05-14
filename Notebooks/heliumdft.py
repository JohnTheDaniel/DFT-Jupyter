#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  heliumdft.py
#
#  Copyright 2018 Martin Gulliksson <martin@martingulliksson.com>
#             and John Daniel Bossér <john.daniel@bosser.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#
import numpy as np
import matplotlib.pyplot as plt
from components.tests import (
    test_helium, test_helium_x, test_helium_xc, test_helium_xc_Vosko
)
from components.misc import get_n

## Command line arguments
arg_verbose = False
arg_plot = False

## r = radius in spherical coordinates
r_max = 15
r_step = 0.015
rr = np.arange(0, r_max, r_step)

## Start with wave function for hydrogen
psi_start = 1 / np.sqrt(np.pi) * np.exp(-rr)

## Conditions for exiting the program
tolerance = 1e-7
max_iterations = 12


def main(args):
    ## Parse command line arguments
    global arg_verbose
    global arg_plot
    for arg in args[1:]:
        if arg == "-v" or arg == "verbose" or arg == "--verbose":
            arg_verbose = True
        if arg == "-p" or arg == "plot" or arg == "--plot":
            arg_plot = True

    ## Calculate energies and wave functions
    #psi, E = test_helium(rr, psi_start, tolerance, max_iterations, arg_verbose)
    #print("E =", E)
    #print()
    #psi_x, E_x = test_helium_x(
    #    rr, psi_start, tolerance, max_iterations, arg_verbose
    #)
    #print("E_x =", E_x)
    #print()
    #psi_xc, E_xc = test_helium_xc(
    #    rr, psi_start, tolerance, max_iterations, arg_verbose
    #)
    #print("E_xc_Perdew =", E_xc)
    #print()
    psi_xc_Vosko, E_xc_Vosko, E_vec = test_helium_xc_Vosko(
        rr, psi_start, tolerance, max_iterations, arg_verbose
    )
    print("E_xc_vosko =", E_xc_Vosko)
    #print()

    ## Plot
    if arg_plot:
        ## Calculate electron densities
        #n = get_n(psi)
        #n_x = get_n(psi_x)
        #n_xc = get_n(psi_xc)
        n_xc_Vosko = get_n(psi_xc_Vosko)

        ## Plot electron densities
        #plt.plot(rr[1:], n[1:], label='No XC')
        #plt.plot(rr[1:], n_x[1:], label='X')
        #plt.plot(rr[1:], n_xc[1:], label='XC-Perdew')
        #plt.plot(rr[1:], n_xc_Vosko[1:], label='XC-Vosko')

        ## Plot settings
        #xmin = 0
        #xmax = 2
        #ymin = 0
        #ymax = n_xc_Vosko[1] + 0.2
        #plt.axis([xmin, xmax, ymin, ymax])
        plt.grid()
        plt.xlabel("Iteration number", fontsize='x-large')
        plt.ylabel("Energy [Ha]", fontsize='x-large')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        #plt.legend(loc='upper right', shadow=False, fontsize='x-large')
        x = np.arange(1, len(E_vec) + 1)
        plt.plot(x, E_vec)

        ## Display the plot
        plt.show()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
