#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  hartree.py
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


def get_hartree_potential(rr, n):
    ## Position array properties
    h = rr[1] - rr[0]  # step size
    N = len(rr) - 2  # number of points

    ## Poisson equation
    A = -2 * np.diagflat(np.ones(N)) / h ** 2
    A += np.diagflat(np.ones(N - 1), 1) / h ** 2
    A += np.diagflat(np.ones(N - 1), -1) / h ** 2
    B = -2 * np.pi * n[1:-1] * rr[1:-1]

    ## Solve it
    sol = np.linalg.solve(A, B)

    ## Add boundary points
    U_0 = np.concatenate(([0], sol, [0]))

    ## Calculate one electron hartree potential
    U = U_0 + rr / rr[-1]
    V_sH = U[1:] / rr[1:]
    V_sH = np.concatenate(([1], V_sH))
    return V_sH
