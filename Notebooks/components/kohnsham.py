#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  kohnsham.py
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
import scipy.integrate
from components.energy import calc_energy


def solve_ks(rr, Z, V_H=None, V_xc=None):
    ## Position array properties
    h = rr[1] - rr[0]  # step size
    N = len(rr) - 2  # number of points

    A = -2 * np.diagflat(np.ones(N)) / h ** 2
    A += np.diagflat(np.ones(N - 1), 1) / h ** 2
    A += np.diagflat(np.ones(N - 1), -1) / h ** 2

    KS = -1 / 2 * A - np.diagflat(Z / rr[1:-1])

    if V_H is not None:
        KS += np.diagflat(V_H[1:-1])
    if V_xc is not None:
        KS += np.diagflat(V_xc[1:-1])

    eigvals, eigvecs = np.linalg.eig(KS)
    u = eigvecs[:, np.argmin(eigvals)]
    u = np.concatenate(([0], u, [0]))
    u /= np.sqrt(scipy.integrate.simps(u ** 2, rr))
    psi = np.concatenate(
        ([u[1] / (1 - h * Z)], u[1:] / (np.sqrt(4 * np.pi) * rr[1:]))
    )
    E = calc_energy(rr, u, psi, np.min(eigvals), Z, V_H, V_xc)
    return (E, psi)
