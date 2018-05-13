#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  energy.py
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
import scipy.integrate
from components.misc import get_n, get_rs
from components.xc import get_eps_x_RLDA, get_eps_c_Vosko


def calc_energy(rr, u, psi, eps, Z, V_H=None, V_xc=None):
    V = 0
    if V_H is not None:
        V += V_H / 2
    if V_xc is not None:
        n = get_n(psi)
        r_s = get_rs(n)
        V += V_xc - get_eps_x_RLDA(n) - get_eps_c_Vosko(n)
    E = 2 * eps - 2 * scipy.integrate.simps(u[1:] ** 2 * V[1:], rr[1:])
    return E
