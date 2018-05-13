#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  effective_potential.py
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
from components.hartree import get_hartree_potential
from components.xc import get_V_x, get_V_xc, get_V_xc_Vosko

def get_effective_potential(rr, n, N_electrons, xc_type):
    ## Get external potential
    V_ext = -Z / rr[1:-1]

    ## Get single electron Hartree potential
    V_sH = get_hartree_potential(rr[1:-1], n[1:-1])

    ## Get total Hartree potential
    V_H = N_electrons * V_sH

    ## Get exchange correlation potential, type depends on function input
    if xc_type == 0:
        V_xc = 0
    elif xc_type == 1:
        V_xc = get_V_x(n[1:-1])
    elif xc_type == 2:
        V_xc = get_V_xc(n[1:-1])
    elif xc_type == 3:
        V_xc = get_V_xc_Vosko(n[1:-1])
    else:
        raise ValueError('Invalid exchange correlation type argument.')

    ## Add the potential terms to form the effective potential
    V_eff = V_ext + V_H + V_xc
    V_eff_diag = np.diagflat(V_eff)

    return V_eff_diag
