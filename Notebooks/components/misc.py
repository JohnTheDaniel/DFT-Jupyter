#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  misc.py
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


def get_n(Psi):
    return 2 * np.abs(Psi) ** 2


def get_rs(n):
    r_s = np.zeros(np.shape(n))
    for i in range(len(n)):
        if n[i] != 0:
            r_s[i] = np.cbrt(3 / (4 * np.pi * n[i]))
    return r_s
