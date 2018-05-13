#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  xc.py
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
from components.misc import get_rs

A = 0.0311
B = -0.048
C = 0.0020
D = -0.0116
gamma = -0.1423
beta_1 = 1.0529
beta_2 = 0.3334

a = 0.0621814
b = 3.72744
c = 12.9352
Q = np.sqrt(4 * c - b ** 2)
y0 = -0.10498
Y0 = y0 ** 2 + b * y0 + c


def get_V_xc_Vosko(n):
    V_x = get_eps_x(n) + get_Deps_x_times_n(n)
    beta = np.cbrt(3 * np.pi ** 2 * n) / c
    mu = np.sqrt(1 + beta ** 2)
    S = 3 * np.log(beta + mu) / (2 * beta * mu) - 1 / 2
    get_nans = np.isnan(S)
    S[get_nans] = 0
    V_x_RLDA = V_x * S
    V_c = get_eps_c_Vosko(n) + n * get_Deps_c_Vosko(n)
    V_xc = V_x_RLDA + V_c
    return V_xc


def get_V_xc(n):
    r_s = get_rs(n)
    V_x = get_eps_x_RLDA(n) + get_Deps_x_times_n(n)
    V_c = get_eps_c(r_s) + n * get_Deps_c(r_s)
    V_xc = V_x + V_c
    return V_xc


def get_V_x(n):
    V_x = get_eps_x_RLDA(n) + get_Deps_x_times_n(n)
    V_xc = V_x
    return V_xc


def get_eps_x(n):
    e_x = -3 / 4 * np.cbrt(3 * n / np.pi)
    return e_x


def get_eps_x_RLDA(n):
    e_x = get_eps_x(n)
    eps_x_RLDA = e_x
    beta = np.cbrt(3 * np.pi ** 2 * n) / c
    mu = np.sqrt(1 + beta ** 2)
    for i in range(0, len(n)):
        if n[i] > 0:
            R = 1 - 3 / 2 * (
                (beta[i] * mu[i] - np.log(beta[i] + mu[i])) / beta[i] ** 2
            ) ** 2
            eps_x_RLDA[i] = e_x[i] * R
    return eps_x_RLDA


def get_Deps_x_times_n(n):
    De_x_times_n = -1 / 4 * np.cbrt(3 * n / np.pi)
    return De_x_times_n


def get_eps_c(r_s):
    eps_c = np.zeros(np.shape(r_s))
    for i in range(0, len(r_s)):
        if r_s[i] >= 1:
            eps_c[i] = gamma / (1 + beta_1 * np.sqrt(r_s[i]) + beta_2 * r_s[i])
        elif r_s[i] > 0:
            eps_c[i] = A * np.log(r_s[i]) + B + C * r_s[i] * np.log(
                r_s[i]
            ) + D * r_s[
                i
            ]
    return eps_c


def get_eps_c_Vosko(n):
    eps_c = np.zeros(np.shape(n))
    r_s = np.zeros(np.shape(n))
    for i in range(len(n)):
        if n[i] != 0:
            r_s[i] = np.cbrt(3 / (4 * np.pi * n[i]))
            y = np.sqrt(r_s[i])
            Y = y ** 2 + b * y + c

            e1 = np.log(y ** 2 / Y)
            e2 = 2 * b / Q * np.arctan(Q / (2 * y + b))
            e3 = np.log((y - y0) ** 2 / (Y))
            e4 = 2 * (b + 2 * Y0) / Q * np.arctan(Q / (2 * y + b))

            eps_c[i] = A / 2 * (e1 + e2 - b * y0 / Y0 * (e3 + e4))
    return eps_c


def get_Deps_c_Vosko(n):
    Deps_c = np.zeros(np.shape(n))
    for i in range(len(n)):
        if n[i] != 0:
            N = n[i]
            Deps_c[i] = a * (
                -b
                * y0
                * (
                    (
                        (
                            -y0
                            + 2
                            ** (2 / 3)
                            * 3
                            ** (1 / 6)
                            * (1 / N)
                            ** (1 / 6)
                            / (2 * np.pi ** (1 / 6))
                        )
                        ** 2
                        * (
                            2
                            ** (2 / 3)
                            * 3
                            ** (1 / 6)
                            * b
                            * (1 / N)
                            ** (1 / 6)
                            / (12 * np.pi ** (1 / 6) * N)
                            + 6
                            ** (1 / 3)
                            * (1 / N)
                            ** (1 / 3)
                            / (6 * np.pi ** (1 / 3) * N)
                        )
                        / (
                            2
                            ** (2 / 3)
                            * 3
                            ** (1 / 6)
                            * b
                            * (1 / N)
                            ** (1 / 6)
                            / (2 * np.pi ** (1 / 6))
                            + c
                            + 6
                            ** (1 / 3)
                            * (1 / N)
                            ** (1 / 3)
                            / (2 * np.pi ** (1 / 3))
                        )
                        ** 2
                        - 2
                        ** (2 / 3)
                        * 3
                        ** (1 / 6)
                        * (
                            -y0
                            + 2
                            ** (2 / 3)
                            * 3
                            ** (1 / 6)
                            * (1 / N)
                            ** (1 / 6)
                            / (2 * np.pi ** (1 / 6))
                        )
                        * (1 / N)
                        ** (1 / 6)
                        / (
                            6
                            * np.pi
                            ** (1 / 6)
                            * N
                            * (
                                2
                                ** (2 / 3)
                                * 3
                                ** (1 / 6)
                                * b
                                * (1 / N)
                                ** (1 / 6)
                                / (2 * np.pi ** (1 / 6))
                                + c
                                + 6
                                ** (1 / 3)
                                * (1 / N)
                                ** (1 / 3)
                                / (2 * np.pi ** (1 / 3))
                            )
                        )
                    )
                    * (
                        2
                        ** (2 / 3)
                        * 3
                        ** (1 / 6)
                        * b
                        * (1 / N)
                        ** (1 / 6)
                        / (2 * np.pi ** (1 / 6))
                        + c
                        + 6
                        ** (1 / 3)
                        * (1 / N)
                        ** (1 / 3)
                        / (2 * np.pi ** (1 / 3))
                    )
                    / (
                        -y0
                        + 2
                        ** (2 / 3)
                        * 3
                        ** (1 / 6)
                        * (1 / N)
                        ** (1 / 6)
                        / (2 * np.pi ** (1 / 6))
                    )
                    ** 2
                    + 2
                    ** (2 / 3)
                    * 3
                    ** (1 / 6)
                    * (4 * b * y0 + 2 * b + 4 * c + 4 * y0 ** 2)
                    * (1 / N)
                    ** (1 / 6)
                    / (
                        6
                        * np.pi
                        ** (1 / 6)
                        * N
                        * (
                            1
                            + (-b ** 2 + 4 * c)
                            / (
                                b
                                + 2
                                ** (2 / 3)
                                * 3
                                ** (1 / 6)
                                * (1 / N)
                                ** (1 / 6)
                                / np.pi
                                ** (1 / 6)
                            )
                            ** 2
                        )
                        * (
                            b
                            + 2
                            ** (2 / 3)
                            * 3
                            ** (1 / 6)
                            * (1 / N)
                            ** (1 / 6)
                            / np.pi
                            ** (1 / 6)
                        )
                        ** 2
                    )
                )
                / (b * y0 + c + y0 ** 2)
                + 6
                ** (2 / 3)
                * np.pi
                ** (1 / 3)
                * (
                    6
                    ** (1 / 3)
                    * (
                        2
                        ** (2 / 3)
                        * 3
                        ** (1 / 6)
                        * b
                        * (1 / N)
                        ** (1 / 6)
                        / (12 * np.pi ** (1 / 6) * N)
                        + 6
                        ** (1 / 3)
                        * (1 / N)
                        ** (1 / 3)
                        / (6 * np.pi ** (1 / 3) * N)
                    )
                    * (1 / N)
                    ** (1 / 3)
                    / (
                        2
                        * np.pi
                        ** (1 / 3)
                        * (
                            2
                            ** (2 / 3)
                            * 3
                            ** (1 / 6)
                            * b
                            * (1 / N)
                            ** (1 / 6)
                            / (2 * np.pi ** (1 / 6))
                            + c
                            + 6
                            ** (1 / 3)
                            * (1 / N)
                            ** (1 / 3)
                            / (2 * np.pi ** (1 / 3))
                        )
                        ** 2
                    )
                    - 6
                    ** (1 / 3)
                    * (1 / N)
                    ** (1 / 3)
                    / (
                        6
                        * np.pi
                        ** (1 / 3)
                        * N
                        * (
                            2
                            ** (2 / 3)
                            * 3
                            ** (1 / 6)
                            * b
                            * (1 / N)
                            ** (1 / 6)
                            / (2 * np.pi ** (1 / 6))
                            + c
                            + 6
                            ** (1 / 3)
                            * (1 / N)
                            ** (1 / 3)
                            / (2 * np.pi ** (1 / 3))
                        )
                    )
                )
                * (
                    2
                    ** (2 / 3)
                    * 3
                    ** (1 / 6)
                    * b
                    * (1 / N)
                    ** (1 / 6)
                    / (2 * np.pi ** (1 / 6))
                    + c
                    + 6
                    ** (1 / 3)
                    * (1 / N)
                    ** (1 / 3)
                    / (2 * np.pi ** (1 / 3))
                )
                / (3 * (1 / N) ** (1 / 3))
                + 2
                ** (2 / 3)
                * 3
                ** (1 / 6)
                * b
                * (1 / N)
                ** (1 / 6)
                / (
                    3
                    * np.pi
                    ** (1 / 6)
                    * N
                    * (
                        1
                        + (-b ** 2 + 4 * c)
                        / (
                            b
                            + 2
                            ** (2 / 3)
                            * 3
                            ** (1 / 6)
                            * (1 / N)
                            ** (1 / 6)
                            / np.pi
                            ** (1 / 6)
                        )
                        ** 2
                    )
                    * (
                        b
                        + 2
                        ** (2 / 3)
                        * 3
                        ** (1 / 6)
                        * (1 / N)
                        ** (1 / 6)
                        / np.pi
                        ** (1 / 6)
                    )
                    ** 2
                )
            ) / 2

    return Deps_c


def get_Deps_c(r_s):
    Deps_c = np.zeros(np.shape(r_s))
    for i in range(0, len(r_s) - 1):
        if r_s[i] >= 1:
            Deps_c[i] = gamma * (-beta_1 / (2 * np.sqrt(r_s[i])) - beta_2) / (
                beta_1 * np.sqrt(r_s[i]) + beta_2 * r_s[i] + 1
            ) ** 2
        elif r_s[i] > 0:
            Deps_c[i] = A / (r_s[i]) + C * np.log(r_s[i]) + C + D
    return Deps_c
