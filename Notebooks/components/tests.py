#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  tests.py
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
from components.kohnsham import solve_ks
from components.misc import get_n
import time


def test_helium(rr, psi, tol, max_iter, verbose):
    print(
        "Calculating helium wave function and ground state energy without xc..."
    )
    Z = 2
    iteration_number = 0
    E = 0
    E_prev = E + 1
    T0 = time.clock()
    while abs(E - E_prev) > tol and iteration_number < max_iter:
        iteration_number += 1
        if verbose:
            print("####### Iteration", iteration_number, "#######")
        n = get_n(psi)
        V_sH = get_hartree_potential(rr, n)
        V_H = 1 * V_sH
        E_prev = E
        E, psi = solve_ks(rr, Z, V_H)
        if verbose:
            print("E =", E, "[Ha]")
            print("ΔE =", E - E_prev, "[Ha]")
            print("Time =", time.clock() - T0, "s")
            print()
    return psi, E


def test_helium_x(rr, psi, tol, max_iter, verbose):
    print("Calculating helium wave function and ground state energy with x...")
    Z = 2
    iteration_number = 0
    E = 0
    E_prev = E + 1
    T0 = time.clock()
    while abs(E - E_prev) > tol and iteration_number < max_iter:
        iteration_number += 1
        if verbose:
            print("####### Iteration", iteration_number, "#######")
        n = get_n(psi)
        V_sH = get_hartree_potential(rr, n)
        V_H = 2 * V_sH
        V_xc = get_V_x(n)
        E_prev = E
        E, psi = solve_ks(rr, Z, V_H, V_xc)
        if verbose:
            print("E =", E, "[Ha]")
            print("ΔE =", E - E_prev, "[Ha]")
            print("Time =", time.clock() - T0, "s")
            print()
    return psi, E


def test_helium_xc(rr, psi, tol, max_iter, verbose):
    print(
        "Calculating helium wave function and ground state energy with xc-Perdew..."
    )
    Z = 2
    iteration_number = 0
    E = 0
    E_prev = E + 1
    T0 = time.clock()
    while abs(E - E_prev) > tol and iteration_number < max_iter:
        iteration_number += 1
        if verbose:
            print("####### Iteration", iteration_number, "#######")
        n = get_n(psi)
        V_sH = get_hartree_potential(rr, n)
        V_H = 2 * V_sH
        V_xc = get_V_xc(n)
        E_prev = E
        E, psi = solve_ks(rr, Z, V_H, V_xc)
        if verbose:
            print("E =", E, "[Ha]")
            print("ΔE =", E - E_prev, "[Ha]")
            print("Time =", time.clock() - T0, "s")
            print()
    return psi, E


def test_helium_xc_Vosko(rr, psi, tol, max_iter, verbose):
    print(
        "Calculating helium wave function and ground state energy with xc-Vosko..."
    )
    Z = 2
    iteration_number = 0
    E = 0
    E_prev = E + 1
    E_vec = []
    T0 = time.clock()
    while abs(E - E_prev) > tol and iteration_number < max_iter:
        iteration_number += 1
        if verbose:
            print("####### Iteration", iteration_number, "#######")
        n = get_n(psi)
        V_sH = get_hartree_potential(rr, n)
        V_H = 2 * V_sH
        V_xc = get_V_xc_Vosko(n)
        E_prev = E
        E, psi = solve_ks(rr, Z, V_H, V_xc)
        E_vec.append(E)
        if verbose:
            print("E =", E, "[Ha]")
            print("ΔE =", E - E_prev, "[Ha]")
            print("Time =", time.clock() - T0, "s")
            print()
    return psi, E, E_vec
