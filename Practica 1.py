# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:18:54 2021

@author: José Lopez
"""

#Practica 1
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

#1.1
M = np.array([[3,-9,0,5],
     [2, -5, -3, 1],
     [-1, 5, 8, 4]])

A = np.array([[-1, 1, 2]])

B = np.array([[-4, 2, 1, -1]])

#A*M
np.dot(A, M)

#A.T * B
np.dot(A.T, B)

#M*B,T
np.dot(M,B.T)

#A*A.T
np.dot(A,A.T)

#1.2
    #1    
x = np.linspace(-2,2,100)

f1 = 1/(1+np.exp(-x))
f2 = math.tanh(x)
f3 = math.sign
f4 = f1 * (1 - f1)
f5 = 1 - f2**2

funcs = [f1, f2, f3, f4, f5]