# -*- coding: utf-8 -*-
"""
Created in 2021

@author: Robbie
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t, inp):
    k = 27
    if y > 0:
        dydt = inp -k * y
    else:
        dydt = inp
    return dydt

# initial condition
y0 = 10

# time points
t = np.linspace(0,1)
inp = 300.0
# solve ODE
y = odeint(model,y0,t, args=( inp,) )

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
