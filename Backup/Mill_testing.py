# -*- coding: utf-8 -*-
"""
Created in 2021

@author: Robbie
--------------------------------------------------------------------------------------------
Environment used to run the simulation
--------------------------------------------------------------------------------------------
"""

"""
To Do List
 DYNAMIC MODE
 - Test the system for mass balance coherency
 
 STEADY STATE
 - Water Balance not considered in the SS solution
 - Find SS solution for different conditions, i.e. fix the mill inlet and find the SS Feed, Speed, WaterIn
"""

import numpy as np
import matplotlib.pyplot as plt
import Mill_Model

import Gen_PopBal

M_rr = Mill_Model.RR(Mill_Model.hinde_mesh(), 0.4, 40)

# Initial state
M_state = np.zeros(len(M_rr))
M_state[0:-1] = M_rr[0:-1] - M_rr[1:]
M_state[-1] = M_rr[-1]
M_state = 35*M_state

W_state = 10

# Time of simulation
t_sim = np.linspace(0, 1, 61)

# Input Variables
F_in = np.zeros(len(M_rr)) 
F_in = 80*M_state/np.sum(M_state)
F_in = np.tile(F_in, (len(t_sim), 1))

W_in = 25
W_in = np.tile(W_in, (len(t_sim), 1))

Alpha_spd = 0.7
Alpha_spd = np.tile(Alpha_spd, (len(t_sim), 1))

# Create Mill Object
Mill = Mill_Model.Mill(M_state, W_state, F_in[0].flatten(), W_in[0].flatten(), Alpha_spd[0].flatten())

# ODE
temp = Mill.calc_derivative_ore(np.concatenate([Mill.M_state, np.array([Mill.W_state])]), [0,1], F_in[0].flatten(), W_in[0].flatten(), Alpha_spd[0].flatten())

#print(Mill.M_state)
#print(Mill.W_state)
#print(Mill.M)

y = Mill.ODE_solve_func(F_in, W_in, Alpha_spd, t_sim)

print(Mill.M_state)
print(Mill.W_state)
print(Mill.M)




'''
Backup to run a SS simulation
--------------------------------------------------------------------------------------------
# Find SS solution
res = Mill.find_ss_solution()

print('Initial Mill State', np.sum(M_state), ' SS Mill State', np.sum(Mill.M_state))
print('Initial Feed State', np.sum(F_in), ' SS Feed State', np.sum(Mill.F_in))
print('Initial Watr State', W_in, ' SS Watr State', Mill.W_in)
print('SS Power', Mill.Power, 'SS Load', Mill.JT, 'SS Wc', Mill.Wc)

plt.subplot(211)
plt.semilogx(Mill.mesh, np.cumsum(F_in[::-1])[::-1]/np.sum(F_in), 'r')
plt.semilogx(Mill.mesh, np.cumsum(Mill.F_in[::-1])[::-1]/np.sum(Mill.F_in), 'r--')

plt.semilogx(Mill.mesh, np.cumsum(M_state[::-1])[::-1]/np.sum(M_state), 'b')
plt.semilogx(Mill.mesh, np.cumsum(Mill.M_state[::-1])[::-1]/np.sum(Mill.M_state), 'b--')

plt.subplot(212)
plt.semilogx(Mill.mesh, F_in, 'r')
plt.semilogx(Mill.mesh, Mill.F_in, 'r--')

plt.semilogx(Mill.mesh, M_state, 'b')
plt.semilogx(Mill.mesh, Mill.M_state, 'b--')

plt.show()
'''

