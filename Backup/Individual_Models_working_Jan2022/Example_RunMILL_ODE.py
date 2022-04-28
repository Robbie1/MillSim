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

# Initial state
M_state = 20*Gen_PopBal.create_fracdist(Mill_Model.hinde_mesh(), 0.25, 10)[0]

W_state = 20

# Time of simulation
HRS = 5
t_sim = np.linspace(0, HRS, int(HRS*60+1))

# Input Variables
beta = (0.5 + np.random.rand(len(t_sim))*0.0)
d63 = (30 + np.random.rand(len(t_sim))*0.0)
#F_in = (50.0 + np.random.rand(len(t_sim))*0.0)[:,None]*Gen_PopBal.create_fracdist(Mill_Model.hinde_mesh(), beta, d63)
tmp = 25*np.ones(len(t_sim))
tmp[60:] = 0
F_in = (50.0 + tmp)[:,None]*Gen_PopBal.create_fracdist(Mill_Model.hinde_mesh(), beta, d63)

W_in = np.repeat(27.5, len(t_sim)) + np.random.rand(len(t_sim))*0.0
W_in[120:] = 22.5

Alpha_spd = np.repeat(0.7, len(t_sim)) + np.random.rand(len(t_sim))*0.0
Alpha_spd[240:] = 0.65
Alpha_spd[270:] = 0.75

# Create Mill Object
Mill = Mill_Model.Mill(M_state, W_state, F_in[0].flatten(), W_in[0].flatten(), Alpha_spd[0].flatten())


# Run the ODE
Mill.ODE_solve_func(F_in, W_in, Alpha_spd, t_sim)

#--------------------------------------------------------------------------------------------
# Plotting of data
#--------------------------------------------------------------------------------------------
Mill.plot_save()

'''
jt = np.linspace(0.2, 0.9, 15)
wc = np.linspace(0.2, 1.5, 15)
spd = np.linspace(0.4, 0.9, 15)
JT, WC = np.meshgrid(jt, wc)
JT, SPD = np.meshgrid(jt, spd)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(5, 5))
                
ax.plot_surface(JT, WC, Mill.calc_Power_Hulbert(JT, WC, 0.7), color='g', alpha=0.25)
ax.scatter(np.array(Mill.HIST_JT), np.divide(np.array(Mill.HIST_M), np.array(Mill.HIST_W_state))/Mill.Ps, Mill.HIST_Power, marker='.')
ax.scatter(np.array(Mill.HIST_JT)[0], np.divide(np.array(Mill.HIST_M), np.array(Mill.HIST_W_state))[0]/Mill.Ps, Mill.HIST_Power[0], color='r')
ax.scatter(np.array(Mill.HIST_JT)[-1], np.divide(np.array(Mill.HIST_M), np.array(Mill.HIST_W_state))[-1]/Mill.Ps, Mill.HIST_Power[-1], color='k', marker='x')
ax.set_xlabel('LOAD')
ax.set_ylabel('Wc')
plt.show()
'''
