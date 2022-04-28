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
M_state = 30*Gen_PopBal.create_fracdist(Mill_Model.hinde_mesh(), 0.25, 10)[0]

W_state = 10

# Time of simulation
HRS = 3
t_sim = np.linspace(0, HRS, int(HRS*60+1))

# Input Variables
beta = (0.5 + np.random.rand(len(t_sim))*0.0)
d63 = (30 + np.random.rand(len(t_sim))*0.0)
F_in = (80.0 + np.random.rand(len(t_sim))*0.0)[:,None]*Gen_PopBal.create_fracdist(Mill_Model.hinde_mesh(), beta, d63)

W_in = np.repeat(27.5, len(t_sim)) + np.random.rand(len(t_sim))*0.0

Alpha_spd = np.repeat(0.7, len(t_sim)) + np.random.rand(len(t_sim))*0.0

# Create Mill Object
Mill = Mill_Model.Mill(M_state, W_state, F_in[0].flatten(), W_in[0].flatten(), Alpha_spd[0].flatten())

Mill.plot_millmap()
