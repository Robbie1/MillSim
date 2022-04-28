# -*- coding: utf-8 -*-
"""
Created in 2022

@author: Robbie
--------------------------------------------------------------------------------------------
Environment used to run the simulation
--------------------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import Mill_Model
import Sump_Model
import Cyclone_Model
import Circuit_Model

import Gen_PopBal

#------------------------
# General Simulation Settings
#------------------------
# Time of simulation
HRS = 5
t_sim = np.linspace(0, HRS, int(HRS*60+1))

#------------------------
# Mill Settings
#------------------------
# Initial state
Mill_M_state = 20*Gen_PopBal.create_fracdist(Mill_Model.hinde_mesh(), 0.25, 10)[0]
Mill_W_state = 20

# Input Variables
beta = (0.5 + np.random.rand(len(t_sim))*0.1)
d63 = (30 + np.random.rand(len(t_sim))*0.5)
tmp = 5*np.ones(len(t_sim)) + np.random.rand(len(t_sim))*2.5
tmp[0:50] = 0
Mill_F_in = (20.0 + tmp)[:,None]*Gen_PopBal.create_fracdist(Mill_Model.hinde_mesh(), beta, d63)

Mill_W_in = np.repeat(5, len(t_sim)) + np.random.rand(len(t_sim))*0.0
Mill_W_in[150:] = 6

Mill_spd = np.repeat(0.65, len(t_sim)) + np.random.rand(len(t_sim))*0.0
Mill_spd[50:] = 0.6
#------------------------
# SUMP Settings
#------------------------
# Initial state
Sump_M_state = 3*Gen_PopBal.create_fracdist(Sump_Model.hinde_mesh(), 0.25, 10)[0]
Sump_W_state = 1

# Input Variables
Sump_W_in = np.repeat(1, len(t_sim)) + np.random.rand(len(t_sim))*0.0
Sump_W_in[240:] = 0.8

Sump_spd = np.repeat(95, len(t_sim)) + np.random.rand(len(t_sim))*0.0
Sump_spd[60:] = 70
Sump_spd[120:] = 95

#------------------------
# Circuit Initialization
#------------------------
Circuit = Circuit_Model.Circuit([Mill_M_state, Sump_M_state], [Mill_W_state, Sump_W_state],
                                Mill_F_in[0].flatten(), Mill_W_in[0].flatten(), Mill_spd[0].flatten(),
                                Sump_W_in[0].flatten(), Sump_spd[0].flatten())


#------------------------
# Run the ODE of the circuit
#------------------------
#Circuit.ODE_solve_func(Mill_F_in, Mill_W_in, Mill_spd, Sump_W_in, Sump_spd, t_sim)

#------------------------
# Run the ODE iteratively
#------------------------
for i in range(len(t_sim)-2):
    print(type(Mill_F_in[i:(i+2),:]), type(Mill_W_in[i:i+2]), type(Mill_spd[i:i+2]), type(Sump_W_in[i:i+2]), type(Sump_spd[i:i+2]), type(t_sim[i:i+2]) )
    print(np.shape(Mill_F_in[i:(i+2),:]), np.shape(Mill_W_in[i:i+2]), np.shape(Mill_spd[i:i+2]), np.shape(Sump_W_in[i:i+2]), np.shape(Sump_spd[i:i+2]), np.shape(t_sim[i:i+2]))
    Circuit.ODE_solve_func(Mill_F_in[i:(i+2),:], Mill_W_in[i:i+2], Mill_spd[i:i+2], Sump_W_in[i:i+2], Sump_spd[i:i+2], t_sim[i:i+2])

Circuit.Mill.plot_save()
Circuit.Sump.plot_save()
Circuit.Cyclone.plot_save()

