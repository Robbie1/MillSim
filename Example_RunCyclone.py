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
import Cyclone_Model

import Gen_PopBal

# Input Variables
beta = 2.5 
d63  = 0.2 
F_in = 40*Gen_PopBal.create_fracdist(Cyclone_Model.hinde_mesh(), beta, d63)
W_in = 60

# Create Cyclone Object
Cyclone = Cyclone_Model.Cyclone(F_in, W_in, 1.3, 1)
'''
Cyclone.historize_data(2)
Cyclone.historize_data(3)
Cyclone.plot_save()
'''
print('Q = ', Cyclone.Q_var)
print('p_p = ', Cyclone.p_p_var)
print('Lambda = ', Cyclone.labmda_par)
print('Cv = ', Cyclone.C_v_var)
print('P = ', Cyclone.P)
print('d = ', Cyclone.d50_c/1000)
print('R_f = ', Cyclone.R_f)
print('R_v = ', Cyclone.R_v)
print(np.sum(Cyclone.F_u_flow)/np.sum(F_in))
print( Cyclone.W_u_flow/W_in)

plt.subplot(211)
plt.semilogx(Cyclone.mesh, Cyclone.eff_curve)

plt.subplot(212)
plt.semilogx(Cyclone.mesh, Gen_PopBal.get_cumdist(np.ravel(F_in)), color='g', label='Inlet')
plt.semilogx(Cyclone.mesh, Gen_PopBal.get_cumdist(np.ravel(Cyclone.F_u_flow)), color='b', label='Coarse')
plt.semilogx(Cyclone.mesh, Gen_PopBal.get_cumdist(np.ravel(Cyclone.F_o_flow)), color='r', label='Fine')
plt.legend()
plt.show()


