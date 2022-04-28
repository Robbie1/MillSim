# -*- coding: utf-8 -*-
"""
Created in 2022

@author: Robbie
--------------------------------------------------------------------------------------------
Build up a simple milling circuit
--------------------------------------------------------------------------------------------
"""
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

import matplotlib.pyplot as plt
import matplotlib.ticker

import Mill_Model
import Sump_Model
import Cyclone_Model
import Gen_PopBal

from fpdf import FPDF

import os
import copy

class Circuit(object):
    def __init__(self, Mass_states, Water_states, Fresh_Feed_IN, Water_Mill_IN, SPD_Mill_IN, Water_Sump_IN, SPD_Sump_IN):
        Mill_M_state = Mass_states[0]
        Mill_W_state = Water_states[0]
        
        Sump_M_state = Mass_states[1]
        Sump_W_state = Water_states[1]
        
        # Create the Mill object as part of the circuit
        self.Mill = Mill_Model.Mill(Mill_M_state, Mill_W_state, Fresh_Feed_IN, Water_Mill_IN, SPD_Mill_IN)

        # Create the Sump object
        Feed_M_Sump, Feed_W_Sump = self.Mill.calc_outputs(self.Mill.M_state, self.Mill.W_state, self.Mill.SPD)
        self.Sump = Sump_Model.Sump(Sump_M_state, Sump_W_state, Feed_M_Sump, Water_Sump_IN + Feed_W_Sump, SPD_Sump_IN)

        # Create the Cyclone object as part of the circuit
        Feed_M_Cyclone, Feed_W_Cyclone = self.Sump.calc_outputs(self.Sump.M_state, self.Sump.W_state, self.Sump.SPD, self.Sump.Level_Perc)
        self.Cyclone = Cyclone_Model.Cyclone(Feed_M_Cyclone, Feed_W_Cyclone, self.Sump.Ps, 1)
                
    # DYNAMICS
    #----------------------------------
    # --- X Start ---#
    def calc_derivative_ore(self, state, t, Fresh_Feed_IN, Water_Mill_IN, SPD_Mill_IN, Water_Sump_IN, SPD_Sump_IN):
        mesh_size = len(self.Mill.mesh)
        
        # Circuit Inputs - GED
        # Mill Inputs
        Fresh_Feed_IN = np.clip(Fresh_Feed_IN, 1e-6, 1e10)
        Water_Mill_IN = np.clip(Water_Mill_IN, 1e-6, 1e10)
        SPD_Mill_IN   = np.clip(SPD_Mill_IN, 1e-6, 1e10)
        # Sump Inputs
        Water_Sump_IN = np.clip(Water_Sump_IN, 1e-6, 1e10)
        SPD_Sump_IN   = np.clip(SPD_Sump_IN, 1e-6, 1e10)

        # Circuit States
        # Mill States
        Mill_mass_state = np.clip(state[0:mesh_size], 1e-6, 1e10)
        Mill_M_tot = np.clip(np.sum(Mill_mass_state), 1e-6, 1e10)
        Mill_water_state = np.clip(state[mesh_size], 1e-6, 1e10)
        Feed_M_Sump, Feed_W_Sump = self.Mill.calc_outputs(Mill_mass_state, Mill_water_state, SPD_Mill_IN)
        # Sump States
        Sump_mass_state = np.clip(state[(mesh_size+1):-1], 1e-6, 1e10)
        Sump_M_tot = np.clip(np.sum(Sump_mass_state), 1e-6, 1e10)
        Sump_water_state = np.clip(state[-1], 1e-6, 1e10)
        Vol_ore_tmp = Sump_M_tot/self.Sump.Ps
        Vol_water_tmp = Mill_water_state/self.Sump.Pw
        TotVol_tmp = Vol_ore_tmp + Vol_water_tmp
        Sump_Perc_tmp = self.Sump.calc_level_Perc(TotVol_tmp)
        Ore_Sump_bal, W_Sump_Out = self.Sump.calc_outputs(Sump_mass_state, Sump_water_state, SPD_Sump_IN, Sump_Perc_tmp)
        # Cyclone States
        CYC_F_u_flow, CYC_W_u_flow, CYC_F_o_flow, CYC_W_o_flow = self.Cyclone.process_inputs(Ore_Sump_bal, W_Sump_Out, self.Cyclone.no_cycl)

        # Copy all the states
        y = np.copy(state)
        
        # ODE
        #-------------------------------------------------------------------------------------------
        # Mill Part
        # - Solids Part
        Ore_Mill_bal, W_Mill_Out = self.Mill.calc_ode_righthand(Mill_mass_state, Mill_water_state, SPD_Mill_IN)
        for i in range(0, len(Ore_Mill_bal)):
            y[i] =  Fresh_Feed_IN[i] + CYC_F_u_flow[i] - Ore_Mill_bal[i]
        # - Water Part
        y[mesh_size] = Water_Mill_IN + CYC_W_u_flow - W_Mill_Out
        
        # Sump Part
        #  - Solids Part
        for j in range(0, len(Ore_Sump_bal)):
            y[mesh_size + 1 + j] = Feed_M_Sump[j] - Ore_Sump_bal[j]    
        #  - Water Part
        y[-1] = (Feed_W_Sump + Water_Sump_IN)  - W_Sump_Out
        
        return y
    
    #------------------------------------------------------------------------
    ### ODE Functions ###
    #------------------------------------------------------------------------
    def ODE_solve_func(self, Fresh_Feed_IN, Water_Mill_IN, SPD_Mill_IN, Water_Sump_IN, SPD_Sump_IN, ts):
        n = len(self.Mill.mesh)
        spd_init = SPD_Sump_IN[0].flatten()
        
        for i in range(0, len(ts)-1):
            init_states = np.concatenate([self.Mill.M_state, np.array([self.Mill.W_state]), self.Sump.M_state, np.array([self.Sump.W_state])])

            
            # Normal simulation
            y = odeint(self.calc_derivative_ore, init_states, [ts[i], ts[i+1]],
                       args=( tuple( (Fresh_Feed_IN[i].flatten(), Water_Mill_IN[i].flatten(), SPD_Mill_IN[i].flatten(),
                                      Water_Sump_IN[i].flatten(), SPD_Sump_IN[i].flatten() ) ) ))
            '''
            # P Controller and ODE simulation
            spd = np.clip(spd_init + (self.Sump.Level_Perc-75)*0.25,0,100)
            y = odeint(self.calc_derivative_ore, init_states, [ts[i], ts[i+1]],
                       args=( tuple( (Fresh_Feed_IN[i].flatten(), Water_Mill_IN[i].flatten(), SPD_Mill_IN[i].flatten(),
                                      Water_Sump_IN[i].flatten(), spd_init ) ) ))
            spd_init = spd
            '''
            y = y[-1]

            # Update the variables that are dependent on the states
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            self.Mill.update_state(np.clip(y[0:n], 0, 1e10), np.clip(y[n], 0, 1e10), Fresh_Feed_IN[i].flatten(), Water_Mill_IN[i].flatten(), SPD_Mill_IN[i].flatten())
            # This is for playback of the speed inputs
            self.Sump.update_state(np.clip(y[(n+1):(2*n+1)], 1e-6, 1e10), np.clip(y[-1], 1e-6, 1e10), self.Mill.F_out, self.Mill.W_out + Water_Sump_IN[i].flatten(), SPD_Sump_IN[i].flatten())
            # Here I run the P-controller
            #self.Sump.update_state(np.clip(y[(n+1):(2*n+1)], 1e-6, 1e10), np.clip(y[-1], 1e-6, 1e10), self.Mill.F_out, self.Mill.W_out + Water_Sump_IN[i].flatten(), spd)
            
            # The cyclone streams should be added to the mill, maybe I can do this nicer - It is considered in the ODE.
            self.Cyclone.F_u_flow, self.Cyclone.W_u_flow, self.Cyclone.F_o_flow, self.Cyclone.W_o_flow = self.Cyclone.process_inputs(self.Sump.F_out, self.Sump.W_out, self.Cyclone.no_cycl)
            self.Mill.F_in = self.Mill.F_in + self.Cyclone.F_u_flow
            self.Mill.F = np.sum(self.Mill.F_in)
            self.Mill.W_in = self.Mill.W_in + self.Cyclone.W_u_flow
            self.Cyclone.F_u = np.sum(self.Cyclone.F_u_flow)
            self.Cyclone.F_o = np.sum(self.Cyclone.F_o_flow)
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # Historize the data
            self.Mill.historize_data(ts[i+1])
            self.Sump.historize_data(ts[i+1])
            self.Cyclone.historize_data(ts[i+1])

            #print(100*i/len(ts)-1)
        return 
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    ### General Functions ###
    #------------------------------------------------------------------------
    def grab_measurements(self):
        return np.concatenate([np.ravel(self.Mill.grab_measurements()),
                               np.ravel(self.Sump.grab_measurements()),
                               np.ravel(self.Cyclone.grab_measurements())])
