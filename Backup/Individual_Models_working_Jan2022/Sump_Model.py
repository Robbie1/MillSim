# -*- coding: utf-8 -*-
"""
Created in 2021

@author: Robbie
"""
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

import matplotlib.pyplot as plt
import matplotlib.ticker

from fpdf import FPDF

import Gen_PopBal # Custom module for General Population Balance functions 

import os
import copy

# GLOBAL PARAMETERS
global_fig_width = 10
global_fig_height = 2.5
pdf_W = global_fig_width*20
pdf_H = global_fig_height*20

class Sump(object):
    def __init__(self, M_state, Water_state, F_in, W_in, Alpha_spd):
        # Parameters
        # M_state - Mass holdup for each sieve class (tons)
        # Water_state - Water holdup (tons)
        # F_in - Feed of ore (t/h)
        # W_in - Water into the mill (m3/h)
        # Alpha_spd - Mill Speed (%)
        #---------------------------------------------------------
    
        # Sump Parameters
        # Physical Dimensions and certain holdup parameters
        self.Diameter, self.Length, self.Vsump, self.Ps, self.Pw = sump_dimensions()

        self.k_flow = 0.5
        
        # Mesh setup
        self.mesh = hinde_mesh()       
        
        # Sump Start States
        self.M_state = M_state # Sump Ore holdup per size clas (t)
        self.M = np.sum(M_state) # Total Ore holdup (t)
        self.Vol_ore = self.M/self.Ps # (m3)
        
        self.W_state = Water_state # Total Water holdup (t)
        self.Vol_water = self.W_state/self.Pw # (m3)

        self.TotVol = self.Vol_ore + self.Vol_water
        self.Level_Perc = self.calc_level_Perc(self.TotVol)
        
        self.P_comb = self.calc_density(self.M, self.W_state)

        # Sump Variables
        # Feed
        self.F_in = F_in # Ore feed per size class (t/h)
        self.F = np.sum(F_in) # Total feed holdup (t/h)
        
        # Speed
        self.SPD = Alpha_spd # Speed (frac)
        # Water
        self.W_in = W_in # Water (m3/h)
             
        # Mill outputs
        self.F_out, self.W_out = self.calc_outputs(self.M_state, self.W_state, self.SPD, self.Level_Perc)

        self.init_history()
        self.historize_data(0)

    # --- calc_density Start ---#
    def calc_density(self, M, W_state):
        # density in the sump 
        P_comb = (M + W_state)/(np.clip(M/self.Ps,1e-6,1e10) + np.clip(W_state,1e-6,1e10)/self.Pw)
        return np.clip(P_comb, 1e-6, self.Ps)
    # --- calc_density END ---#

    def calc_level_Perc(self, TotVol):
        Level_Perc = 100*(self.TotVol/(np.pi*( (self.Diameter/2)**2 )) )/self.Length
        return np.clip(Level_Perc, 1e-6, 100)
    
    def calc_level_flow_inf(self, Level_Perc):
        a = -9e-5
        b = 0.0202
        c = 0.029
        adjust_factor = a*Level_Perc**2 + b*Level_Perc + c
        return np.clip(adjust_factor, 0.1, -b/(2*a))
    
    # --- Mcum Calc Outputs Start ---#
    def calc_outputs(self, M_state, W_state, SPD, Sump_Perc):
        # Calculate the output streams from the Sump

        Tot_Flow_Out = np.clip(self.k_flow*SPD*self.calc_level_flow_inf(Sump_Perc), 1e-6, 1e10)

        Vol_Frac = 1/( np.sum( M_state/self.Ps ) + W_state/self.Pw )
                
        # ORE Output
        F_out = np.clip( self.calc_density(np.sum(M_state), W_state)*Tot_Flow_Out*Vol_Frac*(M_state/self.Ps), 1e-6, 1e10)
        #F_out = np.clip( self.calc_density(np.sum(M_state), W_state)*Tot_Flow_Out*Vol_Frac*np.sum( M_state/self.Ps ), 0, 1e10)
        #F_out = F_out*(M_state/np.sum( M_state ))
        
        # WATER Output
        W_out = np.clip( self.calc_density(np.sum(M_state), W_state)*Tot_Flow_Out*Vol_Frac*(W_state/self.Pw), 1e-6, 1e10)

        return F_out, W_out
    # --- Mcum Calc Outputs END ---#
    
    # DYNAMICS
    #----------------------------------
    # --- X Start ---#
    def calc_derivative_ore(self, state, t, F, W, SPD):
        # GED Ore Inputs & States
        F = np.clip(F, 1e-6, 1e10)

        M_state_tmp = np.clip(state[0:-1], 1e-6, 1e10)
        M_tmp = np.clip(np.sum(M_state_tmp), 1e-6, 1e10)

        # GED Water Inputs & States
        W = np.clip(W, 1e-6, 1e10)
        W_state_tmp = np.clip(state[-1], 1e-6, 1e10)
      
        # Mass States - Ore in the mill
        y = np.copy(state)

        # Update States
        Vol_ore_tmp = M_tmp/self.Ps
        Vol_water_tmp = W_state_tmp/self.Pw
        TotVol_tmp = Vol_ore_tmp + Vol_water_tmp
        Sump_Perc_tmp = self.calc_level_Perc(TotVol_tmp)
        
        Ore_out, Water_Out = self.calc_outputs(M_state_tmp, W_state_tmp, SPD, Sump_Perc_tmp)
        
        # DERIVATIVES - ORE
        tmp_check = 0
        for i in range(0, len(self.mesh)):
            y[i] =  F[i] - Ore_out[i]                
            
        # DERIVATIVES - WATER
        y[-1] = W - Water_Out

        return y
    
    #------------------------------------------------------------------------
    ### ODE Functions ###
    #------------------------------------------------------------------------
    def ODE_solve_func(self, Feed_In, Water_In, Speed_In, ts):
        n = len(self.mesh)

        for i in range(0, len(ts)-1):
            init_states = np.concatenate([self.M_state, np.array([self.W_state])])

            y = odeint(self.calc_derivative_ore, init_states, [ts[i], ts[i+1]], args=( tuple( (Feed_In[i].flatten(), Water_In[i].flatten(), Speed_In[i].flatten()) ) ))
            y = y[-1]
            
            self.update_state(np.clip(y[0:n], 1e-6, 1e10), np.clip(y[n], 1e-6, 1e10), Feed_In[i].flatten(), Water_In[i].flatten(), Speed_In[i].flatten())
            self.historize_data(ts[i+1])
        return 
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    ### General Functions ###
    #------------------------------------------------------------------------
    def update_state(self, M_state, Water_state, F_in, W_in, Alpha_spd):
        # This function is only used together with the SS solution where the mill internal states can explicitly set
        
        # Sump Start States
        self.M_state = M_state # Sump Ore holdup per size clas (t)
        self.M = np.sum(M_state) # Total Ore holdup (t)
        self.Vol_ore = self.M/self.Ps # (m3)
        
        self.W_state = Water_state # Total Water holdup (t)
        self.Vol_water = self.W_state/self.Pw # (m3)

        self.TotVol = self.Vol_ore + self.Vol_water
        self.Level_Perc = self.calc_level_Perc(self.TotVol)
        
        self.P_comb = self.calc_density(self.M, self.W_state)

        # Sump Variables
        # Feed
        self.F_in = F_in # Ore feed per size class (t/h)
        self.F = np.sum(F_in) # Total feed holdup (t/h)
        
        # Speed
        self.SPD = Alpha_spd # Speed (frac)
        # Water
        self.W_in = W_in # Water (m3/h)
             
        # Mill outputs
        self.F_out, self.W_out = self.calc_outputs(self.M_state, self.W_state, self.SPD, self.Level_Perc)
        
        return

    def init_history(self):
        self.HIST_TIME = []
        
        # Mill Start States
        self.HIST_M_state = [] # Mill Ore holdup per size clas (t)
        self.HIST_M = [] # Total Ore holdup (t)
        self.HIST_W_state = [] # Total Water holdup (t)
        self.HIST_F_in = [] # Ore feed per size class (t/h)
        self.HIST_F = [] # Total feed holdup (t/h)
        self.HIST_SPD = [] # Speed (%)
        self.HIST_W_in = [] # Water (m3/h)
        self.HIST_Level_Perc = []
        self.HIST_dens = []
        
        self.HIST_F_out = []
        self.HIST_W_out = []
        return
    
    def historize_data(self, ts):
        self.HIST_TIME.append(ts)
        # Mill Start States
        self.HIST_M_state.append(self.M_state) # Mill Ore holdup per size clas (t) 
        self.HIST_M.append(self.M) # Total Ore holdup (t)
        self.HIST_W_state.append(self.W_state) # Total Water holdup (t)
        self.HIST_F_in.append(self.F_in) # Ore feed per size class (t/h)
        self.HIST_F.append(self.F) # Total feed holdup (t/h)
        self.HIST_SPD.append(self.SPD) # Speed (%)
        self.HIST_W_in.append(self.W_in) # Water (m3/h)
        self.HIST_Level_Perc.append(self.Level_Perc)
        self.HIST_dens.append(self.P_comb)
        self.HIST_F_out.append(self.F_out)
        self.HIST_W_out.append(self.W_out)
        return
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    ### Plotting functions ###
    #------------------------------------------------------------------------
    def plot_func(self, plot_data, plot_name, plot_color):
        plt.figure(figsize=(global_fig_width, global_fig_height), dpi=80)
        ax = plt.subplot(111)
        for i in range(0, len(plot_data)):
            ax.plot(self.HIST_TIME, plot_data[i], color=plot_color[i], label=plot_name[i])
        ax.set_xlim([self.HIST_TIME[0], self.HIST_TIME[-1]])
        ax.grid()
        ax.legend()
        return

    def plot_func_2ax(self, plot_data, plot_name, plot_color):
        plt.figure(figsize=(global_fig_width, global_fig_height), dpi=80)
        ax = plt.subplot(111)
        for i in range(0, len(plot_data[0])):
            ax.plot(self.HIST_TIME, plot_data[0][i], color=plot_color[0][i], label=plot_name[0][i])
        ax.set_xlim([self.HIST_TIME[0], self.HIST_TIME[-1]])
        ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(6))
        ax.grid()
        ax.legend(loc=2)
        
        ax1 = ax.twinx()
        for i in range(0, len(plot_data[1])):
            ax1.plot(self.HIST_TIME, plot_data[1][i], color=plot_color[1][i], label=plot_name[1][i])
        ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(6))
        ax1.grid(None)
        ax1.legend(loc=1)              
        return

    def plot_save(self):
        pdf = FPDF()
        pdf.add_page()
        
        # TPH (ORE) - Feed & Prod
        name = 'Ore_IN_OUT_TPH.png'
        self.plot_func([self.HIST_F, np.sum(self.HIST_F_out, axis=1)],['Feed (tph)', 'Prod. (tph)'],['b','k'])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # TPH (WATER) - Feed & Prod
        name = 'Water_IN_OUT_TPH.png'
        self.plot_func([self.HIST_W_in, self.HIST_W_out],['Water IN (tph)', 'Water OUT (tph)'],['b','k'])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # TPH Mass balance
        name = 'Total_IN_OUT_TPH.png'
        self.plot_func([np.array(self.HIST_F).flatten() + np.array(self.HIST_W_in).flatten(),
                        np.array(self.HIST_W_out).flatten() + np.sum(self.HIST_F_out, axis=1).flatten()],['IN (tph)', 'OUT (tph)'],['b','k'])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # States - Ore, Water, Wc
        name = 'Sump_States.png'
        self.plot_func_2ax([[self.HIST_M, self.HIST_W_state], [np.ravel(self.HIST_dens)]],[['Ore State','Water State'],['Wc']],[['r','b'],['g']])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # Level, SPD
        name = 'LVL.png'
        self.plot_func_2ax([[self.HIST_Level_Perc], [self.HIST_SPD]],[['Load'],['SPD']],[['r'],['g']])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # PSD - Feed & Product
        name = 'IN_OUT_STATE_PSD.png'
        self.plot_func([Gen_PopBal.get_pval_array(80,self.mesh, self.HIST_F_in),
                        Gen_PopBal.get_pval_array(80,self.mesh, self.HIST_F_out),
                        Gen_PopBal.get_pval_array(80,self.mesh, self.HIST_M_state)],
               ['F80', 'P80', 'S80'],['b', 'k', 'm'])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        pdf.output(os.getcwd() + '\\Reports\\' +'SimReport_Sump.pdf', 'F')
        return

###--- Basic Functionality ---###
#-------------------------------#
def RR(X, Beta, D63_2):
    eps = X/X[0]
    n = np.divide(eps, (1 - 0.999*eps))
    n63_2 = ( D63_2/X[0] )/( 1 - D63_2/X[0] )

    R = 1 - np.exp( -1*(np.divide(n, n63_2)**Beta) )
    return R

###--- These are all model specific parameters ---###
#---------------------------------------------------#
def sump_dimensions():
    # Mill physical dimensions
    Diameter = 1.75 #m
    Length = 2.5 #m
    Vsump = np.pi*( (Diameter/2)**2 )*Length #m3 - sump volume
    
    # Ore properties
    Ps = 3.2 # t/m3 - Ore density
    Pw = 1.0 # t/m3 - Water density

    return Diameter, Length, Vsump, Ps, Pw

def hinde_mesh():
    # see pg 42 of D. Le Roux thesis
    #   - all sizes in mm
    size = np.array([307.2, 217.2, 153.6, 108.6, 76.8,
            54.3, 38.4, 27.2, 19.2, 13.6,
            9.6, 6.8, 4.8, 3.4, 2.4, 
            1.7, 1.2, 0.85, 0.6, 0.42,
            0.30, 0.21, 0.15, 0.106, 0.075])
    return size
