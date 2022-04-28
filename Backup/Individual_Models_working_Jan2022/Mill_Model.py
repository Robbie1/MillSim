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

class Mill(object):
    def __init__(self, M_state, Water_state, F_in, W_in, Alpha_spd):
        # Parameters
        # M_state - Mass holdup for each sieve class (tons)
        # Water_state - Water holdup (tons)
        # F_in - Feed of ore (t/h)
        # W_in - Water into the mill (m3/h)
        # Alpha_spd - Mill Speed (%)
        #---------------------------------------------------------
    
        # Mill Parameters
        # Physical Dimensions and certain holdup parameters
        self.Diameter, self.Length, self.Vmill, self.Xg, self.Xm, self.Jb, self.Pb, self.Wb, self.Ps, self.Ec, self.Gmax = LeRoux_milldimensions()
        # Mesh setup
        self.mesh = hinde_mesh()
        
        # Model parameters
        self.breakage_params = hinde_breakageparams()
        
        
        # Mill Start States
        self.M_state = M_state # Mill Ore holdup per size clas (t)
        self.M = np.sum(M_state) # Total Ore holdup (t)
        self.M_cum = self.calc_Mcum_ent(M_state) # Total amount of Ore bigger as a certain size
        self.W_state = Water_state # Total Water holdup (t)

        # Mill Variables
        # Feed
        self.F_in = F_in # Ore feed per size class (t/h)
        self.F = np.sum(F_in) # Total feed holdup (t/h)
        # Speed
        self.SPD = Alpha_spd # Speed (%)
        # Water
        self.W_in = W_in # Water (m3/h)
      
        # Mill calc states
        self.Wc = self.calc_Wc_ent(self.M, self.W_state) # Ratio of Ore to (Water + Ore)
        self.JT = self.calc_JT_ent(self.M) # Mill load (%) 
        self.Power = self.calc_Power_Hulbert(self.JT, self.M/self.W_state/self.Ps, Alpha_spd) # Mill Power (kW)

        # Breakage parameters - State dependent
        self.kE = self.kE_model(self.JT, self.SPD)
        self.G = self.Discharge_model(self.SPD)
        
        # Mill outputs
        self.F_out, self.W_out = self.calc_outputs(self.M_state, self.W_state, self.SPD)
         
        self.init_history()
        self.historize_data(0)

    # --- Load Calc Start ---#
    def calc_JT_ent(self, M):
        # Load eq. 3.13 pg 40
        # JT = (Wb/Pb + M/Ps + (Ec/(1-Ec))(Wb/Pb + M/Ps) )/Vmill
        #-------------------------------------------------------
        # STATE DEP #
        # M     - mass of ore [sum Mstate]
        # CONSTANTS #
        # Wb    - mass of balls [Constant for now]
        # Ec    - Porosity of charge [Constant for now]
        # Vmill - Volume of mill [Constant]
        # Pb    - Density of balls [Constant for now]
        # Ps    - Density of solids [Constant for now]
        
        JT = (self.Wb/self.Pb + M/self.Ps + (self.Ec/(1-self.Ec))*(self.Wb/self.Pb + M/self.Ps) )/self.Vmill
        return np.clip(JT, 0, 1)
    # --- Load Calc END ---#

    # --- Power Calc Start ---#
    def calc_Power_ent(self, JT, Wc, AlphaSPD):
        # see pg 40 of D. Le Roux
        # P = 10.6(D**2.5)L(1 - 1.03JT)(1 - (0.1/(2**(9 - 10AlphaSPD))) )AlpaSPD[(1-Ec)(Ps/Wc)JT + 0.6Jb(Pb - Ps/Wc)]
        #------------------------------------------------------------------------------------------------------------
        # STATE DEP #
        # JT        - Load as a volume frac [calc_JT or calc_JT_ent]
        # Wc        - Ore/(Ore + Water) [calc_Wc]
        # INPUTVAR #
        # AlphaSPD  - Speed as % critical spd [InputVAR]
        # CONSTANTS #
        # Jb        - Fractional filling balls [Constant for now]
        # Ec        - Porosity of charge [Constant for now]
        # Pb        - Density of balls [Constant for now]
        
        #Power = 10.6*self.Diameter**2.5*self.Length*(1-1.03*JT)*(1 - 0.1/(np.power(2,9-10*AlphaSPD)))*AlphaSPD*( (1-self.Ec)*(self.Ps/Wc)*JT + 0.6*self.Jb*(self.Pb - self.Ps/Wc) )
        Power = np.multiply( 10.6*self.Diameter**2.5*self.Length*(1-1.03*JT), np.multiply( (1 - 0.1/(np.power(2,9-10*AlphaSPD))),np.multiply(AlphaSPD,( (1-self.Ec)*(self.Ps/Wc)*JT + 0.6*self.Jb*(self.Pb - self.Ps/Wc)))))
        return np.clip(Power, 0 , 1e10)

    def calc_Power_Hulbert(self, JT, Wrat, AlphaSPD):
        Esp_0 = 0.6
        Rho_cond = np.copy(Wrat)
        Rho_cond[Rho_cond > 1/(1/Esp_0 - 1)] = 0
        Rho = np.sqrt(1 - (1/Esp_0 - 1)*Rho_cond)
        
        Rho_N = 0.509
        dv = 0.928
        ds = 0.928
        
        JT_pmax = -7.52*np.power(AlphaSPD, 2) + 9.06*AlphaSPD - 2.18
        Pmax = (-2.7*np.power(AlphaSPD, 2) + 3.92*AlphaSPD - 1.2)*1e4

        Power = np.multiply( 1 - dv*np.power(np.divide(JT, JT_pmax) - 1,2) - ds*np.power(np.divide(Rho, Rho_N) - 1,2), Pmax)
        return np.clip(Power, 0 , 1e10)
    # --- Power Calc END ---#

    # --- Wc Calc Start ---#
    def calc_Wc_ent(self, M, W_state):
        # Ore ratio to Ore and Water in the mill 
        Wc = M/(M + W_state)
        return np.clip(Wc, 0, 100)
    # --- Wc Calc END ---#
    
    # --- Mcum Calc Start ---#
    def calc_Mcum_ent(self, M_state):
        # Wi (t) is the mass of material coarser than size xi inside the mill
        # Definition introduced on pg 37
        # -------------------------------------------------------------------
        # STATE DEP #
        # M_state    - Mass states for each size class
        # CONSTANTS #
        # Mesh       - The sieve sizes for each size fraction [Constant]
        
        n = len(self.mesh)
        temp_mat = (np.ones(n) - np.tri(n)) + np.eye(n)
        M_cum = np.asmatrix(M_state)*np.asmatrix(temp_mat)
        M_cum = np.concatenate( [np.zeros(1), np.asarray(M_cum).flatten()[0:-1] ])
        
        return np.clip( M_cum, 0, 1e10)
    # --- Mcum Calc END ---#

    # --- Mcum Calc Outputs Start ---#
    def calc_outputs(self, M_state, W_state, SPD):
        # Calculate the output streams from the mill
        G_update = self.Discharge_model(SPD)
        
        # WATER Output
        F_out = np.clip( np.multiply(G_update, M_state), 0, 1e10)
                
        # WATER Output
        W_out = np.clip( G_update[-1]*np.max(W_state,0) , 0, 1e10)
        return F_out, W_out
    # --- Mcum Calc Outputs END ---#
    
    # MODEL PARAMATERS
    #----------------------------------
    # --- Breakage functions Start ---#
    def kE_model(self, JT, AlphaSPD):
        alhpa1_param = 0.5*self.breakage_params[2]*(0.6607*AlphaSPD + 0.5511) + 0.5*self.breakage_params[2]*( -0.7207*JT + 1.2523)
        alhpa2_param = 0.5*self.breakage_params[3]*(0.5229*AlphaSPD + 0.6471) + 0.5*self.breakage_params[3]
        mu_param =  0.5*self.breakage_params[4]*(-2.8283*AlphaSPD + 2.9091) + 0.5*self.breakage_params[4]
        k1_param = 0.5*self.breakage_params[0]  + 0.5*self.breakage_params[0]*(-0.885*(JT**2) + 0.4425*JT + 0.9535)
        k2_param = 0.5*self.breakage_params[1]*(-2.222*AlphaSPD + 2.5) + 0.5*self.breakage_params[1]
        lambda_param = 0.5*self.breakage_params[5] + 0.5*self.breakage_params[5]
        
        kE = k1_param*( np.divide(np.power(self.mesh, alhpa1_param), 1 +
                                       np.power(self.mesh/mu_param,lambda_param)) +
                                       k2_param*np.power(self.mesh, alhpa2_param))
        kE[0] = None
        return kE

    def Discharge_model(self, AlphaSPD):
        adapt_g = np.clip(-1.111*AlphaSPD**2 + 2.5556*AlphaSPD -0.2444, 0.1, 1.2)
        g = adapt_g*self.Gmax*( np.divide( ( np.log(self.mesh) - np.log(self.Xg) ), ( np.log(self.Xm) - np.log(self.Xg) ) ) )
        g[self.mesh <= self.Xm] = adapt_g*self.Gmax
        g[self.mesh > self.Xg] = 0
        return g
    # --- Breakage functions END ---#

    # DYNAMICS
    #----------------------------------
    # --- X Start ---#
    def calc_derivative_ore(self, state, t, F, W, SPD):
        # GED Ore Inputs & States
        F = np.clip(F, 0, 1e10)

        M_state_tmp = np.clip(state[0:-1], 1e-6, 1e10)
        M_tmp = np.clip(np.sum(M_state_tmp), 1e-6, 1e10)

        # GED Water Inputs & States
        W = np.clip(W, 0, 1e10)
        
        W_state_tmp = np.clip(state[-1], 1e-6, 1e10)
      
        # Mass States - Ore in the mill
        y = np.copy(state)
        # Calculated States
        M_cum_tmp = self.calc_Mcum_ent(M_state_tmp)
        Wc_tmp = self.calc_Wc_ent(M_tmp, W_state_tmp)
        JT_tmp = self.calc_JT_ent(M_tmp)
        #Power_tmp = self.calc_Power_ent(JT_tmp, Wc_tmp, SPD)
        Power_tmp = self.calc_Power_Hulbert(JT_tmp, M_tmp/W_state_tmp/self.Ps, SPD)

        # Update Model related "Functions"
        G_update = self.Discharge_model(SPD)
        kE_update = self.kE_model(JT_tmp, SPD)
        
        # DERIVATIVES - ORE
        tmp_check = 0
        for i in range(0, len(self.mesh)):
            if i == 0:
                # The first size fraction eq 3.1 on pg 24
                # dw1/dt = f1 - g1w1 - w1(P/M)kE2
                y[0] =  F[0] - G_update[0]*np.max(state[0], 0) - np.max(state[0], 0)*(Power_tmp/M_tmp)*kE_update[1]
                tmp_check = tmp_check - np.max(state[0], 0)*(Power_tmp/M_tmp)*kE_update[1]
                
            elif (i > len(self.mesh)-2):
                y[i] =  F[i] - G_update[i]*np.max(state[i], 0) +  M_cum_tmp[i]*(Power_tmp/M_tmp)*kE_update[i]
                tmp_check = tmp_check +  M_cum_tmp[i]*(Power_tmp/M_tmp)*kE_update[i]
                
            else:
                # The first size fraction eq 3.4 on pg 26
                # dwi/dt = fi - giwi - [Mcumi(P/M)kEi  - (wi + Mcumi)(P/M)kEi+1]
                y[i] =  F[i] - G_update[i]*np.max(state[i], 0) + ( np.max(M_cum_tmp[i],0)*(Power_tmp/M_tmp)*kE_update[i] - np.max(state[i] + M_cum_tmp[i],0)*(Power_tmp/M_tmp)*kE_update[i+1] )
                tmp_check = tmp_check + ( np.max(M_cum_tmp[i],0)*(Power_tmp/M_tmp)*kE_update[i] - np.max(state[i] + M_cum_tmp[i],0)*(Power_tmp/M_tmp)*kE_update[i+1] ) 

        if tmp_check > 1e-2:
            print(tmp_check, 'Mass balance problem')
            
        # DERIVATIVES - WATER
        y[-1] = W - G_update[-1]*np.max(state[-1],0) # W_in (m3/h), Gmax (1/h), Water (tons) density of water is 1 ton/m3

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
            
            self.update_state(np.clip(y[0:n], 0, 1e10), np.clip(y[n], 0, 1e10), Feed_In[i].flatten(), Water_In[i].flatten(), Speed_In[i].flatten())
            self.historize_data(ts[i+1])
        return 
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    ### General Functions ###
    #------------------------------------------------------------------------
    def update_state(self, M_state, Water_state, F_in, W_in, Alpha_spd):
        # This function is only used together with the SS solution where the mill internal states can explicitly set
        
        self.M_state = M_state # Mill Ore holdup per size clas (t)
        self.M = np.sum(self.M_state) # Total Ore holdup (t)
        self.M_cum = self.calc_Mcum_ent(self.M_state)
        self.W_state = Water_state # Total Water holdup (t)


        # Mill Variables
        # Feed
        self.F_in = F_in # Ore feed per size class (t/h)
        self.F = np.sum(F_in) # Total feed holdup (t/h)
        # Speed
        self.SPD = Alpha_spd # Speed (%)
        # Water
        self.W_in = W_in # Water (m3/h)

        
        # Mill calc states
        self.Wc = self.calc_Wc_ent(self.M, self.W_state)              # Ratio of Ore to (Water + Ore)    -> f(M, Water)
        self.JT = self.calc_JT_ent(self.M)                            # Mill load (%)                    -> f(M)
        #self.Power = self.calc_Power_Hulbert(self.JT, self.Wc, self.SPD)  # Mill Power (kW)            -> f(JT, Wc, SPD)
        self.Power = self.calc_Power_Hulbert(self.JT, self.M/self.W_state/self.Ps, self.SPD)  # Mill Power (kW)            -> f(JT, Wc, SPD)

        self.F_out, self.W_out = self.calc_outputs(self.M_state, self.W_state, self.SPD)
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
        self.HIST_Wc = [] # Ratio of Ore to (Water + Ore)
        self.HIST_JT = [] # Mill load (%) 
        self.HIST_Power = [] # Mill Power (kW)
        self.HIST_F_out = [] # Output Ore
        self.HIST_W_out = [] # Output water
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
        self.HIST_Wc.append(self.Wc) # Ratio of Ore to (Water + Ore)
        self.HIST_JT.append(self.JT) # Mill load (%) 
        self.HIST_Power.append(self.Power) # Mill Power (kW)
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
        
        # PSD - Feed & Product
        name = 'IN_OUT_STATE_PSD.png'
        self.plot_func_2ax([[Gen_PopBal.get_pval_array(80,self.mesh, self.HIST_F_in)],
                [np.matrix(Gen_PopBal.get_cumdist_array(self.HIST_F_out))[:,-1],
                np.matrix(Gen_PopBal.get_cumdist_array(self.HIST_M_state))[:,-1]]],
               [['F80'], ['P75um', 'S75um']],[['b'],['k', 'm']])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # TPH (WATER) - Feed & Prod
        name = 'Water_IN_OUT_TPH.png'
        self.plot_func([self.HIST_W_in, self.HIST_W_out],['Water IN (tph)', 'Water OUT (tph)'],['b','k'])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # Load, SPD, Power
        name = 'Load_SPD_Power.png'
        self.plot_func_2ax([[self.HIST_JT, self.HIST_SPD], [np.ravel(self.HIST_Power)]],[['Load','SPD'],['Power']],[['r','b'],['g']])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # States - Ore, Water, Wc
        name = 'Mill_States.png'
        self.plot_func_2ax([[self.HIST_M, self.HIST_W_state], [np.ravel(self.HIST_Wc)]],[['Ore State','Water State'],['Wc']],[['r','b'],['g']])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)
        
        pdf.output(os.getcwd() + '\\Reports\\' +'SimReport.pdf', 'F')
        return

    def plot_millmap(self):
        pdf = FPDF()
        pdf.add_page()
        
        # Discharge function
        name = 'Discharge.png'
        plt.figure(figsize=(global_fig_width, global_fig_height), dpi=80)
        plt.semilogx(self.mesh, self.G)
        plt.grid()
        plt.savefig(os.getcwd() + '\\MillMap_Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\MillMap_Trends\\' + name,w=pdf_W, h=pdf_H)
        
        # kE Breakage function
        name = 'kE_breakage.png'
        plt.figure(figsize=(global_fig_width, global_fig_height), dpi=80)
        plt.loglog(self.mesh, self.kE)
        plt.grid()
        plt.savefig(os.getcwd() + '\\MillMap_Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\MillMap_Trends\\' + name,w=pdf_W, h=pdf_H)
                  
        # JT
        name = 'JT.png'
        plt.figure(figsize=(global_fig_width, global_fig_height), dpi=80)
        m = np.linspace(0,110,20)
        plt.plot(m, self.calc_JT_ent(m))
        plt.grid()
        plt.savefig(os.getcwd() + '\\MillMap_Trends\\' + name)
        plt.close()
        pdf.image(os.getcwd() + '\\MillMap_Trends\\' + name,w=pdf_W, h=pdf_H)
                  
        # Mesh Wc
        name = 'Wc.png'
        m = np.arange(1, 110, 1)
        w = np.arange(1, 110, 1)
        MM, WW = np.meshgrid(m, w)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(int(global_fig_width/3), int(global_fig_width/3)))
        ax.plot_surface(MM, WW, self.calc_Wc_ent(MM, WW))
        ax.view_init(elev=35., azim=-135)
        plt.savefig(os.getcwd() + '\\MillMap_Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\MillMap_Trends\\' + name,w=int(pdf_W/3), h=int(pdf_W/3))
                  
        # Mesh Power
        name = 'Power.png'
        jt = np.linspace(0.2, 0.6, 25)
        wc = np.linspace(0.2, 1.5, 25)
        spd = np.linspace(0.45, 0.9, 25)
        JT, WC = np.meshgrid(jt, wc)
        JT, SPD = np.meshgrid(jt, spd)
        fig, ax = plt.subplots(nrows=1,ncols=2,subplot_kw={"projection": "3d"},figsize=(global_fig_width, global_fig_height))
                
        ax[0].plot_surface(JT, WC, self.calc_Power_Hulbert(JT, WC, 0.65), color='g', alpha=0.5)
        ax[0].plot_surface(JT, WC, self.calc_Power_Hulbert(JT, WC, 0.7), color='b', alpha=0.5)
        ax[0].plot_surface(JT, WC, self.calc_Power_Hulbert(JT, WC, 0.75), color='r', alpha=0.5)
        ax[0].set_xlabel('Load')
        ax[0].set_ylabel('Wc')
        ax[0].set_zlabel('Power')
        ax[0].view_init(elev=35., azim=-135)

        ax[1].plot_surface(JT, SPD, self.calc_Power_Hulbert(JT, 0.5, SPD), color='g', alpha=0.5)
        ax[1].plot_surface(JT, SPD, self.calc_Power_Hulbert(JT, 0.7, SPD), color='b', alpha=0.5)
        ax[1].plot_surface(JT, SPD, self.calc_Power_Hulbert(JT, 1.0, SPD), color='r', alpha=0.5)
        ax[1].set_xlabel('Load')
        ax[1].set_ylabel('SPD')
        ax[1].set_zlabel('Power')
        ax[1].view_init(elev=35., azim=-135)
        plt.savefig(os.getcwd() + '\\MillMap_Trends\\' + name) 
                
        plt.close()
        pdf.image(os.getcwd() + '\\MillMap_Trends\\' + name,w=pdf_W, h=pdf_H)

        pdf.output(os.getcwd() + '\\Reports\\' +'MillMap.pdf', 'F')
        return
    
    #------------------------------------------------------------------------
    
    #------------------------------------------------------------------------
    ### Steady State Functions ###
    #------------------------------------------------------------------------
    def der_obj_func(self, X):
        # Objective function to solve the SS, set "calc_derivative_ore" to zero
        n = len(self.mesh)
        
        M_state = X[0:n]            # Mass in the mill - n number of states
        Water_state = X[n]          # Water in the mill - 1 state
        F_in = X[(n+1):(2*n+1)]     # Mass feed to the mill - n number of inputs
        W_in = X[2*n+1]             # Water feed to the mill - 1 input
        Alpha_spd = X[2*n+2]        # Speed of the mill - 1 input

        # Update the state of the mill object
        self.update_state(M_state, Water_state, F_in, W_in, Alpha_spd)

        # Solve the derivative calc, set it to zero
        y = np.sum(np.power(self.calc_derivative_ore(),2))
        return y
    
    def find_ss_solution(self):
        # Solve the derivate to zero and find a SS solution
        X = np.append(self.M_state, self.W_state)
        X = np.append(X, self.F_in)
        X = np.append(X, self.W_in)
        X = np.append(X, self.SPD)

        # Ensure all non negative
        linear_constraint = LinearConstraint(np.eye(len(X)), np.zeros(len(X)), np.ones(len(X))*np.inf )
        
        y = minimize( self.der_obj_func, X, constraints=linear_constraint)
        return y
    #------------------------------------------------------------------------
    
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
def LeRoux_milldimensions():
    # Mill physical dimensions
    Diameter = 4.2 #m
    Length = 4.27 #m
    Vmill = 59.12 #m3 - mill volume

    # Mill grate properties
    Xg = 12 #mm - Effective mesh size for zero discharge
    Xm = 1 #mm - Maximum fine size in discharge rate function
    
    # Ball related properties
    Jb = 0.24 #Static fractional volumetric filling of the mill for balls
    Pb = 7.85 #t/m3 - ball density
    Wb = 66.8 # t - Mass of balls in the mill

    # Ore properties
    Ps = 3.2 # t/m3 - Ore density

    # Other
    Ec = 0.3 # dimensionless [0 - 1] - Charge porosity
    Gmax = 2.75 # Specific discharge rate for water and fines
    return Diameter, Length, Vmill, Xg, Xm, Jb, Pb, Wb, Ps, Ec, Gmax

def hinde_mesh():
    # see pg 42 of D. Le Roux thesis
    #   - all sizes in mm
    size = np.array([307.2, 217.2, 153.6, 108.6, 76.8,
            54.3, 38.4, 27.2, 19.2, 13.6,
            9.6, 6.8, 4.8, 3.4, 2.4, 
            1.7, 1.2, 0.85, 0.6, 0.42,
            0.30, 0.21, 0.15, 0.106, 0.075])
    return size

def hinde_breakageparams():
    # Ke function is described on pg 39, eq 2.10
    # see pg 43 of D. Le Roux thesis for the parameter values
    # Kappa1, Kappa2, Alpha1, Alpha2, Lambda, Mu, Beta, D62.2 - These are parameters to be fitted
    params = np.array([1.13, 3e-6, 1.11, 2.55, 0.33, 1.16, 0.36, 41.1])
    return params
