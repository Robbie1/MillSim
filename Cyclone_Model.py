# -*- coding: utf-8 -*-
"""
Created in 2022

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

class Cyclone(object):
    def __init__(self, F_in, W_in, p_s = 3.21, n=5):
        # INPUTS
        # Q_var     - (m3/h) FLOW
        # p_p_var   - (t/m3) Pulp density
        # p_s       - (t/m3) Solids density
        self.mesh = hinde_mesh()

        # Set the cyclone design parameters - They are fixed
        self.design = design_vars()

        # Set parameters and variables
        # Parameters
        self.p_s = p_s        
        # Variables
        self.F_in    = F_in
        self.F       = np.sum(F_in)
        self.W_in    = W_in
        self.Q_var   = calc_vol(self.F, W_in, self.p_s, 1)
        self.p_p_var = calc_density(self.F, W_in, self.p_s, 1) 
        self.no_cycl = n
        
        self.calc_variable_params()

        self.F_u_flow, self.W_u_flow, self.F_o_flow, self.W_o_flow = self.process_inputs(self.F_in, self.W_in, self.no_cycl)
        self.F_u = np.sum(self.F_u_flow)
        self.F_o = np.sum(self.F_o_flow)

        self.init_history()
        self.historize_data(0)

    def process_inputs(self, F_in, W_in, no_cycl):
        # Make sure everything is updated
        self.F_in    = F_in
        self.F       = np.sum(F_in)
        self.W_in    = W_in
        self.Q_var   = calc_vol(self.F, W_in, self.p_s, 1)
        self.p_p_var = calc_density(self.F, W_in, self.p_s, 1) 
        self.no_cycl = no_cycl
        self.calc_variable_params()

        W_Feed = np.clip(self.Q_var*(1-self.C_v_var),1e-6, 1e9) # Feed water in m3/h
        W_u_flow = np.clip((self.R_f/100)*W_Feed,1e-6, 1e9)     # Water to underflow m3/h
        W_o_flow = np.clip(W_Feed - W_u_flow,1e-6, 1e9)         # Water to overflow in m3/h

        #rec_curve = self.eff_curve + (self.R_f/100)*(1-self.eff_curve)
        F_u_flow = np.clip(F_in*self.eff_curve, 1e-6, 1e9)  # Underflow tph - Coarse
        F_o_flow = np.clip(F_in-F_u_flow, 1e-6, 1e9)        # Overflow tph - Fine
        
        return F_u_flow, W_u_flow, F_o_flow, W_o_flow

    def calc_variable_params(self):
        # Model Parameters
        self.C_v_var     = np.clip((self.p_p_var - 1)/(self.p_s - 1), 0, 1)      # (Fraction)    - Vol fraction of solids in slurry [f(pulp_density)] 
        self.labmda_par  = (10**(1.82*self.C_v_var) )/(8.05*(1-self.C_v_var)**2) # (-)           - Hindered settling correction term

        # Key performance calcs
        self.P = self.calc_pressure(self.Q_var, self.p_p_var, self.no_cycl)
        self.d50_c = self.calc_d50(self.Q_var, self.p_p_var, self.P, self.labmda_par)
        self.R_f = self.calc_Rf(self.Q_var, self.p_p_var, self.P, self.labmda_par)
        self.R_v = self.calc_Rv(self.Q_var, self.p_p_var, self.P, self.labmda_par)
        self.eff_curve = self.eff_curve_lynch(self.d50_c/1000)
    

#----------------------
# Performance Equations
#----------------------
    def calc_pressure(self, Q, p_p_var, n):
        KQ_o = self.design.KQ_o
        D_c = self.design.D_c
        D_o = self.design.D_o
        D_i = self.design.D_i
        L_c = self.design.L_c
        theta_par = self.design.theta_par
        
        design_part = KQ_o*( D_c**(-0.1) )*( (D_o/D_c)**0.68 )*( (D_i/D_c)**0.45 )*( (L_c/D_c)**0.2 )*( theta_par**(-0.1) )
        P =  p_p_var*( (Q/n)/((D_c**2)*design_part) )**2
        return P

    def calc_d50(self, Q, p_p_var, P, labmda_par):
        KD_o = self.design.KD_o
        D_c = self.design.D_c
        D_u = self.design.D_u
        D_o = self.design.D_o
        D_i = self.design.D_i
        L_c = self.design.L_c
        g = self.design.grav_par
        theta_par = self.design.theta_par
        
        d50_c = D_c*KD_o*( D_c**(-0.65) )*( (D_o/D_c)**0.52 )*( ( D_u/D_c)**(-0.50) )*( (D_i/D_c)**0.2 )*( (L_c/D_c)**0.2 )*( theta_par**0.15 )*( (P/(p_p_var*g*D_c) )**(-0.22) )* (labmda_par**0.93)
        return d50_c

    def calc_Rf(self, Q, p_p_var, P, labmda_par):
        KW_o = self.design.KW_o
        D_c = self.design.D_c
        D_u = self.design.D_u
        D_o = self.design.D_o
        D_i = self.design.D_i
        L_c = self.design.L_c
        g = self.design.grav_par
        theta_par = self.design.theta_par
        
        R_f = KW_o*( (D_o/D_c)**(-1.19) )*( ( D_u/D_c)**(2.4) )*( (D_i/D_c)**(-0.5))*( (L_c/D_c)**0.22 )*( theta_par**(-0.24) )*( (P/(p_p_var*g*D_c) )**(-0.53) )* (labmda_par**0.27)
        return np.clip(R_f, 0, 100)

    def calc_Rv(self, Q, p_p_var, P, labmda_par):
        KV_o = self.design.KV_o
        D_c = self.design.D_c
        D_u = self.design.D_u
        D_o = self.design.D_o
        D_i = self.design.D_i
        L_c = self.design.L_c
        g = self.design.grav_par
        theta_par = self.design.theta_par
        
        R_v = KV_o*( (D_o/D_c)**(-0.94) )*( ( D_u/D_c)**(1.83) )*( (D_i/D_c)**(-0.25))*( (L_c/D_c)**0.22 )*( theta_par**(-0.24) )*( (P/(p_p_var*g*D_c) )**(-0.31) )
        return np.clip(R_v, 0, 100)

    def eff_curve_lynch(self, d50_c):
        alpha_eff = self.design.alpha_eff
        mesh = self.mesh
        
        y_eff = ( np.exp(np.clip(alpha_eff*(mesh/d50_c), 0, 60)) - 1 )/( np.exp(np.clip(alpha_eff*(mesh/d50_c), 0, 60)) + np.exp(alpha_eff) - 2 )
        y_eff = np.clip(y_eff, 0., 1.0)
        return y_eff

    def grab_measurements(self):
        return np.array([self.P])
    
    def init_history(self):
        self.HIST_TIME = []
        
        # Mill Start States
        self.HIST_F_in  = [] # Ore feed per size class (t/h)
        self.HIST_F     = [] # Total feed holdup (t/h)
        self.HIST_W_in  = [] # Water (m3/h)
        self.HIST_Q_var      = []
        self.HIST_p_p_var    = []
        self.HIST_no_cycl    = []
        self.HIST_d50_c      = []
        self.HIST_P          = []
        self.HIST_F_u_flow   = []
        self.HIST_W_u_flow   = []
        self.HIST_F_o_flow   = []
        self.HIST_W_o_flow   = []
        self.HIST_F_u = []
        self.HIST_F_o = []
        return
    
    def historize_data(self, ts):
        self.HIST_TIME.append(ts)

        self.HIST_F_in.append(np.ravel(self.F_in))
        self.HIST_F.append(self.F) 
        self.HIST_W_in.append(self.W_in)
        self.HIST_Q_var.append(self.Q_var)
        self.HIST_p_p_var.append(self.p_p_var)
        self.HIST_no_cycl.append(self.no_cycl)
        self.HIST_d50_c.append(self.d50_c)
        self.HIST_P.append(self.P)
        self.HIST_F_u_flow.append(np.ravel(self.F_u_flow))
        self.HIST_W_u_flow.append(self.W_u_flow)
        self.HIST_F_o_flow.append(np.ravel(self.F_o_flow))
        self.HIST_W_o_flow.append(self.W_o_flow)
        self.HIST_F_u.append(self.F_u)
        self.HIST_F_o.append(self.F_o)
        return


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

        # Feed flow and density
        name = 'Cyclone_Flow_dens.png'
        self.plot_func_2ax([[self.HIST_Q_var, np.array(self.HIST_P)],[self.HIST_p_p_var]],
               [['Flow', 'P'],['Dens']],[['r', 'b'],['g']])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)
        
        # TPH (ORE) - Feed & Prod
        name = 'Solids_Cyclone.png'
        self.plot_func([self.HIST_F, self.HIST_F_u, self.HIST_F_o, np.array(self.HIST_F_u) + np.array(self.HIST_F_o)],['Feed_F (tph)', 'U/F_F (tph)', 'O/F_F (tph)', 'Check'],['b','k', 'g', 'y'])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)

        # TPH (WATER) - Feed & Prod
        name = 'Liquid_Cyclone.png'
        self.plot_func([self.HIST_W_in, self.HIST_W_u_flow, self.HIST_W_o_flow, np.array(self.HIST_W_u_flow) + np.array(self.HIST_W_o_flow)],['Feed_W (tph)', 'U/F_W (tph)', 'O/F_W (tph)', 'Check'],['b','k', 'g', 'y'])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)
                
        # PSD - Feed & Product
        name = 'Cyclone_PSD.png'
        self.plot_func([Gen_PopBal.get_pval_array(80,self.mesh, self.HIST_F_in),
                Gen_PopBal.get_pval_array(80,self.mesh, self.HIST_F_u_flow),
                 Gen_PopBal.get_pval_array(80,self.mesh, self.HIST_F_o_flow)],
               ['F80', 'u75um', 'o75um'],['b','k', 'm'])
        plt.savefig(os.getcwd() + '\\Trends\\' + name) 
        plt.close()
        pdf.image(os.getcwd() + '\\Trends\\' + name,w=pdf_W, h=pdf_H)
        
        pdf.output(os.getcwd() + '\\Reports\\' +'SimReport_Cyclone.pdf', 'F')
        return

        
#----------------------
# Equipment Design Variables
#----------------------     
class design_vars(object):
    def __init__(self):
        # Material dependent parameters
        self.KQ_o = 185 # Flow Constant
        self.KD_o = 150    # D50 constant
        self.KW_o = 1062  # Water split constant
        self.KV_o = 573.1 # Water split constant
        self.alpha_eff = 2.5 # Efficiency curve sharpness parameter

        # Cyclone Design Parameters
        self.D_i         = 0.196 # (m)       - Inlet diameter 
        self.D_o         = 0.224 # (m)       - Vortex finder diameter 
        self.D_u         = 0.11  # (m)       - Apex diameter 
        self.D_c         = 0.5   # (m)       - Cyclone cylindrical diameter
        self.L_c         = 1.6   # (m)       - Length of cylindrical section
        self.theta_par   = 16    # (Degrees) - Cone full angle
        self.grav_par    = 9.81   # (m/s2)    - Gravity constant

def hinde_mesh():
    # see pg 42 of D. Le Roux thesis
    #   - all sizes in mm
    size = np.array([307.2, 217.2, 153.6, 108.6, 76.8,
            54.3, 38.4, 27.2, 19.2, 13.6,
            9.6, 6.8, 4.8, 3.4, 2.4, 
            1.7, 1.2, 0.85, 0.6, 0.42,
            0.30, 0.21, 0.15, 0.106, 0.075])
    return size
        
#----------------------
# General based calcs
#----------------------     
def calc_density(F_In, W_In, p_s, p_w):
    return (F_In + W_In)/calc_vol(F_In, W_In, p_s, p_w)

def calc_vol(F_In, W_In, p_s, p_w):
    return F_In/p_s + W_In/p_w
        
        
        
        
