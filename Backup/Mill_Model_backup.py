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

import copy

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
        self.kE = self.kE_model()
        self.G = self.Discharge_model()
        
        # Mill Start States
        self.M_state = M_state # Mill Ore holdup per size clas (t)
        self.M = np.sum(M_state) # Total Ore holdup (t)
        self.M_cum = self.calc_Mcum()
        self.Water = Water_state # Total Water holdup (t)


        # Mill Variables
        # Feed
        self.F_in = F_in # Ore feed per size class (t/h)
        self.F = np.sum(F_in) # Total feed holdup (t/h)
        # Speed
        self.SPD = Alpha_spd # Speed (%)
        # Water
        self.W_in = W_in # Water (m3/h)

        
        # Mill calc states
        self.Wc = self.calc_Wc() # Ratio of Ore to (Water + Ore)
        self.JT = self.calc_JT() # Mill load (%) 
        self.Power = self.calc_Power() # Mill Power (kW)

    '''
    This must change to be an entrant property - the mass state changes
    '''
    def calc_JT(self):
        # The mill Load calcution (%)
        JT = (self.Wb/self.Pb + self.M/self.Ps + (self.Ec/(1-self.Ec))*(self.Wb/self.Pb + self.M/self.Ps) )/self.Vmill
        return JT

    '''
    This must change to be an entrant property - the mass state changes
    '''
    def calc_Power(self):
        # see pg 40 of D. Le Roux
        Power = 10.6*self.Diameter**2.5*self.Length*(1-1.03*self.JT)*(1 - 0.1/(np.power(2,9-10*self.SPD)))*self.SPD*( (1-self.Ec)*(self.Ps/self.Wc)*self.JT + 0.6*self.Jb*(self.Pb - self.Ps/self.Wc) )
        return np.clip(Power, 0 , 1e10)

    def kE_model(self):
        kE = self.breakage_params[0]*( np.divide(np.power(self.mesh, self.breakage_params[2]), 1 +
                                       np.power(self.mesh/self.breakage_params[4],self.breakage_params[5])) +
                                       self.breakage_params[1]*np.power(self.mesh, self.breakage_params[3]))
        return kE

    def Discharge_model(self):
        g = self.Gmax*( np.divide( ( np.log(self.mesh) - np.log(self.Xg) ), ( np.log(self.Xm) - np.log(self.Xg) ) ) )
        g[self.mesh <= self.Xm] = self.Gmax
        g[self.mesh > self.Xg] = 0
        return g

    '''
    This must change to be an entrant property - the mass state changes
    '''
    def calc_Wc(self):
        # Ore ratio to Ore and Water in the mill 
        Wc = self.M/(self.M + self.Water)
        return np.clip(Wc, 0, 100)

    '''
    This must change to be an entrant property - the mass state changes
    '''
    def calc_Mcum(self):
        # Matrix to describe mass fraction larger than a certain size
        n = len(self.mesh)
        temp_mat = (np.ones(n) - np.tri(n)) + np.eye(n)
        M_cum = np.asmatrix(self.M_state)*np.asmatrix(temp_mat)
        return np.asarray(M_cum).flatten()
'''
    def calc_derivative_ore(self):

        # GED
        F_in = np.clip(self.F_in, 0, 1e10)
        M_state = np.clip(self.M_state, 0, 1e10)
        M = np.clip(self.M, 0, 1e10)
        M_cum = np.clip(self.M_cum, 0, 1e10)

        W_in = np.clip(self.W_in, 0, 1e10)
        Water = np.clip(self.Water, 0, 1e10)
        
        # Mass States - Ore in the mill
        y = F_in - np.multiply(M_state, self.G)
        y[0] = y[0] - M_state[0]*(self.Power/M)*self.kE[1]
        y[1:-1] = y[1:-1] + np.multiply(M_cum[1:-1], self.kE[1:-1])*(self.Power/M) - np.multiply(M_state[1:-1] + M_cum[1:-1],self.kE[2:] )*(self.Power/M) 
        y[-1] = y[-1] + M_cum[-1]*(self.Power/M)*self.kE[-1]

        # Water in the mill (m3/h)
        w_state = W_in - self.Gmax*Water # W_in (m3/h), Gmax (1/h), Water (tons) density of water is 1 ton/m3
        y = np.concatenate([y, np.array([w_state])])

        return y
'''
    def calc_derivative_ore(self, state):

        # GED
        F_in = np.clip(self.F_in, 0, 1e10)
        M_state = np.clip(state[0:-1], 0, 1e10)
        M = np.clip(self.M, 0, 1e10)
        M_cum = np.clip(state[-1], 0, 1e10)

        W_in = np.clip(self.W_in, 0, 1e10)
        Water = np.clip(self.Water, 0, 1e10)
        
        # Mass States - Ore in the mill
        for i in range
        '''
        y = F_in - np.multiply(M_state, self.G)
        y[0] = y[0] - M_state[0]*(self.Power/M)*self.kE[1]
        y[1:-1] = y[1:-1] + np.multiply(M_cum[1:-1], self.kE[1:-1])*(self.Power/M) - np.multiply(M_state[1:-1] + M_cum[1:-1],self.kE[2:] )*(self.Power/M) 
        y[-1] = y[-1] + M_cum[-1]*(self.Power/M)*self.kE[-1]
        '''
        
        # Water in the mill (m3/h)
        w_state = W_in - self.Gmax*Water # W_in (m3/h), Gmax (1/h), Water (tons) density of water is 1 ton/m3
        y = np.concatenate([y, np.array([w_state])])

        return y
    #------------------------------------------------------------------------
    ### ODE Functions ###
    #------------------------------------------------------------------------
    def ODE_setup(self, state, t, Feed, Water, Speed):
        # Set Inputs
        self.F_in = Feed
        self.W_in  = Water
        self.SPD = Speed
        
        y = self.calc_derivative_ore(state)
        
        return y

    def ODE_solve_func(self, Feed, Water, Speed, ts):
        n = len(self.mesh)
        init_states = np.concatenate([self.M_state, np.array([self.Water])])
             
        y = odeint(self.ODE_setup, init_states, ts, args=( tuple( (Feed, Water, Speed) ) ))[-1]

        # Avoid the ODE of going to negative mass
        #y[y<0] = 0.0
        
        self.M_state = y[0:n]
        self.M = np.sum(self.M_state) # Total Ore holdup (t)
        self.M_cum = self.calc_Mcum()
        self.Water = y[n] # Total Water holdup (t)

        # Mill calc states
        self.Wc = self.calc_Wc() # Ratio of Ore to (Water + Ore)
        self.JT = self.calc_JT() # Mill load (%) 
        self.Power = self.calc_Power() # Mill Power (kW)
        return y
    #------------------------------------------------------------------------

    #------------------------------------------------------------------------
    ### Steady State Functions ###
    #------------------------------------------------------------------------
    def update_state(self, M_state, Water_state, F_in, W_in, Alpha_spd):
        # This function is only used together with the SS solution where the mill internal states can explicitly set
        
        self.M_state = M_state # Mill Ore holdup per size clas (t)
        self.M = np.sum(M_state) # Total Ore holdup (t)
        self.M_cum = self.calc_Mcum()
        self.Water = Water_state # Total Water holdup (t)


        # Mill Variables
        # Feed
        self.F_in = F_in # Ore feed per size class (t/h)
        self.F = np.sum(F_in) # Total feed holdup (t/h)
        # Speed
        self.SPD = Alpha_spd # Speed (%)
        # Water
        self.W_in = W_in # Water (m3/h)

        
        # Mill calc states
        self.Wc = self.calc_Wc() # Ratio of Ore to (Water + Ore)    -> f(M, Water)
        self.JT = self.calc_JT() # Mill load (%)                    -> f(M)
        self.Power = self.calc_Power() # Mill Power (kW)            -> f(JT, SPD, Wc)
        return

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
        X = np.append(self.M_state, self.Water)
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
    Gmax = 27.5 # Specific discharge rate for water and fines
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
