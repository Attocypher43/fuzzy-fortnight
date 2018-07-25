#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:53:11 2018

@author: robertweinbaum
"""
## TO DO LIST:
## 0. Resolution Plot.
## 1. Why is a symmetric ∆E working?
## 2. Collection efficiency profile?
## 3. I/O Electron Energy Spectrum

## Import Libraries
import numpy as np
import time as timeit
import pylab as plt
from scipy.interpolate import interp1d

## THINGS TO ADD:
## 1. DYNAMICS RANGE
## 2. Better resolution curve plots


## BONJOUR!!
## Begin code timer

start_time = timeit.time()

## Fundamental Constants

m = 9.10938356e-31
q = -1.60217662e-19
eV = 1.60217662e-19

## Spectrometer Geometry ##
##   (all in SI units)   ##

junction_width =  2.0e-3
start_junction_position = 10e-2
end_junction_position = start_junction_position + junction_width
R_pre_flight_tube = 10e-2
R_drift_tube = 40e-2
length = 2.
source_size = 100e-6
MCP_time_resolution = 5e-9

## Electron Parameters ##

E_min = 1*eV                        ## Minimum electron energy

E_max = 100*eV                      ## Maximum electron energy

num_of_energies = 100               ## Number of electron energies 
                                    ##(equally spaced from minimum to maximum)
                                    
cone_angle = 8*np.pi/180.           ## Electron source cone angle (in radians)

num_of_angles = 2                  ## Number of electron initial angles
                                    ##(equally spaced from 0 to cone angle)

n = num_of_angles*num_of_energies   ## Total number of electrons to fly

cone_angle = 8*np.pi/180.           ## Electron source cone angle


## Geneartes a grid of electron energies and initial angles
electron_energies,electron_azimuth = np.meshgrid(np.linspace(E_min,E_max,num_of_energies),
                                                 np.linspace(0,cone_angle,num_of_angles))

## Unpacks the 2D array above into two 1D arrays    
electron_energies = np.ravel(electron_energies, order = 'F')
electron_azimuth = np.ravel(electron_azimuth, order = 'F')

## Simulation Options ##
discretize_voltage = False          ## If True, makes voltage profile discrete
voltage_time_step = 10e-12          ## Update time step for voltage

voltage_start_time = start_junction_position/np.sqrt(2*E_max/m)
voltage_end_time = 1e-6             ## Large enough for all electrons to exit
                                    ## the junction

number_of_junction_steps = 5000     ## Number of steps inside junction

## Parameters for voltage profile ##
##       (all in SI units)       ##
                                    
alpha = -10.1/eV*1e-9*0
beta = 1e-6

## Plotting Options

trajectory_plot = True
trajectory_inside_junction = True
voltage_plot = False
collection_efficiency_plot = False
TOF_plot = False
time_vs_position_plot = True



def run_flight(mode = 'TDRP'):   
    
## Runs an electron flight with either the natural voltage (mode = 'natural') 
## or the time-dependent retarding voltage (mode = 'TDRP'). If 'mode' is not
## specific, then TDRP will be default. Mode is case-sensitive!
    
    ## Define Voltage Profile
    
    if mode == 'TDRP':
        def V(t):
            l_0 = start_junction_position
            l_1 = length - end_junction_position
            E = l_0**2/t**2 * m/2
            
            T = alpha*E + beta
            
            
            U = m/2/q*(l_1**2/(T - t)**2 - l_0**2/t**2)
        
            return U
        
    elif mode == 'natural':
        
        def V(t):
            
            return np.zeros_like(t)
    
        
    else:
        raise ValueError("Mode is not well-defined")
        

    ## Changes continuous voltage profile into a discrete one ##
    
    if discretize_voltage == True:
        voltage_times = np.arange(voltage_start_time,voltage_end_time,voltage_time_step)
        V = interp1d(voltage_times,V(voltage_times),kind = 'nearest')
    
    ## Interaction Region:
    ## This implements a fourth-order Runge-Kutta technique to solve for
    ## electron trajectories within the junction
        
    def interaction_region(t0,x,y,z,vx,vy,vz):
                
        # Initial position #
        x0 = x
        # Distance step #
        dx = junction_width/number_of_junction_steps
        # Vector array of position and x-velocity#
        r = np.vstack((x0,vx))
        # Initial Time #
        t = np.copy(t0)
        initial_time = np.copy(t0)
        
        # Value of advance is 1 if the electron is still in the junction, and 
        # is 0 if not.
        advance = np.ones_like(x)        
        
        # Describes the second order differential equations of motion as two 
        # coupled first order differential equations.

        def f(r,t):
            vx = r[1,:]
            
            fx = vx
            fv = q*V(t)/m/junction_width
                              
            return np.vstack((fx,fv))
        
        if trajectory_inside_junction == True:
            ## Creates arrays to store positions inside junction
            x_inside_junction = np.zeros((n,number_of_junction_steps+2)) 
            y_inside_junction = np.zeros((n,number_of_junction_steps+2)) 
            z_inside_junction = np.zeros((n,number_of_junction_steps+2)) 
            t_inside_junction = np.zeros((n,number_of_junction_steps+2))
            
            ## Starts indexing steps inside junction and initializes starting positions
            i = 0
            x_inside_junction[:,0] = np.copy(r[0,:])
            y_inside_junction[:,i] = np.copy(y)
            z_inside_junction[:,i] = np.copy(z)
            t_inside_junction[:,i] = np.copy(t0)
        ## Runs while (a) electrons are still in the junction and
        ## (b) all electrons in junction have positive velocity
        while min(r[0,:]) < end_junction_position and max(r[1,:]*advance) > 0.0:
            ## calculate which electrons are in the junction and have positive velocity
            ## using a heaviside step function (https://en.wikipedia.org/wiki/Heaviside_step_function)
            
            advance = np.heaviside(end_junction_position - r[0,:],0)*np.heaviside(r[1,:],0)
            ## Choose time step such that electron will only advance a distance dx
            dt = dx/r[1,:]*advance
            ## Run 4th order Runge-Kutta algorithm
            k1 = dt*f(r,t)            
            k2 = dt*f(r + 0.5*k1, t + 0.5*dt)
            k3 = dt*f(r + 0.5*k2, t + 0.5*dt)
            k4 = dt*f(r  + k3, t + dt)
            
            ## Update the position/velocity array
            r += 1/6*(k1 + 2*k2 + 2*k3 +k4) * np.vstack((advance,advance))
                                    
            
            if trajectory_inside_junction == True:
                
            ## Store positions inside junction
                i += 1
                x_inside_junction[:,i] = np.copy(r[0,:])
                y_inside_junction[:,i] = y_inside_junction[:,i-1] + vy * dt
                z_inside_junction[:,i] = z_inside_junction[:,i-1] + vz * dt
                t_inside_junction[:,i] = 1*t
                
            ## Update time                
            t += dt            
                
        ## Velocities are 0 if negative and maintain value if positive
        ## This gives electrons with insufficient kinetic energy a ToF
        ## that is infinite.            
        
        r[1,:] *= np.heaviside(r[1,:],0)
        
        ## Calculate time spent inside the junction
        
        time_in_junction = t - initial_time
        
        
        ## Return time in junction, final positions, final velocities
        ## (and x,y,z trajectorie through junction if trajectory_inside_junction == True)
        if trajectory_inside_junction == True:
            return time_in_junction,r[0,:],r[1,:],x_inside_junction,y_inside_junction,z_inside_junction,t_inside_junction
        
        else:
            return time_in_junction,r[0,:],r[1,:]
        
    ## By cylindrical symmetry, we can put the electrons in the x-y plane.
    electron_elevation = np.pi/2  
    
    ## Calulcate total speed from electron energy
    electron_speed = np.sqrt(2*electron_energies/m)
    
    ## Initialize position vector
    x_0 = 0.0
    y_0 = 0.0
    z_0 = 0.0
    
    ## Initial initial positions
    x_initial = np.zeros(n) + x_0
    y_initial = np.zeros(n) + y_0
    z_initial = np.zeros(n) + z_0
    
    ## Initialize velocity vector using spherical coordinates
    v_x_pre_flight_tube = electron_speed*np.sin(electron_elevation)*np.cos(electron_azimuth)
    v_y_pre_flight_tube = electron_speed*np.sin(electron_elevation)*np.sin(electron_azimuth)
    v_z_pre_flight_tube = electron_speed*np.cos(electron_elevation)
    
    ## Calculate time spent in pre-flight tube
    pre_flight_time = (start_junction_position - x_initial)/v_x_pre_flight_tube
    
    ## Calculate x,y, and z positions at the junction
    x_position_at_junction = x_initial + v_x_pre_flight_tube * pre_flight_time
    y_position_at_junction = y_initial + v_y_pre_flight_tube * pre_flight_time
    z_position_at_junction = z_initial + v_z_pre_flight_tube * pre_flight_time
    
    ## Calculate junction trajectories
    if trajectory_inside_junction == True:
        time_in_junction,x_position_post_junction,v_x_post_junction,x_inside_junction,y_inside_junction,z_inside_junction,t_inside_junction = interaction_region(pre_flight_time,
                                                              x_position_at_junction,
                                                              y_position_at_junction,
                                                              z_position_at_junction,
                                                              v_x_pre_flight_tube,
                                                              v_y_pre_flight_tube,
                                                              v_z_pre_flight_tube)
        
        
    else:
        time_in_junction,x_position_post_junction,v_x_post_junction = interaction_region(pre_flight_time,
                                                              x_position_at_junction,
                                                              y_position_at_junction,
                                                              z_position_at_junction,
                                                              v_x_pre_flight_tube,
                                                              v_y_pre_flight_tube,
                                                              v_z_pre_flight_tube)
    
    ## y and z components of velocity of unaffected by electric field on x axis
    v_y_post_junction = v_y_pre_flight_tube
    v_z_post_junction = v_z_pre_flight_tube
            
    ## Calculate x,y, and z positions at the end of the junction        
    y_position_post_junction = y_position_at_junction + v_y_pre_flight_tube*time_in_junction
    z_position_post_junction = z_position_at_junction + v_z_pre_flight_tube*time_in_junction
    
    ## Calculate flight time after junction
    
    with np.errstate(invalid = 'ignore',divide = 'ignore'):    
        ## We will need to divide by zero to yield the infinite time of flight
        ## for electrons with insufficient kinetic energy.
        post_flight_time = (length - end_junction_position)/v_x_post_junction
        
        ## Caluclate the final x,y, and z positions at MCP
        x_final = x_position_post_junction + v_x_post_junction * post_flight_time
        y_final = y_position_post_junction + v_y_post_junction * post_flight_time
        z_final = z_position_post_junction + v_z_post_junction * post_flight_time
    
    
        ## Calculate the total time of flight
        time_of_flight = pre_flight_time + time_in_junction + post_flight_time        
    
#########################################################################################################        

## Plotting Code##
                        
    if trajectory_plot == True:
    ## Yields x,y cross-section of the electron trajectories
    
        ## Plot geometry of the flight tube
        
        plt.figure()
        plt.plot([0,start_junction_position],[R_pre_flight_tube,R_pre_flight_tube],'k--')
        plt.plot([0,start_junction_position],[-R_pre_flight_tube,-R_pre_flight_tube],'k--')
        plt.plot([start_junction_position,end_junction_position],[R_pre_flight_tube,R_drift_tube],'k--')
        plt.plot([start_junction_position,end_junction_position],[-R_pre_flight_tube,-R_drift_tube],'k--')
        plt.plot([end_junction_position,length],[R_drift_tube,R_drift_tube],'k--')
        plt.plot([end_junction_position,length],[-R_drift_tube,-R_drift_tube],'k--')
        plt.plot([length,length],[R_drift_tube,-R_drift_tube],'k--')
        
        # Plot trajectories of the electrons
        for k in range(n):
            plt.plot([x_initial[k],x_position_at_junction[k]],[y_initial[k],y_position_at_junction[k]],'k')
            plt.plot([x_position_post_junction[k],x_final[k]],[y_position_post_junction[k],y_final[k]],'k')
            if trajectory_inside_junction == True:
                plt.plot(x_inside_junction[k,:],y_inside_junction[k,:],'k--')
        
        plt.show()
    
    if voltage_plot == True:
    ## Yields plots of voltage vs. time and voltage vs. electron energy
    
        voltage_times = np.arange(min(pre_flight_time),max(pre_flight_time + time_in_junction),voltage_time_step/100)
        plt.figure()
        plt.plot(voltage_times*1e9,V(voltage_times))
        plt.title('Voltage vs. Time')
        plt.ylabel('Voltage (V)')
        plt.xlabel('Time (ns)')
        plt.show()
        
        
        plt.figure()
        plt.plot(m/2*start_junction_position**2/voltage_times**2/eV,V(voltage_times))
        plt.title('Voltage vs. Electron Energy')
        plt.ylabel('Voltage (V)')
        plt.xlabel('Electron Energy (eV)')
        plt.show()
        
    if collection_efficiency_plot == True:
    ## Plots collection efficiency vs. energy with fixed angle
    
        angles = electron_azimuth[:num_of_angles]
        collection_efficiency_1 = np.zeros(num_of_angles)
        
        for k in range(num_of_angles):
            electron_indices = np.where(electron_azimuth == angles[k])
            electrons_collected = len(np.where(y_final[electron_indices]**2 + z_final[electron_indices]**2 < R_drift_tube**2)[0])
            collection_efficiency_1[k] = electrons_collected/num_of_energies
            
        plt.figure()
        plt.title('Collection Efficiency vs. Azimuthal Angle for '+str(E_min/eV)+' – '+str(E_max/eV)+' eV Electrons')
        plt.plot(angles/np.pi*180,collection_efficiency_1*100)
        plt.ylabel('Collection Efficiency (%)')
        plt.xlabel('Azimuthal Angle (deg)')
        plt.show()
        
        
    ## Plots collection efficiency vs. energy with fixed angle        
        energies = electron_energies[::num_of_angles]
        collection_efficiency_2 = np.zeros(num_of_energies)
        
        for k in range(num_of_energies):
            electron_indices = np.where(electron_energies == energies[k])
            electrons_collected = len(np.where(y_final[electron_indices]**2 + z_final[electron_indices]**2 < R_drift_tube**2)[0])
            collection_efficiency_2[k] = electrons_collected/num_of_angles
        
        
        plt.figure()
        plt.title('Collection Efficiency vs. Electron Energy For A %sº Spread'%int(cone_angle*180/np.pi))
        plt.plot(energies/eV,collection_efficiency_2*100,'o')
        plt.ylabel('Collection Efficiency (%)')
        plt.xlabel('Electron Energy (eV)')
        plt.show()
    
    if TOF_plot == True:
        
    ## Plots TOF vs. electron energy as well as mapping law
        
        plt.figure()
        plt.plot(electron_energies/eV,time_of_flight*1e6,'o')
        plt.plot(electron_energies/eV,(alpha*electron_energies + beta)*1e6,'--')
        plt.show()       
        
    if time_vs_position_plot == True:
        
        plt.figure()
        plt.title('Time vs. x Position')
        plt.xlabel('Time (µs)')
        plt.ylabel('Position (m)')
        for k in range(n):
            plt.plot([0.0,pre_flight_time[k]],[x_initial[k],x_position_at_junction[k]],'k')
            plt.plot([pre_flight_time[k] + time_in_junction[k], time_of_flight[k]],[x_position_post_junction[k],x_final[k]],'k')
            
            if trajectory_inside_junction == True:
                plt.plot(t_inside_junction[k,:],x_inside_junction[k,:],'k--')

        plt.show()
        
    
        
run_flight()



print('------------------------------------------')
print('|| Total Runtime:',timeit.time() - start_time,'s ||')
print('------------------------------------------')

print('------------------------------------------')
print('|| Number of electrons:',n,'electrons ||')
print('------------------------------------------')