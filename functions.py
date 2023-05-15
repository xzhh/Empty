#!/usr/bin/env python3
#  Copyright (C) 2012-2017(H)
#      Max Planck Institute for Polymer Research
#
#  This file is part of ESPResSo++.
#
#  ESPResSo++ is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ESPResSo++ is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

###########################################################################
#                                                                         #
#  ESPResSo++ Python script for tabulated GROMACS simulation              #
#                                                                         #
###########################################################################





import sys
import os
import time
import espressopp
import mpi4py.MPI as MPI
#from mpi4py import MPI
import logging
import random
import numpy as np
import math
from scipy import constants
from espressopp import Real3D, Int3D
from espressopp.tools import gromacs
from espressopp.tools import decomp
from espressopp.tools import timers
import collections
import h5py

######################################################################
##       H5MD WRITER FUNCTION FOR RECORDING SIMULATION RESULTS      ##
######################################################################
def write_h5md(file, system, step, time, record_time, temperature, energy):
    
    iteration = round(time/record_time)
    
    #list of all atoms in each molecule
    name_list_cation = ["NA1", "NA2", "HC1", "HC2", "HC3", "HC4", "HC5", "HC6","HC7", "HA1", "HA2", "HA3", "H11", "H12", "H13", "H14", "H15", "CW1", "CW2", "CT1", "CS1", "CR1", "C21", "C11", "C12"]
    name_list_anion = ["P1", "F1", "F2", "F3", "F4", "F5", "F6"]
    cation_atoms_number = len(name_list_cation)
    anion_atoms_number = len(name_list_anion)
    cation = "BMI"
    anion = "PF6"
    cation_number = cation_number_temporary = 0
    anion_number = anion_number_temporary = 0
            
    max_pid = int(espressopp.analysis.MaxPID(system).compute()) #total number of particles
     
    pos = np.zeros([max_pid, 3])
    v = np.zeros([max_pid, 3])
    index = np.zeros([max_pid, 3]) #molecule_number, molecule_name, atom_id_in_molecule
    for i in range(len(energy)): energy[i] = round(energy[i],4)

    #iterating over all particles
    for pid in range(1,max_pid+1):
                
        particle = system.storage.getParticle(pid)
        pos[pid-1, :] = np.array([particle.pos[0], particle.pos[1], particle.pos[2]])
        v[pid-1, :] = np.array([particle.v[0], particle.v[1], particle.v[2]])
              
        #this if statement detect whether the particle is in the cation or anion molecule and it adds to the corresponding molecule numbers
        if particle.type not in [10, 11]:
            
            molecule_name = 1#cation
            atom_id_in_molecule = pid - cation_number*len(name_list_cation) - anion_number*len(name_list_anion)            
            #atom_name = name_list_cation[atom_id_in_molecule]
                        
            cation_number_temporary = cation_number
            
            cation_number += math.floor(atom_id_in_molecule/cation_atoms_number)
                                       
        else:
        
            molecule_name = -1#anion    
            atom_id_in_molecule = pid - cation_number*len(name_list_cation) - anion_number*len(name_list_anion)
            #atom_name = name_list_anion[atom_id_in_molecule]
            
            cation_number_temporary = cation_number #otherwise the molecule number repeats when cations finish and anions start
            anion_number_temporary = anion_number
            
            anion_number += math.floor(atom_id_in_molecule/anion_atoms_number)
                
        molecule_number = anion_number_temporary + cation_number_temporary + 1       
        
        index[pid-1, :] = np.array([molecule_number, molecule_name, atom_id_in_molecule])
    

    #write data in h5 file
    with h5py.File(file, 'r+') as h5:

        h5['particles/index'][iteration, :, :] = index#list(np.array([molecule_number, molecule_name, atom_id_in_molecule]).T)
        h5['particles/position'][iteration, :, :] = pos#list(np.array([xpos, ypos, zpos]).T)
        h5['particles/velocity'][iteration, :, :] = v#list(np.array([vx, vy, vz]).T)
        h5['particles/time'][iteration] = time
        h5['particles/step'][iteration] = step
         
        h5['observables/temperature'][iteration] = round(temperature,4)
        h5['observables/time'][iteration] = time
        h5['observables/step'][iteration] = step
        
        h5['energy/value'][iteration, :] = energy[:] #[time_step, time, E_total, E_kinetic, E_potential, E_coulomb, E_bond, E_angle, E_dihedral]
        h5['energy/time'][iteration] = time
        h5['energy/step'][iteration] = step
        
        h5.close()
    
    
    
    
    
##################################################################################
##       LAST TIME FRAME READER AT THE END OF EACH STAGE OF THE SIMULATION      ##
##################################################################################    
def last_time_finder(file, time_path='energy/time'):# "time_path" is where to find time in the h5 file
    with h5py.File(file, 'r') as h5:
        last_time_frame = np.max(np.array(h5[time_path]))
        h5.close()
    
    return last_time_frame
    
    

#############################################################################################################
##       PARTICLES' POSITION AND VELOCITY OVERWRITER FROM H5MD FILE RESULTED FROM PREVIOUS SIMULATION      ##
#############################################################################################################
def overwrite_PositionsAndVelocities(h5_file):
    
    last_time = last_time_finder(h5_file)#what is the last time frame

    
    with h5py.File(h5_file, 'r') as h5:
        index = np.where(np.array(h5['energy/time'])==last_time)[0][0] #what is the index of the last time frame from "energy/time" path in h5 file
        h5.close()
    
        
    with h5py.File(h5_file, 'r') as h5:

        px = h5['particles/position'][index, :, 0]
        py = h5['particles/position'][index, :, 1]
        pz = h5['particles/position'][index, :, 2]
        
        vx = h5['particles/velocity'][index, :, 0] 
        vy = h5['particles/velocity'][index, :, 1] 
        vz = h5['particles/velocity'][index, :, 2]
        
        h5.close()
        
    return px, py, pz, vx, vy, vz
    
