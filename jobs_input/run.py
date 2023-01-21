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
import functions #a python script including all required functions in the same directory
import simulation_data  #a python script that include all simulation parameters
    
######################################################################
##               SIMULATION PARAMETERS                              ##
######################################################################   

#simulation type: start or continue
if (sys.argv[1]=='0'):
    simulation_type = 'start'
elif (sys.argv[1]=='1'):
    simulation_type = 'continue'


USE_EWALD = simulation_data.USE_EWALD #true: use Ewald; false: use Reac field
USE_DPD = simulation_data.USE_DPD #true: use DPD thermostat; false: use Langevin thermostat

SKIP_EQUI = simulation_data.SKIP_EQUI
USE_CONSTRAIN = simulation_data.USE_CONSTRAIN # SHAKE, but unsure whether it works
# simulation parameters (nvt = False is nve)
rc    = simulation_data.rc  # Verlet list cutoff
skin  = simulation_data.skin
tot_types = simulation_data.tot_types
timestep = simulation_data.timestep # 0.001 ps will crash, d't know what's going on
result_directory = simulation_data.result_directory
result_file = simulation_data.result_file

# add shear rate / NSteps
shear_rate = simulation_data.shear_rate
equi_nloops = simulation_data.equi_nloops  # total steps = nloops * isteps
equi_isteps = simulation_data.equi_isteps

# number of prod loops
prod_nloops       = simulation_data.prod_nloops#3000 #3 ns
# number of integration steps performed in each production loop
prod_isteps       = simulation_data.prod_isteps#2000

if (simulation_type=='continue'):
    prod_nloops_start = int((functions.last_time_finder(result_file)/timestep)/prod_isteps) #previously done time

# parameters to convert GROMACS tabulated potential file
sigma = simulation_data.sigma
epsilon = simulation_data.epsilon
c6 = simulation_data.c6
c12 = simulation_data.c12

# GROMACS setup files
grofile = simulation_data.grofile
topfile = simulation_data.topfile

######################################################################
##            READING INPUT DATA AND SETTING RANDOM SEED            ##
######################################################################
defaults, types, atomtypes, masses, charges, atomtypeparameters, bondtypes, bondtypeparams, angletypes, angletypeparams, dihedraltypes, dihedraltypeparams, exclusions, x, y, z, vx, vy, vz, resname, resid, Lx, Ly, Lz =gromacs.read(grofile, topfile)



#overwrite particles' position and velocity from the last frame resulted from previous simulation
if (simulation_type=='continue'):
    x, y, z, vx, vy, vz = functions.overwrite_PositionsAndVelocities(result_file)





num_particles = len(x)
density = num_particles / (Lx * Ly * Lz)
size = (Lx, Ly, Lz)
pidlist=list(range(1,num_particles+1))


# Create random seed
tseed = int( time.time() * 1000.0 )
random.seed( ((tseed & 0xff000000) >> 24) +
             ((tseed & 0x00ff0000) >>  8) +
             ((tseed & 0x0000ff00) <<  8) +
             ((tseed & 0x000000ff) << 24)   )
irand=random.randint(1,99999)

######################################################################
##                  SETTING UP SIMULATION SETUP                     ##
######################################################################
sys.stdout.write('Setting up simulation ...\n')
system = espressopp.System()
system.rng = espressopp.esutil.RNG()
system.rng.seed(irand)
system.bc = espressopp.bc.OrthorhombicBC(system.rng, size)
system.skin = skin

comm = MPI.COMM_WORLD
nodeGrid = decomp.nodeGrid(comm.size,size,rc,skin)
cellGrid = decomp.cellGrid(size, nodeGrid, rc, skin)
system.storage = espressopp.storage.DomainDecomposition(system, nodeGrid, cellGrid)

# add particles to the system and then decompose
props = ['id', 'pos', 'v', 'type', 'mass', 'q']
allParticles = []
for pid in range(num_particles):
    part = [pid + 1, Real3D(x[pid], y[pid], z[pid]),
            Real3D(vx[pid], vy[pid], vz[pid]), types[pid], masses[pid], charges[pid]]
            #Real3D(0.0,0.0,0.0), types[pid], masses[pid], charges[pid]]
    allParticles.append(part)
system.storage.addParticles(allParticles, *props)    
system.storage.decompose()

# set verlet list
vl = espressopp.VerletList(system, cutoff = rc + system.skin)
vl.exclude(exclusions)

#interaction = espressopp.interaction.VerletListLennardJonesGromacs(vl)

# set up LJ interaction according to the parameters read from the .top file
ljinteraction=gromacs.setLennardJonesInteractions(system, defaults, atomtypeparameters, vl,rc)

# bonded 2-body interactions
bondedinteractions=gromacs.setBondedInteractions(system, bondtypes, bondtypeparams)


# bonded 3-body interactions
angleinteractions=gromacs.setAngleInteractions(system, angletypes, angletypeparams)
    

# bonded 4-body interactions
dihedralinteractions=gromacs.setDihedralInteractions(system, dihedraltypes, dihedraltypeparams)


if not USE_EWALD:  #truncated coulomb
    # set up coulomb interactions according to the parameters read from the .top file
    qq_interactions=gromacs.setCoulombInteractions(system, vl, rc, types, epsilon1=1, epsilon2=80, kappa=0)
else:
    #define coulomb interactions with ewald
    coulomb_prefactor = 138.935485
    #alphaEwald     = 1.112583061 #  alpha - Ewald parameter
    #alphaEwald     = 0.660557
    rspacecutoff   = rc #3.0*pow(1/density,1.0/3.0) #  rspacecutoff - the cutoff in real space
    alphaEwald     = 2.885757 
    kspacecutoff   = 15 #  kspacecutoff - the cutoff in reciprocal space
    
    # Add Compensation terms first
    fpl_excl=espressopp.FixedPairList(system.storage)
    fpl_excl.addBonds(exclusions)
    coulombR_potBonded = espressopp.interaction.CoulombMultiSiteCorrectionEwald(coulomb_prefactor, alphaEwald, rspacecutoff)
    coulombR_intBonded = espressopp.interaction.FixedPairListTypesCoulombMultiSiteCorrectionEwald(system,fpl_excl)
    for i in range(tot_types):
        for j in range(i, tot_types):
            if not (i<tot_types-2 and j>=tot_types-2):
                coulombR_intBonded.setPotential(type1=i, type2=j, potential=coulombR_potBonded)
    system.addInteraction(coulombR_intBonded) # cancelling self energies for interatomic interactions
    
    # real-space
    coulombR_potEwald = espressopp.interaction.CoulombRSpace(coulomb_prefactor, alphaEwald, rspacecutoff)
    coulombR_intEwald = espressopp.interaction.VerletListCoulombRSpace(vl)
    for i in range(tot_types):
        for j in range(i, tot_types):
        #print("I-J: ",i,j)
            coulombR_intEwald.setPotential(type1=i, type2=j, potential = coulombR_potEwald)
    system.addInteraction(coulombR_intEwald)
    # k-space
    coulombK_potEwald = espressopp.interaction.CoulombKSpaceEwald(system, coulomb_prefactor, alphaEwald, kspacecutoff)
    coulombK_intEwald = espressopp.interaction.CellListCoulombKSpaceEwald(system.storage, coulombK_potEwald)
    system.addInteraction(coulombK_intEwald)



if USE_CONSTRAIN:
  # apply constrains using SHAKE
  # hydrogenIDs - list of indices of hydrogen atoms
  # constrainedBondsDict - dictionary mapping from a heavy atom to all the light atoms it is bonded to, key: heavy atom index (int), value: list of light atom indices (list of int)
  # constrainedBondsList - list of lists, constrained bonds for use with Rattle.addConstrainedBonds()
    hydrogenIDs, constrainedBondsDict, constrainedBondsList = espressopp.tools.findConstrainedBonds(pidlist, bondtypes, bondtypeparams, masses, massCutoff = 1.1)


if not USE_DPD: # langevin thermostat
    langevin = espressopp.integrator.LangevinThermostat(system)
    langevin.gamma = 0.5
    langevin.temperature = 2.4942 # kT in gromacs units 
    integrator = espressopp.integrator.VelocityVerlet(system)
    integrator.addExtension(langevin)
    integrator.dt = timestep

else: #DPD thrmostat
    dpd = espressopp.integrator.DPDThermostat(system, vl, num_particles)
    dpd.gamma = 1.0 # should be higher then langevin
    dpd.temperature = 2.4942 # kT in gromacs units 
    integrator = espressopp.integrator.VelocityVerlet(system)
    integrator.addExtension(dpd)
    integrator.dt = timestep

if USE_CONSTRAIN:
  # list for each constrained bond which lists: heavy atom index, light atom index, bond length, heavy atom mass, light atom mass
    rattle = espressopp.integrator.Rattle(system, maxit = 1000, tol = 1e-6, rptol = 1e-6)
    rattle.addConstrainedBonds(constrainedBondsList)
    integrator.addExtension(rattle)

# print simulation parameters
print('')
print('number of particles =', num_particles)
print('density = %.4f' % (density))
print('rc =', rc)
print('dt =', integrator.dt)
print('skin =', system.skin)
print('steps =', prod_nloops*prod_isteps)
print('NodeGrid = %s' % (nodeGrid,))
print('CellGrid = %s' % (cellGrid,))
print("RandomSeed    = ", irand)
print('')



# analysis
#configurations = espressopp.analysis.Configurations(system)
#configurations.gather()
temperature = espressopp.analysis.Temperature(system)
#pressure = espressopp.analysis.Pressure(system)
#pressureTensor = espressopp.analysis.PressureTensor(system)

######################################################################
##                       EQUILIBRATION RUN                          ##
######################################################################
conf = espressopp.analysis.Configurations(system)
conf.capacity=1
if not SKIP_EQUI:
    print("starting equilibration ...")
    print("ALSO TEST if a CH bond is shaked or NOT")
    espressopp.tools.analyse.info(system, integrator)
    for step in range(equi_nloops):
        integrator.run(equi_isteps)
        espressopp.tools.analyse.info(system, integrator)
        conf.gather()
        dtmp=conf[0][5]-conf[0][21]
        print("C5-H21: ",dtmp.abs())
    print("equilibration finished")
    integrator.resetTimers()

######################################################################
##         ANALYZING & GETTING PREPARED FOR PRODUCTION              ##
######################################################################  


T = temperature.compute()
#P = pressure.compute()
P = 0
#Pij = pressureTensor.compute()
Pij = [0,0,0,0,0,0]
Ek = 0.5 * T * (3 * num_particles)
Ep = ljinteraction.computeEnergy()
Eb, Ea, Ed=0,0,0
for bd in list(bondedinteractions.values()):Eb+=bd.computeEnergy()
for ang in list(angleinteractions.values()): Ea+=ang.computeEnergy()
for dih in list(dihedralinteractions.values()): Ed+=dih.computeEnergy()
if not USE_EWALD:
    EQQ= qq_interactions.computeEnergy()
    
else:
    EQQ= coulombR_intBonded.computeEnergy()+coulombR_intEwald.computeEnergy()+coulombK_intEwald.computeEnergy()

Etotal = Ek + Ep + EQQ + Eb + Ea + Ed

# switch to a second integrator before starting shear flow simulation
if (shear_rate>0.0):
    integrator2     = espressopp.integrator.VelocityVerletLE(system,shear=shear_rate)
else:
    integrator2     = espressopp.integrator.VelocityVerlet(system)

if USE_CONSTRAIN:
    integrator2.addExtension(rattle)

# set the integration step  
integrator2.dt  = timestep

if (simulation_type=='start'):
    integrator2.step = 0
elif (simulation_type=='continue'):
    integrator2.step = int(functions.last_time_finder(result_file)/timestep)#0 set the integration step to the latest integration step from the previous simulation


integrator2.addExtension(langevin)
# since the interaction cut-off changed the size of the cells that are used
# to speed up verlet list builds should be adjusted accordingly 
#system.storage.cellAdjust()


if (simulation_type=='start'):

    os.system("mkdir -p " + result_directory)
    h5file = open(result_file, 'x') #creating the h5 file for writing the results
    h5file.close()

    #initializing the h5 file
    with h5py.File(result_file, 'w') as h5:

        h5['particles/index'] = list(np.zeros((prod_nloops+1, num_particles, 3))) #[frame] [paricle id] [molecule_number, molecule_name (1 for cation and -1 for anion), atom_id_in_molecule (according to the name_list)]
        h5['particles/position'] = list(np.zeros((prod_nloops+1, num_particles, 3))) #[frame] [paricle id] [xpos, ypos, zpos]
        h5['particles/velocity'] = list(np.zeros((prod_nloops+1, num_particles, 3))) #[frame] [paricle id] [vx, vy, vz]
        h5['particles/time'] = list(np.zeros(prod_nloops+1))
        h5['particles/step'] = list(np.zeros(prod_nloops+1))

        h5['box/size'] = list(np.zeros(3)) #[Lx, Ly, Lz]

        h5['observables/temperature'] = list(np.zeros(prod_nloops+1))
        h5['observables/time'] = list(np.zeros(prod_nloops+1))
        h5['observables/step'] = list(np.zeros(prod_nloops+1))

        h5['energy/value'] = list(np.zeros((prod_nloops+1, 7))) #[frame][E_total, E_kinetic, E_potential, E_coulomb, E_bond, E_angle, E_dihedral]
        h5['energy/time'] = list(np.zeros(prod_nloops+1))
        h5['energy/step'] = list(np.zeros(prod_nloops+1))

        h5.close()



######################################################################
##                          PRODUCTION RUN                          ##
######################################################################

print("starting production ...")
start_time = time.process_time()
#uncomment pdbwrite below to print trajectories
#espressopp.tools.pdb.pdbwrite("traj.pdb", system, append=False, typenames={0:'O', 1:'N', 2:'C', 3:'S', 4:'P'})

if (simulation_type=='start'):
    starting_frame = 0
elif (simulation_type=='continue'):
    starting_frame = prod_nloops_start

for step in range(starting_frame, prod_nloops+1):
    currentStep=step*prod_isteps
    if step>0:
        integrator2.run(prod_isteps) # print out every steps/check steps
    
    T = temperature.compute()
    #P = pressure.compute()
    #Pij = pressureTensor.compute()
    P = 0
    Pij = [0,0,0,0,0,0]
    Ek = 0.5 * T * (3 * num_particles)
    Ep = ljinteraction.computeEnergy()
    if not USE_EWALD:
        EQQ= qq_interactions.computeEnergy()
        #EQQ=.0
    else:
        EQQ= coulombR_intBonded.computeEnergy()+coulombR_intEwald.computeEnergy()+coulombK_intEwald.computeEnergy()
        
    Eb, Ea, Ed=0,0,0
    for bd in bondedinteractions.values():Eb+=bd.computeEnergy()
    for ang in angleinteractions.values(): Ea+=ang.computeEnergy()
    for dih in dihedralinteractions.values(): Ed+=dih.computeEnergy()
    Etotal = Ek + Ep + EQQ + Eb + Ea + Ed
      
    energy_array = [Etotal, Ek, Ep, EQQ, Eb, Ea, Ed]
    
    functions.write_h5md(result_file, system, integrator2.step, integrator2.step * timestep, prod_isteps * timestep, T/constants.k/constants.N_A*1000, energy_array)
    
    #espressopp.tools.pdb.pdbwrite("traj.pdb", system, append=True, typenames={0:'O', 1:'N', 2:'C', 3:'S', 4:'P'})  



######################################################################
##      PRINT SUMMARY E.G. TIMING & NEIGHBOUR LIST INFORMATION      ##
######################################################################
end_time = time.process_time()
timers.show(integrator.getTimers(), precision=4)
sys.stdout.write('Total # of neighbors = %d\n' % vl.totalSize())
sys.stdout.write('Ave neighs/atom = %.1f\n' % (vl.totalSize() / float(num_particles)))
sys.stdout.write('Neighbor list builds = %d\n' % vl.builds)
sys.stdout.write('Integration steps = %d\n' % integrator2.step)
sys.stdout.write('CPU time = %.1f\n' % (end_time - start_time))

