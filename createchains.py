#!/usr/bin/env python3

import os
import sys
import time
import mpi4py.MPI as MPI
import espressopp
import logging
import math
from math import sqrt
from math import exp
time.ctime()
import numpy as np
#from numpy import *

# ==================================================================================================
# Is this a restart or a new run ?
# ==================================================================================================

if len(sys.argv) > 1:
    restart            = True
    restart_type       = sys.argv[1]
    restart_types      = ['nrrw']
    if not restart_type in restart_types:
        print("invalid restart_type - restart_type must be in ", restart_types)
        sys.exit()
    restartfile        = sys.argv[2]
    restart_cycle      = int(sys.argv[3])
else:
    restart            = False
    restart_type       = None
    restart_cycle      = 0

#os.system('g++ -o internaldistances.exe internaldistances.cpp')

# ==================================================================================================
# Preparing output data file
# ==================================================================================================

output = open('info_%s.txt' % sys.argv[0][:-3] ,'w') 
output.write(time.strftime('%a, %d %b %Y %H:%M:%S'))
output.write('\n')
output.write('\n')
output.close()

# ==================================================================================================
# VMD visualization: True or False ?
# ==================================================================================================

VMD                  = False
sock                 = None

# ==================================================================================================
# Defining simulation parameters
# ==================================================================================================

sigma              = 1.0
epsilon            = 1.0
temperature        = 1.0

# ==================================================================================================
# Setting up initial system parameters
# ==================================================================================================

N_chainlength      = 200
N_chains           = N_chainlength*15
density            = 0.849
bondlength         = 0.97
N_particles        = N_chains * N_chainlength
#Lz=Ly=3.2*math.sqrt(N_chainlength)
#Lx=N_particles/density/Lz/Ly
#box=(Lx,Ly,Lz)
L                  = ( (N_chains*N_chainlength) / density ) ** (1.0 / 3.0)
Lx=Ly=Lz=L
box                = (L, L, L)

sigma              = 1.0
epsilon            = 1.0
temperature        = 1.0

skin               = 0.2
d                  = 2.0
rc                 = pow(2, 1.0/6.0) * sigma

xs                 = time.time()
seed               = int((xs % int(xs)) * 1000000)
rng                = espressopp.esutil.RNG()
rng.seed(seed)

nodeGrid           = espressopp.tools.decomp.nodeGrid(MPI.COMM_WORLD.size)
cellGrid           = espressopp.tools.decomp.cellGrid(box, nodeGrid, d, skin)
system             = espressopp.System()
system.rng         = rng
system.bc          = espressopp.bc.OrthorhombicBC(system.rng, box)
system.skin        = skin
system.storage     = espressopp.storage.DomainDecomposition(system, nodeGrid, cellGrid)

k_theta            = 1.5
c_inf              = 1.75 #(from Kremer et al. 2005 and Sukumaran et al. 2005)

average_costheta   = (c_inf-1.0)/(c_inf+1.0)
cosmax             = 2.0*(average_costheta)-1.0
mindis             = bondlength*sqrt(2.0*(1.0+cosmax))


# ==================================================================================================
# Add new N_chains to the system or read from restart file
# ==================================================================================================

if restart:
    file = open(restartfile, 'r')
    number_of_particles_in_file  = int(file.readline())
    box_volume                   = file.readline().split()
    if N_chains * N_chainlength != number_of_particles_in_file:
        print("ERROR: wrong number of particles in restart file")
        file.close()
        sys.exit()

# chain_end_pids: list stores ends of the polymer chains for the end-to-end distance calculation
chain_end_pids                = [] 
# particle ID counter
particle_id                   = 0
for i in range(N_chains):
    # compose one RW chain
    polymer_chain             = []
    # get a random position
    startpos                  = system.bc.getRandomPos()
    # setup particle positions, bond-duples and angle-triples as random walk
    positions, bonds, angles, costheta = espressopp.tools.topology.polymerRW(particle_id, startpos, N_chainlength, bondlength, return_angles =True , mindist=mindis)
    # create the polymer chain
    for k in range(N_chainlength):
        if restart:
           col       = file.readline().split()
           part_id   = int(col[0])
           part_type = int(col[1])
           part_pos  = espressopp.Real3D(float(col[2]), float(col[3]), float(col[4]))
           part_vel  = espressopp.Real3D(float(col[5]), float(col[6]), float(col[7]))
        else:
           part_id   = particle_id + k + 1
           part_type = 0
           part_pos  = positions[k]
           part_vel  = espressopp.Real3D(0.0, 0.0, 0.0)
        particle     = [part_id, part_pos, part_type, part_vel]
        polymer_chain.append(particle)
    # add chain to the system and decompose
    system.storage.addParticles(polymer_chain, 'id', 'pos', 'type', 'v')
    system.storage.decompose()
    chain_end_pids.append((particle_id, particle_id + N_chainlength - 1))
    # particle ID counter
    particle_id += N_chainlength

if restart:
    file.close()


espressopp.tools.writexyz('mc.20.xyz', system, unfolded = True, append = False)
#os.system('./internaldistances.exe createdchains12.xyz intdist_createdchains12.txt')

E= espressopp.analysis.NeighborFluctuation(system, d).compute()

print("number of particles: ", N_particles)
print("number of chains: ", N_chains)
print("chain length: ", N_chainlength)
print("d: " , d)
print("E: ", E[0])
print("<n>: ", E[1])

# ==================================================================================================
# Print system and simulation information
# ==================================================================================================

output = open('info_%s.txt' % sys.argv[0][:-3] ,'a') 
output.write('random seed          = %i \n'         % seed)
output.write('number of particles  = %i \n'         % N_particles)
output.write('number of chains     = %i \n'         % N_chains)
output.write('chain length         = %i \n'         % N_chainlength)
output.write('simulation box       = (%f,%f,%f) \n' % (box[0],box[1],box[2]))
output.write('density              = %f \n'         % density)
output.write('rc                   = %.2f \n'       % rc)
output.write('skin                 = %.2f \n'       % system.skin)
output.write('bondlength           = %.2f \n'       % bondlength)
output.write('k_theta              = %f \n'         % k_theta)
output.write('c_inf                = %f \n'         % c_inf)
output.write('average_cos_theta    = %f \n'         % average_costheta)
output.write('cos_theta_max        = %f \n'         % cosmax)
output.write('mindist              = %f \n'         % mindis)
output.write('NodeGrid             = (%.1f,%.1f,%.1f) \n' % (nodeGrid[0],nodeGrid[1],nodeGrid[2]))
output.write('obj function         = %f \n'         % E[0] )
output.write('average naightbors   = %f \n'         % E[1] )
output.write('\n')

