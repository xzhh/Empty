#!/usr/bin/env python3
#
#  Copyright (C) 2013-2017(H)
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
# 
# -*- coding: utf-8 -*-
#
###########################################################################
#                                                                         #
#  ESPResSo++ Benchmark Python script for a polymer melt                  #
#                                                                         #
###########################################################################

import sys
import time
import espressopp
import random
import math
import mpi4py.MPI as MPI
import logging
import numpy as np
from scipy.signal import argrelextrema
from espressopp import Real3D, Int3D
from espressopp.tools import lammps, gromacs
from espressopp.tools import decomp, timers, replicate
import os
#import gc

def vec_angle(v1, v2):
  """ Returns the angle in radians between vectors 'v1' and 'v2'::

          >>> angle_between((1, 0, 0), (0, 1, 0))
          1.5707963267948966
          >>> angle_between((1, 0, 0), (1, 0, 0))
          0.0
          >>> angle_between((1, 0, 0), (-1, 0, 0))
          3.141592653589793
  """
  v1_u = v1/np.linalg.norm(v1)
  v2_u = v2/np.linalg.norm(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi*180.0
    
# simulation parameters (nvt = False is nve)
rc = pow(2.0, 1.0/6.0)
skin = 0.3
dpd = False
shear_rate = 0.1
timestep = 0.002

# set parameters for simulations
seed               = 654321  # seed for random
density            = 0.849
bondlen            = 0.97
monomers_per_chain = 50      # the number of monomers per a chain
num_chains         = 15*monomers_per_chain     # the number of chains
nc_print = max(100,3*monomers_per_chain) #num_chains
temperature        = 1.0     # set temperature

# number of equil loops
equi_nloops = 100
equi_isteps = 50
# number of prod loops
prod_nloops       = 1
# number of integration steps performed in each production loop
prod_isteps       = 25000

# NReplication
NumDup = 1

######################################################################
### IT SHOULD BE UNNECESSARY TO MAKE MODIFICATIONS BELOW THIS LINE ###
######################################################################
sys.stdout.write('Setting up simulation ...\n')

num_particles = num_chains*monomers_per_chain
num_one_copy = num_particles
Lz=Ly=3.2*math.sqrt(monomers_per_chain)
Lx=num_particles/density/Lz/Ly

# Create random seed
tseed = int( time.time() * 1000.0 )
random.seed( ((tseed & 0xff000000) >> 24) +
             ((tseed & 0x00ff0000) >>  8) +
             ((tseed & 0x0000ff00) <<  8) +
             ((tseed & 0x000000ff) << 24)   )
irand=random.randint(1,99999)

file = open("equil_20.xyz", 'r')
number_of_particles_in_file  = int(file.readline())
box_volume                   = file.readline().split()

if num_chains * monomers_per_chain != number_of_particles_in_file:
  print("ERROR: wrong number of particles in restart file")
  file.close()
  sys.exit()

bonds = []
angles = []
x = []
y = []
z = []
vx = []
vy = []
vz = []
ptype = []
pid = 1
for i in range(num_chains):
  #bonds = []
  #angles = []
  
  for k in range(monomers_per_chain):
    col     = file.readline().split()

    if (len(col) == 8 or len(col) == 5):
      ptype.append(int(col[1]))
      x.append(float(col[2]))
      y.append(float(col[3]))
      z.append(float(col[4]))
      vx.append(float(col[5]))
      vy.append(float(col[6]))
      vz.append(float(col[7]))
      
	
    elif (len(col) == 7 or len(col) == 4):
      ptype.append(0)
      x.append(float(col[1]))
      y.append(float(col[2]))
      z.append(float(col[3]))
      vx.append(float(col[4]))
      vy.append(float(col[5]))
      vz.append(float(col[6]))
    
    if k < monomers_per_chain-1:
      bonds.append((pid+k,pid+k+1))
	
    if k < monomers_per_chain-2:
      angles.append((pid+k, pid+k+1, pid+k+2))
  pid += monomers_per_chain

file.close()

replicate   = (NumDup*1,NumDup*3,NumDup*3)
rp = espressopp.tools.ReplicateParallel()
num_particles, Lx, Ly, Lz = rp.replicate(bonds, angles, x, y, z, Lx, Ly, Lz, *replicate)

size = (Lx, Ly, Lz)
system = espressopp.System()
system.rng = espressopp.esutil.RNG()
system.rng.seed(irand)
system.bc = espressopp.bc.OrthorhombicBC(system.rng, size)

system.skin = skin
comm = MPI.COMM_WORLD
nodeGrid = espressopp.tools.decomp.nodeGrid(comm.size,size,rc,skin)
cellGrid = espressopp.tools.decomp.cellGrid(size,nodeGrid,rc,skin)
system.storage = espressopp.storage.DomainDecomposition(system, nodeGrid, cellGrid)

print("NCPUs         = ", comm.size)
print("nodeGrid      = ", nodeGrid)
print("cellGrid      = ", cellGrid)
print("Npart         = ", num_particles)
print("BoxL          = ", size)
print("RandomSeed    = ", irand)

# add particles to the system and then decompose
# do this in chunks of 1000 particles to speed it up
props    = ['id', 'type', 'mass', 'pos', 'v']
vel_zero = espressopp.Real3D(0.0, 0.0, 0.0)
pid      = 1
mass     = 1.0
seed_particles = []
for i in range(num_one_copy):
  part = [i + 1, ptype[i], mass,
  Real3D(x[i], y[i], z[i]),
  Real3D(vx[i], vy[i], vz[i])]
  seed_particles.append(part)
  #if (i+1) % monomers_per_chain == 0:
  #  system.storage.addParticles(new_particles, *props)
  #  system.storage.decompose()
  #  new_particles = []
#system.storage.addParticles(new_particles, *props)
rp.addParticles(system.storage, 1, seed_particles, *props)
system.storage.decompose()
#sys.exit(0)

# Lennard-Jones with Verlet list
vl = espressopp.VerletList(system, cutoff = rc + system.skin/2.0)
potLJ = espressopp.interaction.LennardJones(1.0, 1.0, cutoff = rc, shift = "auto")
interLJ = espressopp.interaction.VerletListLennardJones(vl)
interLJ.setPotential(type1 = 0, type2 = 0, potential = potLJ)
system.addInteraction(interLJ)

bondlist = espressopp.FixedPairList(system.storage)
anglelist = espressopp.FixedTripleList(system.storage)
#bondlist.addBonds(bonds)
#anglelist.addTriples(angles)
rp.addBonds(bondlist)
rp.addTriples(anglelist)
del bonds
del angles
#potFENE = espressopp.interaction.Harmonic(K=30.0, r0=0.0)
#interFENE = espressopp.interaction.FixedPairListHarmonic(system, bondlist, potFENE)
#potFENE = espressopp.interaction.FENECapped(K=30.0, r0=0.0, rMax=1.5, r_cap=1.499)
#interFENE = espressopp.interaction.FixedPairListFENECapped(system, bondlist, potFENE)
potFENE = espressopp.interaction.FENE(K=30.0, r0=0.0, rMax=1.5)
interFENE = espressopp.interaction.FixedPairListFENE(system, bondlist, potFENE)
system.addInteraction(interFENE)
# Cosine with FixedTriple list
potCosine = espressopp.interaction.Cosine(K=1.5, theta0=0.0)
interCosine = espressopp.interaction.FixedTripleListCosine(system, anglelist, potCosine)
#interCosine.setPotential(type1 = 0, type2 = 0, potential = potCosine)
system.addInteraction(interCosine)

# integrator
integrator = espressopp.integrator.VelocityVerlet(system)
integrator.dt = timestep

if (dpd):
  thermo=espressopp.integrator.DPDThermostat(system, vl, ntotal=num_particles)
  thermo.gamma = 1.0
  thermo.tgamma = 0.0
  thermo.temperature = temperature
  integrator.addExtension(thermo)
else:
  thermo = espressopp.integrator.LangevinThermostat(system)
  thermo.gamma = 1.0
  thermo.temperature = temperature
  integrator.addExtension(thermo)
  

# print simulation parameters
print(' ')
print('number of particles =', num_particles)
print('density = %.4f' % (density))
print('rc =', rc)
print('dt =', integrator.dt)
print('skin =', system.skin)
print('thermostat(DPD) =', dpd)
print('steps =', prod_nloops*prod_isteps)
print('NodeGrid = %s' % (nodeGrid))
print('CellGrid = %s' % (cellGrid))
print(' ')

# analysis
#conf = espressopp.analysis.Configurations(system)
#conf.gather()

temperature = espressopp.analysis.Temperature(system)
pressure = espressopp.analysis.Pressure(system)
pressureTensor = espressopp.analysis.PressureTensor(system)

fmt = '%5d %8.4f %10.5f %8.5f %12.3f %12.3f %12.3f %12.3f %12.3f\n'

espressopp.tools.pdb.pdbwrite('input.pdb', system, append=True)
espressopp.tools.analyse.info(system, integrator)
integrator.run(10)
espressopp.tools.analyse.info(system, integrator)
sys.exit(0)

#Equilibration
print("starting equilibration ...")
espressopp.tools.analyse.info(system, integrator)
for step in range(equi_nloops):
  integrator.run(equi_isteps)
  espressopp.tools.analyse.info(system, integrator)
  if (math.isnan(interFENE.computeEnergy())):
    print("FENE becomes NaN after equilibration")
    sys.exit(0)
print("equilibration finished")
print("NRESORT>", integrator.getNumResorts())

#T = temperature.compute()
#P = pressure.compute()
#Pij = pressureTensor.compute()
#Ek = 0.5 * T * (3 * num_particles)
#Ep = interLJ.computeEnergy()
#Eb = interFENE.computeEnergy()
#Ea = interCosine.computeEnergy()
#Etotal = Ek + Ep + Eb + Ea
#sys.stdout.write(' step     T          P       Pxy        etotal      ekinetic      epair        ebond       eangle\n')
#sys.stdout.write(fmt % (0, T, P, Pij[3], Etotal, Ek, Ep, Eb, Ea))

########################################################################
# RUN shear flow MD                                                    #
########################################################################

# cancelling thermostat
#thermo.disconnect()
# set all integrator timers to zero again (they were increased during warmup)
integrator.resetTimers()
# set integrator time step to zero again
integrator.step = 0

if (shear_rate>0.0):
  integrator2     = espressopp.integrator.VelocityVerletLE(system,shear=shear_rate,viscosity=ifVisc)
else:
  integrator2     = espressopp.integrator.VelocityVerlet(system)
# set the integration step  
integrator2.dt  = timestep
integrator2.step = 0

integrator2.addExtension(thermo)
#fixpositions = espressopp.integrator.FixPositions(system, fixedWall, fixMask)
#integrator2.addExtension(fixpositions)

#IMPORTANT
system.storage.cellAdjust()

print("starting production ...")
espressopp.tools.analyse.info(system, integrator2)
#sock = espressopp.tools.vmd.connect(system)
filename = "traj.pdb"
if os.path.exists(filename):
  os.remove(filename)

start_time = time.process_time()
for step in range(prod_nloops+1):
  if step > 0:  
    integrator2.run(prod_isteps)
    espressopp.tools.analyse.info(system, integrator2)
    if (math.isnan(interFENE.computeEnergy())):
      print("FENE becomes NaN during production")
      sys.exit(0)
    #espressopp.tools.pdb.pdbwrite(filename, system, append=True)
    #espressopp.tools.xyzfilewrite(filename, system, velocities = False, charge = False, append=True, atomtypes={0:'X'})

end_time = time.process_time()
print("production finished")

T = temperature.compute()
P = pressure.compute()
Pij = pressureTensor.compute()
Ek = 0.5 * T * (3 * num_particles)
Ep = interLJ.computeEnergy()
Eb = interFENE.computeEnergy()
Ea = interCosine.computeEnergy()
Etotal = Ek + Ep + Eb + Ea
sys.stdout.write(fmt % (prod_nloops*prod_isteps, T, P, Pij[3], Etotal, Ek, Ep, Eb, Ea))
sys.stdout.write('\n')

# print timings and neighbor list information
timers.show(integrator2.getTimers(), precision=3)
sys.stdout.write('Total # of neighbors = %d\n' % vl.totalSize())
sys.stdout.write('Ave neighs/atom = %.1f\n' % (vl.totalSize() / float(num_particles)))
sys.stdout.write('Neighbor list builds = %d\n' % vl.builds)
sys.stdout.write('Integration steps = %d\n' % integrator2.step)
sys.stdout.write('CPUs = %i CPU time per CPU = %.1f\n' % (comm.size,end_time - start_time))
