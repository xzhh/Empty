#!/usr/bin/env python2
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
from espressopp import Real3D, Int3D
from espressopp.tools import lammps, gromacs
from espressopp.tools import decomp, timers, replicate
import os
#import gc


# simulation parameters (nvt = False is nve)
ifVisc=False
if os.path.exists("FLAG_VIS"):
    ifVisc=True
rc = 1.12 ##pow(2.0, 1.0/6.0)
skin = 0.3
dpd = True
ifbond = True
skipEqui=False
skipPPA=False
skipProf=True
skipOri=True
skipMSID=True
FLAG_MD=not (skipProf and skipOri)
shear_rate = 0.02
timestep = 0.002
ts_warm = 0.0002

# set parameters for simulations
seed               = 654321  # seed for random
density            = 0.84
bondlen            = 0.97
monomers_per_chain = 100      # the number of monomers per a chain
num_chains         = 15*monomers_per_chain     # the number of chains
nc_print = max(100,3*monomers_per_chain) #num_chains
temperature        = 1.0     # set temperature

warmup_nloops      = 2000 # based on normal timestep
warmup_isteps      = 50
equi_nloops = 2000 # based on normal timestep
equi_isteps = 100
e2_nloops = 500
# number of prod loops
prod_nloops       = 2000 #200
# number of integration steps performed in each production loop
prod_isteps       = 50
msid_nloops = prod_nloops/4

######################################################################
### IT SHOULD BE UNNECESSARY TO MAKE MODIFICATIONS BELOW THIS LINE ###
######################################################################
sys.stdout.write('Setting up simulation ...\n')

num_particles = num_chains*monomers_per_chain
Lz=Ly=3.2*math.sqrt(monomers_per_chain)
Lx=num_particles/density/Lz/Ly
size = (Lx, Ly, Lz)

# Create random seed
tseed = int( time.time() * 1000.0 )
random.seed( ((tseed & 0xff000000) >> 24) +
             ((tseed & 0x00ff0000) >>  8) +
             ((tseed & 0x0000ff00) <<  8) +
             ((tseed & 0x000000ff) << 24)   )
irand=random.randint(1,99999)

system = espressopp.System()
system.rng = espressopp.esutil.RNG()
system.rng.seed(irand)
system.bc = espressopp.bc.OrthorhombicBC(system.rng, size)

system.skin = skin
comm = MPI.COMM_WORLD
if comm.size%4 ==0:
  nodeGrid = espressopp.Int3D(comm.size/4,2,2)
else:
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
bondlist = espressopp.FixedPairList(system.storage)
anglelist = espressopp.FixedTripleList(system.storage)
pid      = 1
chain_type = 0
mass     = 1.0
chain = []
for i in range(num_chains):
  startpos = system.bc.getRandomPos()
  #positions, bonds, angles = espressopp.tools.topology.polymerRW(pid, startpos, monomers_per_chain, bondlen, return_angles=True, rng=None)#system.rng)
  positions, bonds = espressopp.tools.topology.polymerRW(pid, startpos, monomers_per_chain, bondlen, rng=None)#system.rng)
  for k in range(monomers_per_chain):
    part = [pid + k, chain_type, mass, positions[k], vel_zero]
    chain.append(part)
  pid += monomers_per_chain
  #chain_type += 1
  system.storage.addParticles(chain, *props)
  system.storage.decompose()
  chain = []
  bondlist.addBonds(bonds)
  #if (ifring):
  #  bondlist.addBonds([(pid-1,pid-monomers_per_chain)])
  #anglelist.addTriples(angles)
  #print bonds
  #print angles
  #print positions[0]
  #sys.exit(0)
system.storage.decompose()


# Lennard-Jones with Verlet list
vl = espressopp.VerletList(system, cutoff = rc + system.skin)
vl_intra = espressopp.VerletList(system, cutoff = rc + system.skin)

exlist=[]
for nc in range(num_chains):
  for i in range(nc*monomers_per_chain+1,(nc+1)*monomers_per_chain):
    for j in range(i+1,(nc+1)*monomers_per_chain+1):
      exlist.append((i,j))
vl.exclude(exlist)
vl_intra.exclude(vl.getAllPairs())

potLJ = espressopp.interaction.LennardJones(0.01, 1.0, cutoff = rc, shift = "auto")
interLJ = espressopp.interaction.VerletListLennardJones(vl)
interLJ.setPotential(type1 = 0, type2 = 0, potential = potLJ)
system.addInteraction(interLJ)

potLJ_intra = espressopp.interaction.LennardJones(1.0, 1.0, cutoff = rc, shift = "auto")
interLJ_intra = espressopp.interaction.VerletListLennardJones(vl_intra)
interLJ_intra.setPotential(type1 = 0, type2 = 0, potential = potLJ_intra)
system.addInteraction(interLJ_intra)

if (ifbond):
  print("Num_BondList  = ", bondlist.size())
  #potFENE = espressopp.interaction.Harmonic(K=30.0, r0=0.0)
  #interFENE = espressopp.interaction.FixedPairListHarmonic(system, bondlist, potFENE)
  potFENE = espressopp.interaction.FENECapped(K=30.0, r0=0.0, rMax=1.5, r_cap=1.499)
  interFENE = espressopp.interaction.FixedPairListFENECapped(system, bondlist, potFENE)
  #potFENE = espressopp.interaction.FENE(K=30.0, r0=0.0, rMax=1.5)
  #interFENE = espressopp.interaction.FixedPairListFENE(system, bondlist, potFENE)
  system.addInteraction(interFENE)
## Cosine with FixedTriple list
#   potCosine = espressopp.interaction.Cosine(K=1.5, theta0=3.1415926)
#   interCosine = espressopp.interaction.FixedTripleListCosine(system, anglelist, potCosine)
#   #interCosine.setPotential(type1 = 0, type2 = 0, potential = potCosine)
#   system.addInteraction(interCosine)


# integrator
integrator = espressopp.integrator.VelocityVerlet(system)
  
integrator.dt = ts_warm

if (dpd):
  langevin=espressopp.integrator.DPDThermostat(system, vl)
  langevin.gamma=5.0
  langevin.tgamma=0.0
  langevin.temperature = temperature
  integrator.addExtension(langevin)
else:
  langevin = espressopp.integrator.LangevinThermostat(system)
  langevin.gamma = 5.0
  langevin.temperature = temperature
  integrator.addExtension(langevin)
  
  
system.bc          = espressopp.bc.OrthorhombicBC(system.rng, size)

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

#integrator.run(0)
#espressopp.tools.pdb.pdbwrite('input.pdb', system, append=True)
#sys.exit(0)

steepest = espressopp.integrator.MinimizeEnergy(system, gamma=0.001, ftol=0.1, max_displacement=0.001, variable_step_flag=False)
for k in range(50):
  steepest.run(10)
  if (math.isnan(interFENE.computeEnergy())):
    print("FENE becomes NaN after minimization")
    sys.exit(0)
  espressopp.tools.analyse.info(system, steepest)
#espressopp.tools.pdbwrite('input.pdb', system, molsize=20,typenames=None)
#sys.exit(0)

#Warm-up
print("starting warm-up ...")
nwarm=int(warmup_nloops*timestep/ts_warm)
espressopp.tools.analyse.info(system, integrator)
for step in range(nwarm):
  # incresing strength of force
  potLJ.epsilon = max((step+1.0)/nwarm,0.01)*1.0
  interLJ.setPotential(type1=0, type2=0, potential=potLJ)
  #if (ifbond):
  #  potFENE.K= (step+1.0)/nwarm*30.0
  #  interFENE.setPotential(potential=potFENE)
  
  # perform warmup_isteps integraton steps
  #print "STEP ",step,"> ",potLJ.epsilon,potFENE.K
  integrator.run(warmup_isteps)
  # print status info
  espressopp.tools.analyse.info(system, integrator) 
  if (math.isnan(interFENE.computeEnergy())):
    print("FENE becomes NaN after warm-up")
    sys.exit(0)
  
print("warmup finished")

#replace the capped potential
if (ifbond):
  system.removeInteraction(2) 
  potFENE = espressopp.interaction.FENE(K=30.0, r0=0.0, rMax=1.5)
  interFENE = espressopp.interaction.FixedPairListFENE(system, bondlist, potFENE)
  system.addInteraction(interFENE)

#Equilibration
print("starting equilibration ...")
nequi=int(equi_nloops*timestep/ts_warm)/2
espressopp.tools.analyse.info(system, integrator)
for step in range(nequi):
  integrator.run(equi_isteps)
  espressopp.tools.analyse.info(system, integrator)
  if (math.isnan(interFENE.computeEnergy())):
    print("FENE becomes NaN after equilibration")
    sys.exit(0)
    
print("restore timestep ...")
integrator.dt = timestep
for step in range(equi_nloops/2):
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
# RUN shear flow MD                                                  #
########################################################################


# cancelling thermostat
#langevin.disconnect()
# set all integrator timers to zero again (they were increased during warmup)
integrator.resetTimers()
# set integrator time step to zero again
integrator.step = 0

if (shear_rate>0.0):
  integrator2     = espressopp.integrator.VelocityVerletLE(system,shear=shear_rate)
else:
  integrator2     = espressopp.integrator.VelocityVerlet(system)
# set the integration step  
integrator2.dt  = timestep
integrator2.step = 0

integrator2.addExtension(langevin)
#fixpositions = espressopp.integrator.FixPositions(system, fixedWall, fixMask)
#integrator2.addExtension(fixpositions)
# Define a Lees-Edwards BC instead of std orthorhombic

#system.bc=espressopp.bc.LeesEdwardsBC(system.rng, integrator2, size, shear=shear_rate)

# since the interaction cut-off changed the size of the cells that are used
# to speed up verlet list builds should be adjusted accordingly 
system.storage.cellAdjust()

print("starting production ...")
espressopp.tools.analyse.info(system, integrator2)
#sock = espressopp.tools.vmd.connect(system)
filename = "traj.pdb"
if os.path.exists(filename):
  os.remove(filename)

#Preparation
if FLAG_MD:
  pos = espressopp.analysis.Configurations(system)
  vel = espressopp.analysis.Velocities(system)
  bin_size=50
  mass=1 #here assume all mass=1
  #integrator2.run(0)
  pos.capacity=1
  vel.capacity=1

if not skipMSID:
  msid = espressopp.analysis.MeanSquareInternalDist(system,monomers_per_chain,start_pid=1)
start_time = time.process_time()
for step in range(prod_nloops+1):
  if (not skipMSID) and (step % msid_nloops ==0):
    msid.gather()
    result = msid.compute()
    file_MSID = open("msid.dat","a")
    for i in xrange(monomers_per_chain-1):
      line = "%d %f\n" % (i+1,result[i]/(i+1))
      file_MSID.write(line)
    line="\n"
    file_MSID.write(line)
    file_MSID.close()
    del result

  if step > 0:  
    integrator2.run(prod_isteps)
    espressopp.tools.analyse.info(system, integrator2)
    if ifVisc:
      print("SIGXZ> %d %.6f" % (step*prod_isteps,system.sumP_xz))
    if (math.isnan(interFENE.computeEnergy())):
      print("FENE becomes NaN during production")
      sys.exit(0)
    #espressopp.tools.pdb.pdbwrite(filename, system, append=True)
    #espressopp.tools.xyzfilewrite(filename, system, velocities = False, charge = False, append=True, atomtypes={0:'X'})
  
  if FLAG_MD:
    #fetch coordinates and velocities
    pos.gather()
    vel.gather()
    #get no. of the current step
    currentStep=step*prod_isteps
    #Init
    if not skipProf:
      cnt=[0]*bin_size
      vx =[0]*bin_size
      vz =[0]*bin_size
      sx =[0]*bin_size
      ty =[0]*bin_size
      tz =[0]*bin_size
    mo=[.0,.0,.0,.0]
    #ori=[[.0,.0,.0]]*num_chains
    for j in range(1,num_particles+1):
      #print "IDX>",j
      # define shear speed of the current particle
      spdShear=shear_rate*(pos[0][j][2]-Lz/2.0)
      
      #momentum
      mo[0]+=vel[0][j][0]
      mo[1]+=vel[0][j][1]
      mo[2]+=vel[0][j][2]
      mo[3]+=vel[0][j][0]+spdShear
      
      #profile
      if not skipProf:
        bid=int(pos[0][j][2]/Lz*bin_size)
        if bid<0:
          bid=0
        elif bid>=bin_size:
          bid=bin_size-1
        #collect profile info by layers along gradient-dir
        cnt[bid]+=1;
        vx[bid]+=vel[0][j][0]
        vz[bid]+=vel[0][j][2]
        sx[bid]+=vel[0][j][0]+spdShear
        ty[bid]+=vel[0][j][1]*vel[0][j][1]*mass/1.0
        tz[bid]+=vel[0][j][2]*vel[0][j][2]*mass/1.0
      
      #collect end-to-end orientation (normailized)
      if not skipOri:
        print("Current Ori is wrong(shearOffset)")
        sys.exit(0)
        mono_idx=(j-1)%monomers_per_chain
        chain_idx=(j-1-mono_idx)/monomers_per_chain
        if chain_idx<nc_print:
          if mono_idx==0:
            ori=Real3D(.0,.0,.0)
            dim=[0,0,0]
            ori-=pos[0][j]
          else:
            l=pos[0][j]-pos[0][j-1]
            if l[0]<-Lx/2.0:
              dim[0]+=1
            elif l[0]>Lx/2.0:
              dim[0]-=1
            if l[1]<-Ly/2.0:
              dim[1]+=1
            elif l[1]>Ly/2.0:
              dim[1]-=1
            if l[2]<-Lz/2.0:
              dim[2]+=1
            elif l[2]>Lz/2.0:
              dim[2]-=1
            if mono_idx==monomers_per_chain-1:
              ori+=pos[0][j]
              ori+=Real3D(dim[0]*Lx,dim[1]*Ly,dim[2]*Lz)
              ori_abs=ori.abs()
              print("CHAIN-%0d> %10d %.6f %.6f %.6f" %(chain_idx,currentStep,ori[0]/ori_abs,ori[1]/ori_abs,ori[2]/ori_abs))
              del ori
              #gc.collect()
    
    #print data
    print("MOMENTUM> %10d %.6f %.6f %.6f %.6f" %(currentStep,mo[0]/num_particles,mo[1]/num_particles,mo[2]/num_particles,mo[3]/num_particles))
    if not skipProf:
      for j in range(bin_size):
        #print "COUNT>",cnt[j]
        zb=(j+0.5)/bin_size
        if cnt[j]>0:
          print("DISTR> %10d %.3f %.6f" %(currentStep,zb,cnt[j]/num_particles*bin_size))
          print("VX> %10d %.3f %.6f" %(currentStep,zb,vx[j]/cnt[j]))
          print("VZ> %10d %.3f %.6f" %(currentStep,zb,vz[j]/cnt[j]))
          if shear_rate>.0:
            print("VSHEAR> %10d %.3f %.6f" %(currentStep,zb,sx[j]/cnt[j]/(shear_rate*Lz/2.0)))
          print("TY> %10d %.3f %.6f" %(currentStep,zb,ty[j]/cnt[j]))
          print("TZ> %10d %.3f %.6f" %(currentStep,zb,tz[j]/cnt[j]))
        
    #release memory
      del cnt
      del vx 
      del vz 
      del sx 
      del ty 
      del tz 
    del mo 
    #gc.collect()
    #pos.clear()
    #vel.clear()

end_time = time.process_time()
print("production finished")

if (ifbond):
  T = temperature.compute()
  P = pressure.compute()
  Pij = pressureTensor.compute()
  Ek = 0.5 * T * (3 * num_particles)
  Ep = interLJ.computeEnergy()
  Eb = interFENE.computeEnergy()
  #Ea = interCosine.computeEnergy()
  Etotal = Ek + Ep + Eb #+ Ea
  #sys.stdout.write(fmt % (prod_nloops*prod_isteps, T, P, Pij[3], Etotal, Ek, Ep, Eb, Ea))
  sys.stdout.write(fmt % (prod_nloops*prod_isteps, T, P, Pij[3], Etotal, Ek, Ep, Eb, 0.0))
  sys.stdout.write('\n')

# print timings and neighbor list information
timers.show(integrator2.getTimers(), precision=3)
sys.stdout.write('Total # of neighbors = %d\n' % vl.totalSize())
sys.stdout.write('Ave neighs/atom = %.1f\n' % (vl.totalSize() / float(num_particles)))
sys.stdout.write('Neighbor list builds = %d\n' % vl.builds)
sys.stdout.write('Integration steps = %d\n' % integrator2.step)
sys.stdout.write('CPUs = %i CPU time per CPU = %.1f\n' % (comm.size,end_time - start_time))

# set all integrator2 timers to zero again 
integrator2.resetTimers()
# set integrator time step to zero again
integrator2.step = 0

########################################################################
# RUN PPA                                                              #
########################################################################
if not skipPPA:
  #if os.path.exists("FLAG_V"):
  #  os.remove("FLAG_V")
  
  system.removeInteraction(2) 
  system.removeInteraction(1) 
  system.removeInteraction(0) 
  langevin.disconnect()
  # the equilibration uses a different interaction cutoff therefore the current
  # verlet list is not needed any more and would waste only CPU time
  
  potLJ = espressopp.interaction.LennardJones(1.0, 1.0, cutoff = rc, shift = "auto")
  interLJ = espressopp.interaction.VerletListLennardJones(vl)
  interLJ.setPotential(type1 = 0, type2 = 0, potential = potLJ)
  system.addInteraction(interLJ)
  
  if (ifbond):
    #potFENE = espressopp.interaction.Harmonic(K=30.0, r0=0.0)
    #interFENE = espressopp.interaction.FixedPairListHarmonic(system, bondlist, potFENE)
    #potFENE = espressopp.interaction.FENECapped(K=30.0, r0=0.0, rMax=1.5, r_cap=1.49)
    #interFENE = espressopp.interaction.FixedPairListFENECapped(system, bondlist, potFENE)
    potFENE = espressopp.interaction.FENE(K=100.0, r0=0.0, rMax=1.5)
    interFENE = espressopp.interaction.FixedPairListFENE(system, bondlist, potFENE)
    system.addInteraction(interFENE)
  ## Cosine with FixedTriple list
  #   potCosine = espressopp.interaction.Cosine(K=1.5, theta0=3.1415926)
  #   interCosine = espressopp.interaction.FixedTripleListCosine(system, anglelist, potCosine)
  #   #interCosine.setPotential(type1 = 0, type2 = 0, potential = potCosine)
  #   system.addInteraction(interCosine)
  
  # fix x,y and z coord axis
  print("Fix head-end of all polymer chains")
  fixMask = espressopp.Int3D(1,1,1)
  # create a particel group that will contain the fixed particles
  fixedWall  = espressopp.ParticleGroup(system.storage)
  
  for nc in range(num_chains):
    fixedWall.add(nc*monomers_per_chain+1)
    fixedWall.add((nc+1)*monomers_per_chain)
    
  system.storage.decompose()
  fixpositions = espressopp.integrator.FixPositions(system, fixedWall, fixMask)
  integrator.addExtension(fixpositions)
  espressopp.tools.analyse.info(system, integrator)
  
  zero_vel = espressopp.Real3D(0.)
  for i in range(1,num_particles+1):
    #i=0
    system.storage.modifyParticle(i, 'v', zero_vel)
    #v = system.storage.getParticle(i).v
    #print "V2: ",v
    #sys.exit(0)
  
  thermostat = espressopp.integrator.LangevinThermostat(system)
  thermostat.gamma = 20.0
  thermostat.temperature = 0.001
  integrator.addExtension(thermostat)
  system.storage.decompose()
  
  file2="traj_ppa.pdb"
  for step in range(20):
    integrator.run(50)
    espressopp.tools.analyse.info(system, integrator)
    if (math.isnan(interFENE.computeEnergy())):
      print("FENE becomes NaN during production")
      sys.exit(0)
    #espressopp.tools.pdb.pdbwrite(file2, system, append=True)
    #espressopp.tools.xyzfilewrite(file2, system, velocities = False, charge = False, append=True, atomtypes={0:'X'})
    
  thermostat.gamma=0.5
  for step in range(200):
    integrator.run(100)
    espressopp.tools.analyse.info(system, integrator)
    if (math.isnan(interFENE.computeEnergy())):
      print("FENE becomes NaN during production")
      sys.exit(0)
    #espressopp.tools.pdb.pdbwrite(file2, system, append=True)
    #espressopp.tools.xyzfilewrite(file2, system, velocities = False, charge = False, append=True, atomtypes={0:'X'})
  
  # post-analysis
  conf = espressopp.analysis.Configurations(system)
  conf.gather()
  r2_sum=0
  cnt=0
  clen=[]
  diam=[]
  rlen=[]
  for i in range(num_chains):
    dim=[0,0,0]
    lsum=0
    #coor=[]
    jj=monomers_per_chain*i+1
    #coor.append(conf[0][jj])
    #coor.append(conf[0][jj])
  
    for j in range(1,monomers_per_chain):
      jj=monomers_per_chain*i+j
      l=conf[0][jj+1]-conf[0][jj]
      if l[2]<-Lz/2.0:
        dim[2]+=1
        l[2]+=Lz
        l[0]+=system.shearOffset
      elif l[2]>Lz/2.0:
        dim[2]-=1
        l[2]-=Lz
        l[0]-=system.shearOffset
      if l[1]<-Ly/2.0:
        dim[1]+=1
        l[1]+=Ly
      elif l[1]>Ly/2.0:
        dim[1]-=1
        l[1]-=Ly
      if l[0]<-Lx/2.0:
        while l[0]<-Lx/2.0:
          dim[0]+=1
          l[0]+=Lx
      elif l[0]>Lx/2.0:
        while l[0]>Lx/2.0:
          dim[0]-=1
          l[0]-=Lx
      lsum+=l.abs()
      #coor.append(conf[0][jj+1]+Real3D(dim[0]*Lx,dim[1]*Ly,dim[2]*Lz))
      #coor[0][0]+=coor[j+1][0]
      #coor[0][1]+=coor[j+1][1]
      #coor[0][2]+=coor[j+1][2]
    clen.append(lsum)
    
    #coor[0][0]/=monomers_per_chain
    #coor[0][1]/=monomers_per_chain
    #coor[0][2]/=monomers_per_chain
    ##print "COOR> ",i,coor
    
    #maxd=0
    #for j in range(1,monomers_per_chain):
    #  dtmp=coor[j]-coor[0]
    #  if (dtmp.abs()>maxd):
    #    maxd=dtmp.abs()
    #diam.append(maxd*2.0)
    
    rtmp=conf[0][monomers_per_chain*(i+1)]-conf[0][monomers_per_chain*i+1]
    rtmp[2]+=float(dim[2])*Lz
    rtmp[0]+=float(dim[2])*system.shearOffset
    rtmp[1]+=float(dim[1])*Ly
    rtmp[0]+=float(dim[0])*Lx
    r2_sum+=rtmp.abs()**2
    rlen.append(rtmp.abs())
    
  
  r2_msq=r2_sum/num_chains
  klen=[]
  elen=[]
  znum=[]
  #tbd=[]
  for i in range(num_chains):
    c2=clen[i]**2
    r2=rlen[i]**2
    klen.append(r2/clen[i])
    elen.append(r2/c2*float(monomers_per_chain-1))
    #tbd.append(math.sqrt(elen[i])*clen[i]/(monomers_per_chain-1))
    znum.append(float(monomers_per_chain)/float(monomers_per_chain-1)*(c2/r2-1))
    
  print("---------------")
  print("R_SQUARE> ",r2_msq)
  print("R_MEAN> ",math.sqrt(r2_msq))
  print("---------------")
  print("CONTOUR> ",clen[0:nc_print])
  print("CT_AVG> ",sum(clen)/float(len(clen)))
  print("---------------")
  print("KUHN> ",klen[0:nc_print])
  print("APP_AVG> ",sum(klen)/float(len(klen)))
  print("---------------")
  bpp=[x/(monomers_per_chain-1) for x in clen]
  print("BPP> ",bpp[0:nc_print])
  print("BPP_AVG> ",sum(bpp)/float(len(bpp)))
  print("---------------")
 #print("TUBED> ",tbd)
 #print("DIAMETER> ",diam)
 #print("---------------")
  print("NENTANG> ",elen[0:nc_print])
  print("N_AVG> ",sum(elen)/float(len(elen)))
  print("---------------")
  print("ZENTANG> ",znum[0:nc_print])
  print("Z_AVG> ",sum(znum)/float(len(znum)))
  print("---------------")
