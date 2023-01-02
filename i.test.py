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
import gc

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
  
def wrap(a,b,Lx,Ly,Lz,offs):
  l=[.0,.0,.0]
  for i in range(3):
    l[i]=b[i]-a[i]
  if l[2]<-Lz/2.0:
    while l[2]<-Lz/2.0:
      l[2]+=Lz
      l[0]+=offs
  elif l[2]>Lz/2.0:
    while l[2]>Lz/2.0:
      l[2]-=Lz
      l[0]-=offs
  if l[1]<-Ly/2.0:
    while l[1]<-Ly/2.0:
      l[1]+=Ly
  elif l[1]>Ly/2.0:
    while l[1]>Ly/2.0:
      l[1]-=Ly
  if l[0]<-Lx/2.0:
    while l[0]<-Lx/2.0:
      l[0]+=Lx
  elif l[0]>Lx/2.0:
    while l[0]>Lx/2.0:
      l[0]-=Lx
  return l

def getCOM(x,n1,n,Lx,Ly,Lz,offs):
  xcom=[x[n1][0],x[n1][1],x[n1][2]]
  p0=[x[n1][0],x[n1][1],x[n1][2]]

  for k in range(n1+1,(n1+n)):
    p1=[x[k][0],x[k][1],x[k][2]]
    l=[.0,.0,.0]
    for i in range(3):
      l[i]=p1[i]-p0[i]
    if l[2]<-Lz/2.0:
      while l[2]<-Lz/2.0:
        l[2]+=Lz
        l[0]+=offs
    elif l[2]>Lz/2.0:
      while l[2]>Lz/2.0:
        l[2]-=Lz
        l[0]-=offs
    if l[1]<-Ly/2.0:
      while l[1]<-Ly/2.0:
        l[1]+=Ly
    elif l[1]>Ly/2.0:
      while l[1]>Ly/2.0:
        l[1]-=Ly
    if l[0]<-Lx/2.0:
      while l[0]<-Lx/2.0:
        l[0]+=Lx
    elif l[0]>Lx/2.0:
      while l[0]>Lx/2.0:
        l[0]-=Lx
    for i in range(3):
      p0[i]+=l[i]
      xcom[i]+=p0[i]

  for i in range(3):
        xcom[i]/=float(n)
  return xcom

# simulation parameters (nvt = False is nve)
rc = pow(2.0, 1.0/6.0)
skin = 0.3
ifVisc = True
dpd = True
ifbond = True
skipPPA=False
shear_rate = 0.001
timestep = 0.002

# set parameters for simulations
seed               = 654321  # seed for random
density            = 0.849
bondlen            = 0.97
monomers_per_chain = 30      # the number of monomers per a chain
num_chains         = 15*monomers_per_chain     # the number of chains
nc_print = max(100,3*monomers_per_chain) #num_chains
temperature        = 1.0     # set temperature

# number of prod loops
prod_nloops       = 200 #200
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
if comm.size>4 and comm.size%4 ==0:
  nodeGrid = espressopp.Int3D(comm.size/4,2,2)
else:
  nodeGrid = espressopp.tools.decomp.nodeGrid(comm.size,size,rc,skin)
#nodeGrid = espressopp.tools.decomp.nodeGrid(n=comm.size,rc=rc,skin=skin)
cellGrid = espressopp.tools.decomp.cellGrid(size,nodeGrid,rc,skin)
system.storage = espressopp.storage.DomainDecomposition(system, nodeGrid, cellGrid)

file = open("equil_20.xyz", 'r')
number_of_particles_in_file  = int(file.readline())
box_volume                   = file.readline().split()

if num_chains * monomers_per_chain != number_of_particles_in_file:
  print("ERROR: wrong number of particles in restart file")
  file.close()
  sys.exit()

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
mass     = 1.0

#bonds = []
#angles = []
for i in range(num_chains):
  polymer_chain       = []
  bonds = []
  angles = []
  
  for k in range(monomers_per_chain):
    col     = file.readline().split()

    if (len(col) == 8 or len(col) == 5):
      part_type = int(col[1])
      part_pos  = espressopp.Real3D(float(col[2]), float(col[3]), float(col[4]))
      part_vel  = espressopp.Real3D(float(col[5]), float(col[6]), float(col[7]))
	
    elif (len(col) == 7 or len(col) == 4):
      part_type = 0
      part_pos  = espressopp.Real3D(float(col[1]), float(col[2]), float(col[3]))
      part_vel  = espressopp.Real3D(float(col[4]), float(col[5]), float(col[6]))

    particle   = [pid+k, part_type, mass, part_pos, part_vel]
    polymer_chain.append(particle)
    
    #if i==0:
    #  print(particle)
    if k < monomers_per_chain-1:
      bonds.append((pid+k,pid+k+1))
	
    if k < monomers_per_chain-2:
      angles.append((pid+k, pid+k+1, pid+k+2))
    
  #print(len(col),angles[0])

  system.storage.addParticles(polymer_chain, *props)
  system.storage.decompose()
  bondlist.addBonds(bonds)
  anglelist.addTriples(angles)
  
  pid += monomers_per_chain

file.close()
system.storage.decompose()
del bonds
del angles
#sys.exit(0)

# Lennard-Jones with Verlet list
vl = espressopp.VerletList(system, cutoff = rc + system.skin/2.0)
potLJ = espressopp.interaction.LennardJones(1.0, 1.0, cutoff = rc, shift = "auto")
interLJ = espressopp.interaction.VerletListLennardJones(vl)
interLJ.setPotential(type1 = 0, type2 = 0, potential = potLJ)
system.addInteraction(interLJ)

if (ifbond):
  print("Num_BondList  = ", bondlist.size())
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
  thermo.gamma = 001.0
  thermo.tgamma = 0.0
  thermo.temperature = temperature
  integrator.addExtension(thermo)
else:
  thermo = espressopp.integrator.LangevinThermostat(system)
  thermo.gamma = 001.0
  thermo.temperature = temperature
  integrator.addExtension(thermo)
  
  
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

#espressopp.tools.pdb.pdbwrite('input.pdb', system, append=True)
#espressopp.tools.analyse.info(system, integrator)
#integrator.run(0)
#espressopp.tools.analyse.info(system, integrator)
#sys.exit(0)

#Equilibration
print("starting equilibration (TEST) ...")
espressopp.tools.analyse.info(system, integrator)
for step in range(200):
  integrator.run(50)
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

#Preparation (MSD)
pos_cur=[] #mid point of chain
pos_sav=[]
pcm_sav=[] #com of chain
pcm_t0=[]
del_t=timestep*float(prod_isteps)
print("DELTA_T: ",del_t)
conf  = espressopp.analysis.Configurations(system)
conf.capacity=1
#conf.gather()
dpl=[.0]*(num_chains*3)
d_stream=[.0]*num_chains
dtr_bin=50
dtr_xx=[.0]*dtr_bin
dtr_cnt=[0]*dtr_bin
dtr_bwitdh=shear_rate*timestep*float(prod_isteps)*Lz/20.0
print("BWITDH: ",dtr_bwitdh)
start_time = time.process_time()
nstep_div100=prod_nloops/100
nstep_div20=nstep_div100*5

vel = espressopp.analysis.Velocities(system)
vel.capacity=1
#vel.gather()
zbin=50
dz=Lz/float(zbin)
zvol=dz*Lx*Ly
rstep=int(0.8*prod_nloops) #report after simulation run by 80%
for step in range(prod_nloops+1):
  if step == 0:
    conf.gather()
    for k in range(num_chains):
      #determine the xyz of central monomer
      if monomers_per_chain%2==1:
        cid1=int(monomers_per_chain*k+(monomers_per_chain+1)/2+0.00001)
      else:
        cid1=int(monomers_per_chain*k+monomers_per_chain/2+0.00001)
      pos_cur.append(conf[0][cid1])
      pos_sav.append(conf[0][0])
      if monomers_per_chain%2==0:
        cid2=cid1+1
        l=wrap(conf[0][cid1],conf[0][cid2],Lx,Ly,Lz,system.shearOffset)
        for i in range(3):
          pos_cur[k][i]+=l[i]/2.0
      #determine the xyz of CoM
      ptmp=getCOM(conf[0],k*monomers_per_chain+1,monomers_per_chain,Lx,Ly,Lz,system.shearOffset)
      l=wrap(pos_cur[k],ptmp,Lx,Ly,Lz,system.shearOffset)
      pcm_sav.append([pos_cur[k][0]+l[0],pos_cur[k][1]+l[1],pos_cur[k][2]+l[2]])
      pcm_t0.append([pcm_sav[k][0],pcm_sav[k][1],pcm_sav[k][2]])
      #print("CHAIN %d STEP %d [%.4f %.4f %.4f] (%.4f %.4f %.4f) / %.4f %.4f" %(k,step,pos_cur[k][0],pos_cur[k][1],pos_cur[k][2],pos_cur[k][0]+l[0],pos_cur[k][1]+l[1],pos_cur[k][2]+l[2],dpl[k*3+0],0.0))
#      print(pcm_sav[1],system.storage.getParticle(30).pos)
      del ptmp
      del l

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
    
    conf.gather()
    # calculate MSD for central monomers (initialize)
    gxx=0.0
    gyy=0.0
    gzz=0.0
    gxz=0.0
    #d_stream=[]
    for k in range(num_chains):
      for i in range(3):
        pos_sav[k][i]=pos_cur[k][i]

      if monomers_per_chain%2==1:
        cid1=int(monomers_per_chain*k+(monomers_per_chain+1)/2+0.00001)
      else:
        cid1=int(monomers_per_chain*k+monomers_per_chain/2+0.00001)
      for i in range(3):
        pos_cur[k][i]=conf[0][cid1][i]
      if monomers_per_chain%2==0:
        cid2=cid1+1
        l=wrap(conf[0][cid1],conf[0][cid2],Lx,Ly,Lz,system.shearOffset)
        for i in range(3):
          pos_cur[k][i]+=l[i]/2.0
      #caculate the displacement of central monomers for prod_isteps
      l=wrap(pos_sav[k],pos_cur[k],Lx,Ly,Lz,system.shearOffset)
      for i in range(3):
        dpl[k*3+i]+=l[i]
        pos_cur[k][i]=pos_sav[k][i]+l[i]
      #determine CoM for the current step (in the same image space at t'=t-dt)
      ptmp=getCOM(conf[0],k*monomers_per_chain+1,monomers_per_chain,Lx,Ly,Lz,system.shearOffset)
      lcm=wrap(pcm_sav[k],ptmp,Lx,Ly,Lz,system.shearOffset)
      for i in range(3):
        ptmp[i]=pcm_sav[k][i]+lcm[i]
      #calculate shear contribution in x propagation
      d_stream[k]+=shear_rate*del_t*(ptmp[2]-Lz/2.0+lcm[2]*0.5)
      #print("CHAIN %d STEP %d [%.4f %.4f %.4f] (%.4f %.4f %.4f) / %.4f %.4f" %(k,step,pos_cur[k][0],pos_cur[k][1],pos_cur[k][2],ptmp[0],ptmp[1],ptmp[2],dpl[k*3+0],d_stream[k]))
#      dpl[k*3+0]-=shear_rate*del_t*(ptmp[2]-Lz/2.0+lcm[2]*0.5)
#      dpl[k*3+0]-=shear_rate*del_t*(ptmp[2]-Lz/2.0+lcm[2]*float(step-1))
      for i in range(3): pcm_sav[k][i]=ptmp[i]
      del l
      del lcm
      del ptmp

      xtmp=dpl[k*3+0]-d_stream[k]
      #check equi-distribution of GXX at diff stream layers
#      if step==prod_nloops:
      if step%nstep_div20==0:
        bi=int(abs(d_stream[k])/dtr_bwitdh/float(step))
        if bi<dtr_bin:
          dtr_xx[bi]+=xtmp*xtmp
          dtr_cnt[bi]+=1
      gxx+=xtmp*xtmp
      gyy+=dpl[k*3+1]*dpl[k*3+1]
      gzz+=dpl[k*3+2]*dpl[k*3+2]
      gxz+=xtmp*dpl[k*3+2]
    print("MSD> %.3f %.6f" %(step*timestep*prod_isteps,(gxx+gyy+gzz)/float(num_chains)))
#    print("G00> %.3f %.6f" %(step*timestep*prod_isteps,d0*d0))
    print("GXX> %.3f %.6f" %(step*timestep*prod_isteps,gxx/float(num_chains)))
    print("GYY> %.3f %.6f" %(step*timestep*prod_isteps,gyy/float(num_chains)))
    print("GZZ> %.3f %.6f" %(step*timestep*prod_isteps,gzz/float(num_chains)))
    print("GXZ> %.3f %.6f" %(step*timestep*prod_isteps,gxz/float(num_chains)))
#    sys.exit(0)
#    if step==prod_nloops:
    if step%nstep_div20==0:
#      espressopp.tools.pdb.pdbwrite("traj_md.pdb", system, append=True)
      for i in range(dtr_bin):
        if dtr_cnt[i]>0:  
          print("STREAM> %d %.2f %.2f %d" %(step,float((i+0.5)*0.1),dtr_xx[i]/float(dtr_cnt[i]),dtr_cnt[i]))
        else:
          print("STREAM> %d %.2f %.2f %d" %(step,float((i+0.5)*0.1),.0,0))
        dtr_xx[i]=.0
        dtr_cnt[i]=0
#      sys.exit(0)
    if step%nstep_div100==0:
      r2_sum=.0
      for i in range(num_chains):
        dim=[0,0,0]
        jj=monomers_per_chain*i+1
      
        for j in range(1,monomers_per_chain):
          jj=monomers_per_chain*i+j
          l=conf[0][jj+1]-conf[0][jj]
          if l[2]<-Lz/2.0:
            while l[2]<-Lz/2.0:
              dim[2]+=1
              l[2]+=Lz
              l[0]+=system.shearOffset
          elif l[2]>Lz/2.0:
            while l[2]>Lz/2.0:
              dim[2]-=1
              l[2]-=Lz
              l[0]-=system.shearOffset
          if l[1]<-Ly/2.0:
            while l[1]<-Ly/2.0:
              dim[1]+=1
              l[1]+=Ly
          elif l[1]>Ly/2.0:
            while l[1]>Ly/2.0:
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
        
        rtmp=conf[0][monomers_per_chain*(i+1)]-conf[0][monomers_per_chain*i+1]
        rtmp[2]+=float(dim[2])*Lz
        rtmp[0]+=float(dim[2])*system.shearOffset
        rtmp[1]+=float(dim[1])*Ly
        rtmp[0]+=float(dim[0])*Lx
        r2_sum+=rtmp.abs()**2
      r2_msq=r2_sum/num_chains
      print("R_ETE> %d %.2f" %(step,math.sqrt(r2_msq)))
      #sys.exit(0)
    
    #calculate VX
    if step>=rstep:
      vel.gather()
      # calculate z-layer profiles
      znum=[0]*zbin
      vx=[.0]*zbin
      for k in range(num_particles):
        zi=math.floor(conf[0][k][2]/dz)
        if zi>zbin-1:
          zi=zbin-1
        vshear=shear_rate*(conf[0][k][2]-Lz/2.0)
        znum[zi]+=1
        if shear_rate > .0:
          vx[zi]+=vel[0][k][0]+vshear
      for z in range(zbin):
        if znum[z]>0:
          if shear_rate > .0:
            vx[z]/=float(znum[z])*shear_rate*Lz/2.0
        zpos=float(z+0.5)/float(zbin)
        print("VX> %.3f %.6f" %(zpos,vx[z]))
  conf.clear()
#  gc.collect()
end_time = time.process_time()
print("production finished")

#delete
del pos_cur
del pos_sav
del pcm_sav
del conf
del dpl
#gc.collect()

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
  thermo.disconnect()
  # the equilibration uses a different interaction cutoff therefore the current
  # verlet list is not needed any more and would waste only CPU time
  
#  exlist=[]
#  for nc in range(num_chains):
#    for i in range(nc*monomers_per_chain+1,(nc+1)*monomers_per_chain):
#      for j in range(i+1,(nc+1)*monomers_per_chain+1):
#        exlist.append((i,j))
#  vl.exclude(exlist)
  potLJ = espressopp.interaction.LennardJones(1.0, 1.0, cutoff = rc, shift = "auto")
  interLJ = espressopp.interaction.VerletListLennardJones(vl)
  interLJ.setPotential(type1 = 0, type2 = 0, potential = potLJ)
  system.addInteraction(interLJ)
  system.lebcMode = -9
  
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
  #espressopp.tools.pdb.pdbwrite(file2, system, append=False)
  for step in range(20):
    integrator.run(50)
    espressopp.tools.analyse.info(system, integrator)
    if (math.isnan(interFENE.computeEnergy())):
      print("FENE becomes NaN during production")
      sys.exit(0)
    #if step%4==3:
    #  espressopp.tools.pdb.pdbwrite(file2, system, append=True)
    #espressopp.tools.xyzfilewrite(file2, system, velocities = False, charge = False, append=True, atomtypes={0:'X'})
    
  thermostat.gamma=0.5 
  for step in range(200):
    integrator.run(100)
    espressopp.tools.analyse.info(system, integrator)
    if (math.isnan(interFENE.computeEnergy())):
      print("FENE becomes NaN during production")
      sys.exit(0)
    #if step%20==19:
    #  espressopp.tools.pdb.pdbwrite(file2, system, append=True)
    #espressopp.tools.xyzfilewrite(file2, system, velocities = False, charge = False, append=True, atomtypes={0:'X'})
  
  # post-analysis
  conf = espressopp.analysis.Configurations(system)
  conf.gather()
  r2_sum=0
  cnt=0
  clen=[]
  diam=[]
  rlen=[]
  kink_60  = []
  kink_90  = []
  kink_all = []
  for i in range(num_chains):
    dim=[0,0,0]
    lsum=0
    #coor=[]
    jj=monomers_per_chain*i+1
    #coor.append(conf[0][jj])
    #coor.append(conf[0][jj])
    
    bvec=[]
    agl_jj5=np.array([])
  
    for j in range(1,monomers_per_chain):
      jj=monomers_per_chain*i+j
      l=conf[0][jj+1]-conf[0][jj]
      if l[2]<-Lz/2.0:
        while l[2]<-Lz/2.0:
          dim[2]+=1
          l[2]+=Lz
          l[0]+=system.shearOffset
      elif l[2]>Lz/2.0:
        while l[2]>Lz/2.0:
          dim[2]-=1
          l[2]-=Lz
          l[0]-=system.shearOffset
      if l[1]<-Ly/2.0:
        while l[1]<-Ly/2.0:
          dim[1]+=1
          l[1]+=Ly
      elif l[1]>Ly/2.0:
        while l[1]>Ly/2.0:
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
      
      bvec.append(l)
      if j>5:
        agl_jj5=np.append(agl_jj5,vec_angle(bvec[j-6],bvec[j-1]))
    
    local_max=argrelextrema(agl_jj5, np.greater)
    cnt_60=0
    cnt_90=0
    cnt_all=0
    for k in range(len(local_max[0])):
      if agl_jj5[local_max[0][k]]>60.0:
        cnt_60+=1
        if agl_jj5[local_max[0][k]]>90.0:
          cnt_90+=1
      cnt_all+=1
    kink_60.append(cnt_60)
    kink_90.append(cnt_90)
    kink_all.append(cnt_all)
    del agl_jj5
    del bvec
    
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
    
#Summarize the analysis
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
  print("OFFSET> ",system.shearOffset)
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
  print("KINKS> ",kink_60[0:nc_print])
  print("K_AVG> ",sum(kink_60)/float(len(kink_60)),sum(kink_90)/float(len(kink_90)),sum(kink_all)/float(len(kink_all)))
  print("---------------")
