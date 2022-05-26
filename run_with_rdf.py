#!/usr/bin/env python2
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

def write_gro(filename, system, typenames, time):
    # only use for the CG model of BMIM-PF6
    out_stream = open(filename, 'w')
    out_stream.write("CG BMIM-PF6 t=" + str(time) + " \n")
    max_pid = int(espressopp.analysis.MaxPID(system).compute())
    out_stream.write(str(max_pid) + "\n")
    
    mol_num = 1
    
    for i in range(max_pid):
        pid = i+1
        particle = system.storage.getParticle(pid)
        xpos   = particle.pos[0]
        ypos   = particle.pos[1]
        zpos   = particle.pos[2]
        if particle.type != 4:
            out_stream.write('{:>5}'.format(mol_num) + '{:<5}'.format("BMI") + '{:>5}'.format(typenames[particle.type]) 
                             + '{:>5}'.format(pid) + '{:>8}'.format("%.3f"%(xpos)) 
                             + '{:>8}'.format("%.3f"%(ypos)) + '{:>8}'.format("%.3f"%(zpos)) + '\n')
            if particle.type == 3:
                mol_num += 1
        else: 
            out_stream.write('{:>5}'.format(mol_num) + '{:<5}'.format("PF6") + '{:>5}'.format(typenames[particle.type]) 
                             + '{:>5}'.format(pid) + '{:>8}'.format("%.3f"%(xpos)) 
                             + '{:>8}'.format("%.3f"%(ypos)) + '{:>8}'.format("%.3f"%(zpos)) + '\n')
            mol_num += 1 
    out_stream.write(str(system.bc.boxL[0]) + ' ' + str(system.bc.boxL[1]) + ' ' + str(system.bc.boxL[2]) + ' ' +
    str(0) + ' ' + str(0) + ' ' + str(0) + ' ' +
    str(0) + ' ' + str(shear_rate*time*system.bc.boxL[0]) + ' ' + str(0)+"\n")
    out_stream.close()   

bool_msd=False
bool_ori=False
bool_rdf=True
if os.path.exists("FLAG_VIS"):
    bool_visc=True
# simulation parameters (nvt = False is nve)
rc    = 1.5  # Verlet list cutoff
skin  = 0.3
timestep = 0.001
tot_types= 5 # The total num of atomic types, 5 (0~4) in this case

# add shear rate / NSteps
shear_rate = 0.010
equi_nloops = 200  # total steps = nloops * isteps
equi_isteps = 50
# number of prod loops
prod_nloops       = 5000 #10 ns
# number of integration steps performed in each production loop
prod_isteps       = 100
# set parameters for calculating partial g(r)
if bool_rdf: 
    rdf_pair = [[2,4],[3,4],[4,4]] # input wanted type pairs (starting from Type-0 ...)
    rdf_dr = 0.01 # width for each ridial shell
    rdf_nlast=10000 # The last $rdf_nlast steps to be sampled and averaged

# GROMACS tabulated potentials files
tabCT_CTg = "table_CT_CT.xvg"    # non-bonded
tabCT_PFg = "table_CT_PF.xvg"
tabI1_CTg = "table_I1_CT.xvg"
tabI1_I1g = "table_I1_I1.xvg"
tabI1_I2g = "table_I1_I2.xvg"
tabI1_I3g = "table_I1_I3.xvg"
tabI1_PFg = "table_I1_PF.xvg"
tabI2_CTg = "table_I2_CT.xvg"
tabI2_I2g = "table_I2_I2.xvg"
tabI2_I3g = "table_I2_I3.xvg"
tabI2_PFg = "table_I2_PF.xvg"
tabI3_CTg = "table_I3_CT.xvg"
tabI3_I3g = "table_I3_I3.xvg"
tabI3_PFg = "table_I3_PF.xvg"
tabPF_PFg = "table_PF_PF.xvg"

taba0g = "table_a0.xvg"     # angles, not used
taba1g = "table_a1.xvg"
taba2g = "table_a2.xvg"
taba3g = "table_a3.xvg"  
taba4g = "table_a4.xvg"

tabb0g  = "table_b0.xvg"    # bonds, not used
tabb1g  = "table_b1.xvg"
tabb2g  = "table_b2.xvg"
tabb3g  = "table_b3.xvg"

##tabd6g = "table_d6.xvg"     # dihedrals
##tabd7g = "table_d7.xvg"
##tabd8g = "table_d8.xvg"
##tabd9g = "table_d9.xvg"

spline = 3                  # spline interpolation type (1, 2, 3) [linear, Akima or cubic splines]

tabfilesnb = [
tabI1_I1g, tabI1_I2g, tabI1_I3g, tabI1_CTg, tabI1_PFg, 
tabI2_I2g, tabI2_I3g, tabI2_CTg, tabI2_PFg, 
tabI3_I3g, tabI3_CTg, tabI3_PFg, tabCT_CTg, tabCT_PFg, tabPF_PFg]
#tabfiles2b = [tabb0g, tabb1g, tabb5g, tabb12g, tabb17g, tabb19g, tabb40g]
#tabfiles3b = [taba1g, taba2g, taba3g, taba4g, taba5g]
#tabfiles4b = [tabd6g, tabd7g, tabd8g, tabd9g]


# parameters to convert GROMACS tabulated potential file
sigma = 1.0
epsilon = 1.0
c6 = 1.0
c12 = 1.0

# GROMACS setup files
grofile = "start.gro"
topfile = "topology.top"

# Generate dictonary of potentials for all particle types (relies on table file names).
# Input is list of gromacs tabulated non-bonded potentials file names
# Output is something like {"A_A":potAA, "A_B":potAB, "B_B":potBB}

def genTabPotentials(tabfilesnb):
    potentials = {}
    for fg in tabfilesnb:
        fe = fg.split(".")[0]+".tab" # name of espressopp file
        gromacs.convertTable(fg, fe, sigma, epsilon, c6, c12)
        pot = espressopp.interaction.Tabulated(itype=spline, filename=fe, cutoff=rc)
        t1, t2 = fg[6:8], fg[9:11] # type 1, type 2
        potentials.update({t1+"_"+t2: pot})
        if t1 != t2:
            potentials.update({t2+"_"+t1: pot})
    return potentials

# define types of particles (used for non-bonded interactions)
#particleTypes = {"A":["A1m", "A2m", "A1r", "A2r"],"B":["B1u","B2u", "B1d", "B2d"]}

particleTypes = {"I1":["I1"], "I3":["I3"], "I2":["I2"], "CT":["CT"], "PF":["PF"]}
num_types = 5
potentials = genTabPotentials(tabfilesnb)


######################################################################
##  IT SHOULD BE UNNECESSARY TO MAKE MODIFICATIONS BELOW THIS LINE  ##
######################################################################
#defaults, types, atomtypes, masses, charges, atomtypeparameters, bondtypes, bondtypeparams, angletypes, angletypeparams, dihedraltypes, dihedraltypeparams, exclusions, x, y, z, vx, vy, vz, resname, resid, Lx, Ly, Lz =gromacs.read(grofile, topfile)
defaults, types, atomtypes, masses, charges, atomtypeparameters, bondtypes, bondtypeparams, angletypes, angletypeparams, exclusions, x, y, z, vx, vy, vz, resname, resid, Lx, Ly, Lz =gromacs.read(grofile,topfile)

num_particles = len(x)
density = num_particles / (Lx * Ly * Lz)
size = (Lx, Ly, Lz)

# Create random seed
tseed = int( time.time() * 1000.0 )
random.seed( ((tseed & 0xff000000) >> 24) +
             ((tseed & 0x00ff0000) >>  8) +
             ((tseed & 0x0000ff00) <<  8) +
             ((tseed & 0x000000ff) << 24)   )
irand=random.randint(1,99999)

# Setup obj of system
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
    allParticles.append(part)
system.storage.addParticles(allParticles, *props)    
system.storage.decompose()



# Tabulated Verlet list for non-bonded interactions
print(potentials)
print(particleTypes)
vl = espressopp.VerletList(system, cutoff = rc + system.skin)
internb = espressopp.interaction.VerletListTabulated(vl)
gromacs.setTabulatedInteractions(potentials, particleTypes, system, internb)

vl.exclude(exclusions)

# Create an effective verlet list which excludes bonds, angles, etc.
#vl_eff = espressopp.VerletList(system, cutoff=rc + system.skin,exclusionlist=exclusions)
###print vl_eff.getAllPairs()

#qq_interactions=gromacs.setCoulombInteractions(system, vl_lj, rc, types, epsilon1=1, epsilon2=80, kappa=0)
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
for i in range(tot_types-1):
  if i!=3:
    for j in range(i, tot_types-1):
      if j!=3:
        coulombR_intBonded.setPotential(type1=i, type2=j, potential=coulombR_potBonded)
system.addInteraction(coulombR_intBonded) # cancelling self energies for interatomic interactions

coulombR_potEwald = espressopp.interaction.CoulombRSpace(coulomb_prefactor, alphaEwald, rspacecutoff)
coulombR_intEwald = espressopp.interaction.VerletListCoulombRSpace(vl)
for i in range(tot_types):
  if i!=3:
    for j in range(i, tot_types):
      if j!=3:
        #print "I-J: ",i,j
        coulombR_intEwald.setPotential(type1=i, type2=j, potential = coulombR_potEwald)
system.addInteraction(coulombR_intEwald)

coulombK_potEwald = espressopp.interaction.CoulombKSpaceEwald(system, coulomb_prefactor, alphaEwald, kspacecutoff)
coulombK_intEwald = espressopp.interaction.CellListCoulombKSpaceEwald(system.storage, coulombK_potEwald)
system.addInteraction(coulombK_intEwald)


# bonded 2-body interactions
bondedinteractions=gromacs.setBondedInteractions(system, bondtypes, bondtypeparams)
# alternative, manual way for setting up bonded interactions
#for k, v in bonds.iteritems(): # k is number of potential table, v is bondlist
#    fpl = espressopp.FixedPairList(system.storage)
#    fpl.addBonds(v)
#    fg = "table_b"+k+".xvg"
#    fe = fg.split(".")[0]+".tab" # name of espressopp file
#    gromacs.convertTable(fg, fe, sigma, epsilon, c6, c12)
#    potTab = espressopp.interaction.Tabulated(itype=spline, filename=fe)
#    interb = espressopp.interaction.FixedPairListTabulated(system, fpl, potTab)
#    system.addInteraction(interb)


# bonded 3-body interactions
angleinteractions=gromacs.setAngleInteractions(system, angletypes, angletypeparams)
# alternative, manual way for setting up angular interactions
#for k, v in angles.iteritems(): # k is number of potential table, v is anglelist
#    ftl = espressopp.FixedTripleList(system.storage)
#    ftl.addTriples(v)
#    fg = "table_a"+k+".xvg"
#    fe = fg.split(".")[0]+".tab" # name of espressopp file
#    gromacs.convertTable(fg, fe, sigma, epsilon, c6, c12)
#    potTab = espressopp.interaction.TabulatedAngular(itype=spline, filename=fe)
#    intera = espressopp.interaction.FixedTripleListTabulatedAngular(system, ftl, potTab)
#    system.addInteraction(intera)
    

## bonded 4-body interactions
#dihedralinteractions=gromacs.setDihedralInteractions(system, dihedraltypes, dihedraltypeparams)
## alternative, manual way for setting up dihedral interactions:
##for k, v in dihedrals.iteritems(): # k is number of potential table, v is anglelist
##    fql = espressopp.FixedQuadrupleList(system.storage)
##    fql.addQuadruples(v)
##    fg = "table_d"+k+".xvg"
##    fe = fg.split(".")[0]+".tab" # name of espressopp file
##    gromacs.convertTable(fg, fe, sigma, epsilon, c6, c12)
##    potTab = espressopp.interaction.TabulatedDihedral(itype=spline, filename=fe)
##    interd = espressopp.interaction.FixedQuadrupleListTabulatedDihedral(system, fql, potTab)
##    system.addInteraction(interd)


# langevin thermostat
langevin = espressopp.integrator.LangevinThermostat(system)
langevin.gamma = 1.0
langevin.temperature = 2.4942 # kT in gromacs units 
integrator = espressopp.integrator.VelocityVerlet(system)
integrator.addExtension(langevin)
integrator.dt = timestep

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
print('')

##print(types)
#integrator.run(0)
#espressopp.tools.analyse.info(system, integrator)
#sys.exit(0)

## minimization
#steepest = espressopp.integrator.MinimizeEnergy(system, gamma=0.001, ftol=0.001, max_displacement=0.001, variable_step_flag=False)
#espressopp.tools.analyse.info(system, steepest)
#for k in range(20):
#  steepest.run(10)
#  espressopp.tools.analyse.info(system, steepest)
##sys.exit(0)
# analysis
#configurations = espressopp.analysis.Configurations(system)
#configurations.gather()
temperature = espressopp.analysis.Temperature(system)
#pressure = espressopp.analysis.Pressure(system)
#pressureTensor = espressopp.analysis.PressureTensor(system)

#Equilibration
print("starting equilibration ...")
espressopp.tools.analyse.info(system, integrator)
for step in range(equi_nloops):
  integrator.run(equi_isteps)
  espressopp.tools.analyse.info(system, integrator)
print("equilibration finished")
integrator.resetTimers()

fmt = '%5d %8.4f %11.4f %11.4f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n'

T = temperature.compute()
#P = pressure.compute()
P = 0
#Pij = pressureTensor.compute()
Pij = [0,0,0,0,0,0]
Ek = 0.5 * T * (3 * num_particles)
Ep = internb.computeEnergy()
Eb, Ea, Ed=0,0,0
for bd in bondedinteractions.values():Eb+=bd.computeEnergy()
for ang in angleinteractions.values(): Ea+=ang.computeEnergy()
#for dih in dihedralinteractions.values(): Ed+=dih.computeEnergy()
EQQ= coulombR_intBonded.computeEnergy()+coulombR_intEwald.computeEnergy()+coulombK_intEwald.computeEnergy()

Etotal = Ek + Ep + EQQ + Eb + Ea + Ed
sys.stdout.write(' step     T          P          Pxy        etotal      ekinetic      epair         ecoul         ebond       eangle       edihedral\n')
sys.stdout.write(fmt % (0, T, P, Pij[3], Etotal, Ek, Ep, EQQ, Eb, Ea, Ed))

# switch to a second integrator before starting shear flow simulation
if (shear_rate>0.0):
  integrator2     = espressopp.integrator.VelocityVerletLE(system,shear=shear_rate)
else:
  integrator2     = espressopp.integrator.VelocityVerlet(system)
# set the integration step  
integrator2.dt  = timestep
integrator2.step = 0
integrator2.addExtension(langevin)
# since the interaction cut-off changed the size of the cells that are used
# to speed up verlet list builds should be adjusted accordingly 
#system.storage.cellAdjust()

first_traj = True
os.system("mkdir -p out")
out_energy = open("out/energy.dat", 'w')
out_energy.write(' step T P Pxx Pyy Pzz Pxy Pxz Pyz etotal ekinetic epair ecoul ebond eangle edihedral\n')
out_energy.write(str(0) + " " + str(T/constants.k/constants.N_A*1000) + " " +  str(P) + " " +  str(Pij[0]) + " " +  str(Pij[1]) + " " +  str(Pij[2]) + " " +  str(Pij[3]) + " " +  str(Pij[4]) + " " +  str(Pij[5]) + " " +  str(Etotal) + " " +  str(Ek) + " " +  str(Ep) + " " +  str(EQQ) + " " +  str(Eb) + " " +  str(Ea) + " " +  str(Ed) + " \n")


print("starting production ...")
start_time = time.process_time()
#uncomment pdbwrite below to print trajectories
#espressopp.tools.pdb.pdbwrite("traj.pdb", system, append=False, typenames={0:'O', 1:'N', 2:'C', 3:'S', 4:'P'})

#Preparation of analysis
if bool_ori or bool_rdf:
  pos = espressopp.analysis.Configurations(system)
  pos.capacity=1
  
if bool_ori:
  vel = espressopp.analysis.Velocities(system)
  bin_size=50
  mass=1 #here assume all mass=1
  vel.capacity=1
  
if bool_msd:
  conf  = espressopp.analysis.Configurations(system)
  conf.capacity=2
  conf.gather()
  dpl=[0.0,0.0,0.0]
  dpl=dpl*(num_particles+1)
  num_residues=num_particles/5
  #dpl=list(dpl)

if bool_rdf:
    list_t = []
    type_end = []
    type_end.append(0)
    for k in range(num_types):
        for pid in range(num_particles):
            if types[pid]==k:
                list_t.append(pid+1)
        type_end.append(len(list_t))
    #print("RDF-TYPE: ",type_end)
    rdf_maxr = min(Lx,Ly,Lz)
    rdf_bin = int(0.5*rdf_maxr/rdf_dr)
    #print("RDF-BIN: ",rdf_bin,rdf_maxr)
    rdf_array = []
    for k in range(len(rdf_pair)):
        arr_tmp=[0]*rdf_bin
        rdf_array.append(arr_tmp)
        del arr_tmp

for step in range(prod_nloops+1):
    currentStep=step*prod_isteps
    if step>0:
       integrator2.run(prod_isteps) # print out every steps/check steps
       if bool_visc:
         print("SIGXZ> %d %.6f" % (step*prod_isteps,system.sumP_xz))
    
    if bool_msd:
       conf.gather()
       #print("#1 - ",conf[1][1],conf[1][1280])
       #print("#0 - ",conf[0][1],conf[0][1280])
       msd_I2=[0.0,0.0,0.0]
       msd_PF=[0.0,0.0,0.0]
       for k in range(1,num_particles+1):
         for dim in range(3):
           tmp=conf[0][k][dim]-conf[1][k][dim]+0.5*size[dim]
           fl=math.floor((tmp)/size[dim])
           dpl[k*3+dim]=dpl[k*3+dim]+tmp-fl*size[dim]-0.5*size[dim]
           if k > 1024:
             msd_PF[dim]=msd_PF[dim]+dpl[k*3+dim]*dpl[k*3+dim]
           elif k%4 == 3:
             msd_I2[dim]=msd_I2[dim]+dpl[k*3+dim]*dpl[k*3+dim]
       print("MSD_I2_D> ",currentStep*timestep,(msd_I2[0]+msd_I2[1]+msd_I2[2])/num_residues)
       print("MSD_I2_Y> ",currentStep*timestep,msd_I2[1]/num_residues)
       print("MSD_I2_Z> ",currentStep*timestep,msd_I2[2]/num_residues)
       print("MSD_PF_D> ",currentStep*timestep,(msd_PF[0]+msd_PF[1]+msd_PF[2])/num_residues)
       print("MSD_PF_Y> ",currentStep*timestep,msd_PF[1]/num_residues)
       print("MSD_PF_Z> ",currentStep*timestep,msd_PF[2]/num_residues)
       
       
    #collect end-to-end orientation (normailized)
    #ori=[[.0,.0,.0]]*num_chains
    if bool_ori or bool_rdf:
      pos.gather()
      
    if bool_ori:
	  print("DON'T USE: shearOffset errors")
	  sys.exit(0)
      monomers_per_residue=4
      for j in range(1,1024+1):
        mono_idx=(j-1)%monomers_per_residue
        residue_idx=(j-1-mono_idx)/monomers_per_residue
        if residue_idx<num_residues:
          if mono_idx==2: #I2
            ori=Real3D(.0,.0,.0)
            dim=[0,0,0]
            ori-=pos[0][j]
            idx_I2=j
          elif mono_idx==3: #CT
            l=pos[0][j]-pos[0][idx_I2]
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
              
            ori+=pos[0][j]
            ori+=Real3D(dim[0]*Lx,dim[1]*Ly,dim[2]*Lz)
            ori_abs=ori.abs()
            print("CHAIN-%0d> %10d %.6f %.6f %.6f" %(residue_idx,currentStep,ori[0]/ori_abs,ori[1]/ori_abs,ori[2]/ori_abs))
            del ori
    
    T = temperature.compute()
    #P = pressure.compute()
    #Pij = pressureTensor.compute()
    Ek = 0.5 * T * (3 * num_particles)
    Ep = internb.computeEnergy()
    EQQ= coulombR_intBonded.computeEnergy()+coulombR_intEwald.computeEnergy()+coulombK_intEwald.computeEnergy()
        
    Eb, Ea, Ed=0,0,0
    for bd in bondedinteractions.values():Eb+=bd.computeEnergy()
    for ang in angleinteractions.values(): Ea+=ang.computeEnergy()
    #for dih in dihedralinteractions.values(): Ed+=dih.computeEnergy()
    Etotal = Ek + Ep + EQQ + Eb + Ea + Ed
    sys.stdout.write(fmt % (step*(prod_isteps), T, P, Pij[3], Etotal, Ek, Ep, EQQ, Eb, Ea, Ed))
    out_energy.write(str(step*(prod_isteps)) + " " + str(T/constants.k/constants.N_A*1000) + " " +  str(P) + " " +  str(Pij[0]) + " " +  str(Pij[1]) + " " +  str(Pij[2]) + " " +  str(Pij[3]) + " " +  str(Pij[4]) + " " +  str(Pij[5]) + " " +  str(Etotal) + " " +  str(Ek) + " " +  str(Ep) + " " +  str(EQQ) + " " +  str(Eb) + " " +  str(Ea) + " " +  str(Ed) + " \n")


    #espressopp.tools.pdb.pdbwrite("traj.pdb", system, append=True, typenames={0:'O', 1:'N', 2:'C', 3:'S', 4:'P'})
    
    if bool_rdf and step>(prod_nloops-rdf_nlast):
        for p in range(len(rdf_pair)):
            itype=rdf_pair[p][0]
            jtype=rdf_pair[p][1]
            
            istart=type_end[itype]
            ilast=type_end[itype+1]
            
            cnt=0
            for i in range(istart,ilast):
                idx=list_t[i]
                
                jlast=type_end[jtype+1]
                jstart=type_end[jtype]
                    
                for j in range(jstart,jlast):
                    jdx=list_t[j]
                    if idx!=jdx:
                        l=pos[0][jdx]-pos[0][idx]
                        if l[2]<-Lz/2.0:
                          l[2]+=Lz
                          l[0]+=system.shearOffset
                        elif l[2]>Lz/2.0:
                          l[2]-=Lz
                          l[0]-=system.shearOffset
                        if l[1]<-Ly/2.0:
                          l[1]+=Ly
                        elif l[1]>Ly/2.0:
                          l[1]-=Ly
                        if l[0]<-Lx/2.0:
						  while l[0]<-Lx/2.0: l[0]+=Lx
                        elif l[0]>Lx/2.0:
                          while l[0]>Lx/2.0: l[0]-=Lx
                        
                        dist=l.abs()
                        bin_index=int(dist/rdf_dr);
                        if bin_index<rdf_bin:
                            rdf_array[p][bin_index]+=1
                            
                        cnt+=1

    
out_energy.close()
# print timings and neighbor list information
end_time = time.process_time()

if bool_rdf:
    M_PI = 3.1415926535897932384626433832795029
    for p in range(len(rdf_pair)):
        #print(rdf_array[p])
        idx=rdf_pair[p][0]
        jdx=rdf_pair[p][1]
        itot_f=float(type_end[idx+1]-type_end[idx])
        jtot_f=float(type_end[jdx+1]-type_end[jdx])
        #rho_i=itot_f/Lx/Ly/Lz
        rho_j=jtot_f/Lx/Ly/Lz
        
        if jdx==idx:
            rad_tmp=rdf_bin*rdf_dr
            vol_tmp=4.0/3.0*M_PI*pow(rad_tmp,3)
            rho_j*=1.0-1.0/vol_tmp
            
        for i in range(rdf_bin):
            rdf_rad=(i+0.5)*rdf_dr
            rdf_fac=4.0*M_PI*rdf_rad*rdf_rad*rdf_dr*itot_f*rho_j
            print("RDF-%d-%d: %.3f %.6f" %( rdf_pair[p][0], rdf_pair[p][1], rdf_rad, rdf_array[p][i]/float(rdf_nlast)/rdf_fac))
      
      
timers.show(integrator.getTimers(), precision=4)
sys.stdout.write('Total # of neighbors = %d\n' % vl.totalSize())
sys.stdout.write('Ave neighs/atom = %.1f\n' % (vl.totalSize() / float(num_particles)))
sys.stdout.write('Neighbor list builds = %d\n' % vl.builds)
sys.stdout.write('Integration steps = %d\n' % integrator2.step)
sys.stdout.write('CPU time = %.1f\n' % (end_time - start_time))

