#!/usr/bin/env python3

import sys
import time
import random
import math
import mpi4py.MPI as MPI
import espressopp
import logging
from espressopp import Real3D, Int3D
from espressopp.tools import lammps, gromacs
from espressopp.tools import decomp, timers, replicate
import os

start_time = time.time()

k_theta       = 1.5

try:

    with open('lastcycle.txt') as f: pass
    inputfile     = open('lastcycle.txt' ,'r')
    col           = inputfile.readline().split()
    restart_cycle    =   int(col[0])
    restart_tau      = float(col[1])
    restart_type     =       col[2]
    rcutnext_initial = float(col[3])
    restartfile = '%s_%d.xyz' % (restart_type,restart_cycle)


except IOError as e:

    restart_type     = 'warmup'
    restart_cycle    = 0
    rcutnext_initial = 1.1
    restart_tau      = 0
    restartfile   = 'prepack_100.xyz'

restart_types   = ['warmup','relax','equil']

if not restart_type in restart_types:
    print("invalid restart_type - restart_type must be in ", restart_types)
    sys.exit()

# ==================================================================================================
# Setting up initial system parameters
# ==================================================================================================

density            = 0.849
bondlength         = 0.97
monomers_per_chain = 50
num_chains         = 15*monomers_per_chain
nc_print = max(100,3*monomers_per_chain) #num_chains

sigma              = 1.0
epsilon            = 1.0
temperature        = 1.0

skin               = 0.3 #1.0
rc                 = pow(2, 1.0/6.0) * sigma

LJ_capradius_initial  = pow(2, 1.0/6.0) * sigma
LJ_capradius_final    = 0.80 * sigma

LJ_capnext_high     = pow(2, 1.0/6.0) * sigma
LJ_capnext_final    = 0.80 * sigma
LJ_capnext_initial  = rcutnext_initial * sigma

VMD     = False
sock    = None
restart = False
time_max = 60*60*24

# ==================================================================================================
# Defining simulation parameters
# ==================================================================================================

warmup_cycles        = 100
warmup_ramp          = 85
warmup_change_stage  = 70
warmup_steps_stage_1 = 50000
warmup_steps_stage_2 = 100000
warmup_timestep      = 0.0001
warmup_gamma         = 1.0
msid_bondrange       = 40

relax_cycles         = 1
relax_steps          = 100000
relax_timestep       = 0.001
relax_gamma          = 0.5

equil_cycles         = 10
equil_steps          = 200000
equil_timestep       = 0.005
equil_gamma          = 0.5

warmup_start_time  = 0
warmup_end_time    = 0
relax_start_time   = 0
relax_end_time     = 0
equil_start_time   = 0
equil_end_time     = 0

# ==================================================================================================
# Setting up the simulation box
# ==================================================================================================
sys.stdout.write('Setting up simulation ...\n')

num_particles        = num_chains * monomers_per_chain
Lz=Ly=3.2*math.sqrt(monomers_per_chain)
Lx=num_particles/density/Lz/Ly
size                = (Lx, Ly, Lz)

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

system.skin        = skin
comm = MPI.COMM_WORLD
if comm.size>4 and comm.size%4 ==0:
  nodeGrid = espressopp.Int3D(comm.size/4,2,2)
else:
  nodeGrid = espressopp.tools.decomp.nodeGrid(comm.size,size,rc,skin)
cellGrid = espressopp.tools.decomp.cellGrid(size,nodeGrid,rc,skin)
system.storage = espressopp.storage.DomainDecomposition(system, nodeGrid, cellGrid)

# ==================================================================================================
# Read configuration from file
# ==================================================================================================
#print("RST_FILE: ",restartfile,restart_type)
#sys.exit(0)
file = open(restartfile, 'r')
number_of_particles_in_file  = int(file.readline())
box_volume                   = file.readline().split()

if num_chains * monomers_per_chain != number_of_particles_in_file:
    print("ERROR: wrong number of particles in restart file")
    file.close()
    sys.exit()

FENE_pair_bonds            = espressopp.FixedPairList(system.storage)
Cosine_angle_bonds         = espressopp.FixedTripleList(system.storage)

next_nearest_neighbors = []
nearest_neighbors      = []
mass     = 1.0
props    = ['id', 'type', 'mass', 'pos', 'v']
vel_zero = espressopp.Real3D(0.0, 0.0, 0.0)
particle_id  = 1
p_incr = 0


for i in range(num_chains):
  polymer_chain       = []
  bonds = []
  angles = []
  
  for k in range(monomers_per_chain):
    col     = file.readline().split()
    part_id   = int(col[0])
    if part_id == 0:
      p_incr = 1

    if (len(col) == 8 or len(col) == 5):
      part_type = int(col[1])
      part_pos  = espressopp.Real3D(float(col[2]), float(col[3]), float(col[4]))
      part_vel  = espressopp.Real3D(float(col[5]), float(col[6]), float(col[7]))
	
    elif (len(col) == 7 or len(col) == 4):
      part_type = 0
      part_pos  = espressopp.Real3D(float(col[1]), float(col[2]), float(col[3]))
      part_vel  = espressopp.Real3D(float(col[4]), float(col[5]), float(col[6]))

    particle   = [part_id+p_incr, part_type, mass, part_pos, part_vel]
    polymer_chain.append(particle)
    #if k<5:
    #  print("ID %d - INFO: "%(part_id+p_incr),particle)
   
    #bonds = []
    #angles = []
    if k < monomers_per_chain-1:
      bonds.append((particle_id+k,particle_id+k+1))
	
    if k < monomers_per_chain-2:
      angles.append((particle_id+k, particle_id+k+1, particle_id+k+2))
    

  system.storage.addParticles(polymer_chain, *props)
  system.storage.decompose()
  
  particle_id += monomers_per_chain

file.close()
#system.storage.decompose()
#print("QUIT: ",warmup_start_time,restart_type)
#sys.exit(0)
# ==================================================================================================
# Define non bonded pair, bonded pair and bonded angular interaction lists
# ==================================================================================================

FENE_pair_bonds.addBonds(bonds)
Cosine_angle_bonds.addTriples(angles)

WCA_verlet_list            = espressopp.VerletList(system, cutoff = rc)
WCA_verlet_list_warmup     = espressopp.VerletList(system, cutoff = rc)

WCA_next_nearest_neighbors = espressopp.FixedPairList(system.storage)
WCA_nearest_neighbors      = espressopp.FixedPairList(system.storage)

for l in range(len(angles)):
    next_nearest_neighbors.append((angles[l][0], angles[l][2]))
for q in range(len(bonds)):
    nearest_neighbors.append((bonds[q][0],bonds[q][1]))

WCA_verlet_list_warmup.exclude(next_nearest_neighbors)
WCA_verlet_list_warmup.exclude(nearest_neighbors)

WCA_next_nearest_neighbors.addBonds(next_nearest_neighbors)
WCA_nearest_neighbors.addBonds(nearest_neighbors)

# ==================================================================================================
# Define interactions
# ==================================================================================================

# the capped WCA interaction uses the verlet list which excludes the nearest and next-nearest neighbors (for the warm-up)
WCA_capped_potential    = espressopp.interaction.LennardJonesCapped(sigma, epsilon, cutoff = rc, caprad = LJ_capradius_initial)
WCA_capped_interaction  = espressopp.interaction.VerletListLennardJonesCapped(WCA_verlet_list_warmup)
WCA_capped_interaction.setPotential(type1 = 0,type2 = 0,potential = WCA_capped_potential)

# the capped WCA interaction for next-nearest neighbors (for the warmup)
WCA_capped_next_potential = espressopp.interaction.LennardJonesCapped(epsilon, sigma, cutoff=rc, caprad = LJ_capnext_initial)
WCA_capped_next_interaction = espressopp.interaction.FixedPairListLennardJonesCapped(system, WCA_next_nearest_neighbors, potential = WCA_capped_next_potential)

# we also need to define the full excluded volume interaction for nearest neighbors (for the warmup)
WCA_full_nearest_interaction = espressopp.interaction.FixedPairListLennardJones(system,
                                                                         WCA_nearest_neighbors,
                                                                         potential = espressopp.interaction.LennardJones(epsilon, sigma, cutoff=rc))

# the WCA interaction full excluded volume (for relaxation and equilibration)
WCA_interaction         = espressopp.interaction.VerletListLennardJones(WCA_verlet_list)
WCA_potential           = WCA_interaction.setPotential(type1     = 0,
                                                       type2     = 0,
                                                       potential = espressopp.interaction.LennardJones(epsilon, sigma, cutoff = rc))

# define FENE bond interaction
FENE_interaction        = espressopp.interaction.FixedPairListFENE(system, FENE_pair_bonds, potential=espressopp.interaction.FENE(K=30.0, r0=0.0, rMax=1.5))

# define Cosine angular interaction
Cosine_interaction      = espressopp.interaction.FixedTripleListCosine(system, Cosine_angle_bonds, potential=espressopp.interaction.Cosine(K=k_theta, theta0=0.0))


#print("POSITION BEF. FOLD")
#idx=4501
#print(system.storage.getParticle(idx).pos)
##conf = espressopp.analysis.Configurations(system)
#conf = espressopp.analysis.ConfigurationsExt(system)
##conf.unfolded = False
#conf.capacity=1
#conf.gather()
#print(system.storage.getParticle(idx).pos,conf[0][idx])
#sys.exit(0)
# ==================================================================================================
# Setup Velocity Verlet integrator with Langevin thermostat
# ==================================================================================================

langevin = espressopp.integrator.LangevinThermostat(system)
langevin.gamma        = warmup_gamma
langevin.temperature  = temperature
integrator            = espressopp.integrator.VelocityVerlet(system)
integrator.addExtension(langevin)

# ==================================================================================================
# Print system and simulation information
# ==================================================================================================

output = open('info_%s.txt' % sys.argv[0][:-3] ,'a')
output.write(time.strftime('%a, %d %b %Y %H:%M:%S'))
output.write('\n\n%s\n\n'                            % espressopp.Version().info())
output.write('random seed           = %d \n'         % irand)
output.write('number of CPUs        = %d \n'         % espressopp.MPI.COMM_WORLD.size)
output.write('number of particles   = %d \n'         % num_particles)
output.write('number of chains      = %d \n'         % num_chains)
output.write('chain length          = %d \n'         % monomers_per_chain)
output.write('simulation box        = (%f,%f,%f) \n' % (size[0],size[1],size[2]))
output.write('density               = %f \n'         % density)
output.write('rc                    = %.2f \n'       % rc)
output.write('skin                  = %.2f \n'       % system.skin)
output.write('langevin temperature  = %.2f \n'       % langevin.temperature)
output.write('langevin warmup gamma = %.2f \n'       % warmup_gamma)
output.write('langevin relax gamma  = %.2f \n'       % relax_gamma)
output.write('langevin equil gamma  = %.2f \n'       % equil_gamma)
output.write('LJ capradius initial  = %f \n'         % LJ_capradius_initial)
output.write('LJ capradius final    = %f \n'         % LJ_capradius_final)
output.write('LJ capnext initial    = %f \n'         % LJ_capnext_initial)
output.write('LJ capnext final      = %f \n'         % LJ_capnext_final)
output.write('LJ capnext high       = %f \n'         % LJ_capnext_high)
output.write('bondlength            = %.2f \n'       % bondlength)
output.write('k_theta               = %f \n'         % k_theta)
output.write('NodeGrid              = (%.1f,%.1f,%.1f) \n' % (nodeGrid[0],nodeGrid[1],nodeGrid[2]))
output.write('CellGrid              = (%.1f,%.1f,%.1f) \n' % (cellGrid[0],cellGrid[1],cellGrid[2]))
output.write('warmup cycles         = %d \n'         % warmup_cycles)
output.write('warmup steps stage 1  = %d \n'         % warmup_steps_stage_1)
output.write('warmup steps stage 2  = %d \n'         % warmup_steps_stage_2)
output.write('warmup timestep       = %f \n'         % warmup_timestep)
output.write('relax cycles          = %d \n'         % relax_cycles)
output.write('relax steps           = %d \n'         % relax_steps)
output.write('relax timestep        = %f \n'         % relax_timestep)
output.write('equil cycles          = %d \n'         % equil_cycles)
output.write('equil steps           = %d \n'         % equil_steps)
output.write('equil timestep        = %f \n\n'       % equil_timestep)
output.close()

# ==================================================================================================
# Setup analysis
# ==================================================================================================

temperature    = espressopp.analysis.Temperature(system)
pressure       = espressopp.analysis.Pressure(system)
pressureTensor = espressopp.analysis.PressureTensor(system)
msid           = espressopp.analysis.MeanSquareInternalDist(system,monomers_per_chain)

def analyze_info(step,tau,step_type):

  T      = temperature.compute()
  P      = pressure.compute()
  Pij    = pressureTensor.compute()
  Ek     = (3.0/2.0) * T
  Etotal = 0.0
  tot    = '%5d %10d %8.6f %10.6f %10.6f %12.8f' % (step, tau, T, P, Pij[3], Ek)
  tt     = ''

  for k in range(system.getNumberOfInteractions()):
    e       = system.getInteraction(k).computeEnergy()/num_particles
    Etotal += e
    tot    += ' %12.8f' % e
    tt     += '      e%d     ' % k
  tot += ' %12.8f' % Etotal
  tt  += '    etotal   '

  tot2 = tot + '   %s \n' % ( time.strftime('%H:%M:%S'))  
  tot += ' %s %12.8f %12.8f\n' % (time.strftime('%H:%M:%S'), WCA_capped_potential.caprad, WCA_capped_next_potential.caprad)
  tt  += ' time  cap_radius  next_cap_radius\n'
  
  if step == restart_cycle or step == 0:
       output = open('info_%s.txt' % sys.argv[0][:-3] ,'a') 
       output.write(' cycle     tau         T          P        Pxy       ekinetic ' + tt)                
       output.close()

  if step_type == 'warmup':
       output = open('info_%s.txt' % sys.argv[0][:-3] ,'a') 
       output.write(tot)
       output.close()

  if step_type == 'relax' or step_type == 'equil' :
       output = open('info_%s.txt' % sys.argv[0][:-3] ,'a') 
       output.write(tot2)
       output.close()

# ==================================================================================================
# Warm-up
# ==================================================================================================

if not restart and restart_type == 'warmup':

    if (k_theta == 0.0): 
        msid_initial = open('average_intdist_0.txt','r')
    elif (k_theta == 0.75): 
        msid_initial = open('average_intdist_0p75.txt','r')
    elif (k_theta == 1.0): 
        msid_initial = open('average_intdist_1.txt','r')
    elif (k_theta == 1.5): 
        msid_initial = open('average_intdist_1p5.txt','r')
    elif (k_theta == 2.0): 
        msid_initial = open('average_intdist_2.txt','r')

    msid_zero = []
    for k in range(msid_bondrange):
            col = msid_initial.readline().split()
            msid_zero.append(float(col[1]))
    msid_initial.close()

    output = open('info_%s.txt' % sys.argv[0][:-3] ,'a') 
    output.write('starting warmup ... \n')
    output.write('\n')
    output.close()

    system.addInteraction(WCA_capped_interaction)
    system.addInteraction(WCA_capped_next_interaction)
    system.addInteraction(WCA_full_nearest_interaction)
    system.addInteraction(FENE_interaction)
    system.addInteraction(Cosine_interaction)

    if (VMD == True and sock == None):
        sock = espressopp.tools.vmd.connect(system, molsize = monomers_per_chain)

    if restart_type != 'warmup':
        i = 0
        tau = 0
    else:
        i = restart_cycle
        tau = restart_tau

    integrator.dt     = warmup_timestep
    warmup_steps      = warmup_steps_stage_1
    warmup_start_time = time.process_time()
    
    #espressopp.tools.analyse.info(system, integrator)
    #integrator.run(10)
    #espressopp.tools.analyse.info(system, integrator)
    #sys.exit(0)

    while not restart and i < warmup_cycles:

        cycle_begin_time = time.time()

        espressopp.tools.fastwritexyz('warmup_%d.xyz' % i, system)
        
        msid.gather()
        result = msid.compute()

        file = open("intdist_warmup_%d.txt" % i,"w")
        for c in range(monomers_per_chain-1):
            line = "%d %f\n" % (c+1,result[c]/(c+1))
            file.write(line)
        file.close()

        output = open('lastcycle.txt' ,'w')
        output.write('%d %f %s %12.8f' % (i, tau,'warmup', WCA_capped_next_potential.caprad) )
        output.close()

        WCA_capped_potential.caprad = LJ_capradius_initial + (LJ_capradius_final-LJ_capradius_initial)*i/(warmup_cycles-1)
        WCA_capped_interaction.setPotential(type1 = 0,type2 = 0,potential = WCA_capped_potential)

        if (i > warmup_ramp):

            LJ_capnext = WCA_capped_next_potential.caprad - 0.01

        else:

            msid_cycle = []
            msid_dif = 0.0
            msid_read = open('intdist_warmup_%d.txt' % i ,'r')
            for k in range(msid_bondrange):
                col = msid_read.readline().split()
                msid_cycle.append(float(col[1]))
                if (k>10):
                    msid_dif += msid_zero[k] - msid_cycle[k]
            msid_read.close()
		
            if   (msid_dif>0.0001):
                LJ_capnext = WCA_capped_next_potential.caprad - 0.01
            elif (msid_dif<-0.0001):
                LJ_capnext = WCA_capped_next_potential.caprad + 0.01
            else:
                LJ_capnext = WCA_capped_next_potential.caprad

        if (LJ_capnext < LJ_capnext_final):  LJ_capnext = LJ_capnext_final
        if (LJ_capnext > LJ_capnext_high):   LJ_capnext = LJ_capnext_high

        WCA_capped_next_potential.caprad = LJ_capnext
        WCA_capped_next_interaction.setPotential(potential = WCA_capped_next_potential)

        analyze_info(i,tau, 'warmup')

        if (i > warmup_change_stage): warmup_steps = warmup_steps_stage_2

        integrator.run(warmup_steps)
        tau += warmup_steps * warmup_timestep

        if VMD == True:
            espressopp.tools.vmd.imd_positions(system,sock,folded=False)

        i += 1

        current_time = time.time()
        cycle_time = current_time - cycle_begin_time
        if (i == warmup_change_stage): cycle_time = (warmup_steps_stage_2/warmup_steps_stage_1)*cycle_time
        remaining_time = time_max - (current_time - start_time)
        if ((remaining_time/cycle_time) < 1): restart = True
        else: restart = False
        print("warmup i: ", i , "restart: ", restart)

    analyze_info(i,tau, 'warmup')
    espressopp.tools.fastwritexyz('warmup_%d.xyz' % i, system)

    msid.gather()
    result = msid.compute()

    file = open("intdist_warmup_%d.txt" % i,"w")
    for c in range(monomers_per_chain-1):
       line = "%d %f\n" % (c+1,result[c]/(c+1))
       file.write(line)
    file.close()

    if (i == warmup_cycles):

        os.system('cp warmup_%d.xyz relax_0.xyz' % i)
        output = open('lastcycle.txt' ,'w')
        output.write('%d %d %s %f' % (0, 0,'relax',0.0) )
        output.close()
        restart_type = 'relax'
        restart_cycle = 0
        restart_tau = 0 

    else:

        output = open('lastcycle.txt' ,'w')
        output.write('%d %f %s %12.8f' % (i, tau,'warmup', WCA_capped_next_potential.caprad) )
        output.close()

    system.removeInteraction(4)
    system.removeInteraction(3)
    system.removeInteraction(2)
    system.removeInteraction(1)
    system.removeInteraction(0)

    WCA_verlet_list_warmup.disconnect()
    warmup_end_time = time.process_time()
    espressopp.tools.analyse.final_info(system, integrator, WCA_verlet_list_warmup, warmup_start_time, warmup_end_time)

# ==================================================================================================
# Relaxation
# ==================================================================================================

if not restart and restart_type == 'relax':

    output = open('info_%s.txt' % sys.argv[0][:-3] ,'a') 
    output.write('starting relaxation ...\n')
    output.write('\n')
    output.close()

    langevin.gamma = relax_gamma

    system.addInteraction(WCA_interaction)
    system.addInteraction(FENE_interaction)
    system.addInteraction(Cosine_interaction)

    if (VMD == True and sock == None):
        sock = espressopp.tools.vmd.connect(system, molsize = monomers_per_chain)

    if restart_type != 'relax':
        i   = 0
        tau = 0
    else:
        i = restart_cycle
        tau = restart_tau

    integrator.dt = relax_timestep
    relax_start_time = time.process_time()

    while not restart and i < relax_cycles:

        cycle_begin_time = time.time()

        analyze_info(i,tau, 'relax')
        espressopp.tools.fastwritexyz('relax_%d.xyz' % i, system)

        output = open('lastcycle.txt' ,'w')
        output.write('%d %d %s %f' % (i, tau,'relax',0.0) )
        output.close()

        integrator.run(relax_steps)
        tau += relax_steps * relax_timestep

        if VMD == True:
            espressopp.tools.vmd.imd_positions(system,sock,folded=False)

        i += 1

        current_time = time.time()
        cycle_time = current_time - cycle_begin_time
        remaining_time = time_max - (current_time - start_time)
        if ((remaining_time/cycle_time) < 1): restart = True
        else: restart = False
        print("relax i: ", i , "restart: ", restart)
        
    analyze_info(i,tau, 'relax')
    espressopp.tools.fastwritexyz('relax_%d.xyz' % i, system)

    if (i == relax_cycles):

        os.system('cp relax_%d.xyz equil_0.xyz' % i)
        output = open('lastcycle.txt' ,'w')
        output.write('%d %d %s %f' % (0, 0,'equil',0.0) )
        output.close()
        restart_type = 'equil'
        restart_cycle = 0
        restart_tau = 0        

    else:

        output = open('lastcycle.txt' ,'w')
        output.write('%d %f %s %12.8f' % (i, tau,'relax', 0.0) )
        output.close()

    system.removeInteraction(2)
    system.removeInteraction(1)
    system.removeInteraction(0)

    relax_end_time   = time.process_time()
    espressopp.tools.analyse.final_info(system, integrator, WCA_verlet_list , relax_start_time, relax_end_time)

# ==================================================================================================
# Equilibration
# ==================================================================================================

if not restart and restart_type == 'equil':

    output = open('info_%s.txt' % sys.argv[0][:-3] ,'a') 
    output.write('starting equilibration ...\n')
    output.write('\n')
    output.close()

    langevin.gamma = equil_gamma

    system.addInteraction(WCA_interaction)
    system.addInteraction(FENE_interaction)
    system.addInteraction(Cosine_interaction)

    if (VMD == True and sock == None):
        sock = espressopp.tools.vmd.connect(system, molsize = monomers_per_chain)

    if restart_type != 'equil':
        i = 0
        tau = 0 
    else:
        i = restart_cycle
        tau = restart_tau

    integrator.dt = equil_timestep
    equil_start_time = time.process_time()

    while not restart and i < equil_cycles:

        cycle_begin_time = time.time()

        analyze_info(i,tau, 'equil')
        espressopp.tools.fastwritexyz('equil_%d.xyz' % i, system)

        msid.gather()
        result = msid.compute()

        file = open("intdist_equil_%d.txt" % i,"w")
        for c in range(monomers_per_chain-1):
            line = "%d %f\n" % (c+1,result[c]/(c+1))
            file.write(line)
        file.close()

        output = open('lastcycle.txt' ,'w')
        output.write('%d %d %s %f' % (i, tau,'equil',0.0) )
        output.close()

        integrator.run(equil_steps)
        tau += equil_steps * equil_timestep

        if VMD == True:
            espressopp.tools.vmd.imd_positions(system,sock,folded=False)

        i += 1

        current_time = time.time()
        cycle_time = current_time - cycle_begin_time
        remaining_time = time_max - (current_time - start_time)
        if ((remaining_time/cycle_time) < 1): restart = True
        else: restart = False
        print("equil i: ", i , "restart: ", restart)

    analyze_info(i,tau, 'equil')
    espressopp.tools.fastwritexyz('equil_%d.xyz' % i, system)

    msid.gather()
    result = msid.compute()

    file = open("intdist_equil_%d.txt" % i,"w")
    for c in range(monomers_per_chain-1):
       line = "%d %f\n" % (c+1,result[c]/(c+1))
       file.write(line)
    file.close()

    output = open('lastcycle.txt' ,'w')
    output.write('%d %d %s %f' % (i, tau,'equil',0.0) )
    output.close()

    system.removeInteraction(2)
    system.removeInteraction(1)
    system.removeInteraction(0)

    equil_end_time = time.process_time()
    espressopp.tools.analyse.final_info(system, integrator, WCA_verlet_list , equil_start_time, equil_end_time)

# ==================================================================================================
# Termination
# ==================================================================================================

output = open('last.txt' ,'w')
output.write('%d %f %s' % (i, tau, restart_type) )
output.close()

