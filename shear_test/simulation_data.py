#!/usr/bin/env python3

#waiting and sleep (real) time in second for submitting subsequent simulation with a python script
wait_time = 5 #waiting time for checking whether a submitted simulation is started or not
sleep_time = 5*60 + 30 #sleeping time for finishing a started simulation


# GROMACS setup files for starting the simulation
grofile = "start.gro"
topfile = "topology.top"
temperature = 400 #[K]

# directory and file path for recording the results
result_directory = "out/"
result_file = result_directory + "result.h5"
convert_directory = result_directory + "convert/"
xtc_file = result_directory + "result.xtc"
unwrapped_xtc_file = result_directory + "unwrapped_centered.xtc"

# add shear rate / NSteps
shear_rate = 0.001
equi_nloops = 200  # total steps = nloops * isteps
equi_isteps = 50



# number of production loops
prod_nloops       = 200 #10 [ns]
# number of integration steps performed in each production loop
prod_isteps       = 1000 #every 5 [ps]



USE_DPD = False #true: use DPD thermostat; false: use Langevin thermostat

SKIP_EQUI = True
USE_CONSTRAIN = False#True # SHAKE, but unsure whether it works

# simulation parameters (nvt = False is nve)
rc    = 1.2  # Verlet list cutoff
skin  = 0.2 #0.3
tot_types = 12
timestep = 0.0005 # 0.001 ps will crash, d't know what's going on




#Ewald interaction parameters
USE_EWALD = False #true: use Ewald; false: use Reac field
coulomb_prefactor = 138.935485
alphaEwald = 2.602845 #2.885757
kspacecutoff = 8 #15
epsilon1 = 1.0 #relative dieelectric constant: equivalent to epsilon_r in Gromacs
epsilon2 = 13.0 #dielectric constant of the reaction field: equivalent to epsilon_rf in Gromacs
kappa = 0 #parameter for the generalized reaction field and when set to zero it describes the normal reaction field method


#Langevin thermostat parameters
Langevin_gamma = 0.5
Langevin_temperature = 8.314e-3 * temperature #kT in Gromacs units


#DPD thermostat parameters
DPD_gamma = 1.0 #should be higher than Langevin
DPD_temperature = 8.314e-3 * temperature #kT in Gromacs units
