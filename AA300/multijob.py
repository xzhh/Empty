#!/usr/bin/env python3

#############################################################################################
##       THIS SRCIPT IS DESIGNED TO SUBMIT SEVERAL SUBSEQUENT JOBS TO THE HPC CLUSTER      ##
#############################################################################################

import os
import time
import h5py
import numpy as np
import subprocess
import simulation_data

    
##################################################################################
##       LAST TIME FRAME READER AT THE END OF EACH STAGE OF THE SIMULATION      ##
##################################################################################    
def last_time_finder(file, time_path='energy/time'):# "time_path" is where to find time in the h5 file
    with h5py.File(file, 'r') as h5:
        last_time_frame = np.max(np.array(h5[time_path]))
        h5.close()
                 
    return last_time_frame

###############################################################
##              status finder of a specific job ID           ##
###############################################################
jobID_finder = lambda output: int(output[20:].rstrip())

def status_finder(jobID):
    output = subprocess.check_output("squeue -j " + str(jobID), shell=True)
    status = str(output).split()
    return status[-4]



new_simulation = True # True for brand new simulation and False for the continuation of a previous simulation
simulation_time = simulation_data.timestep*simulation_data.prod_nloops*simulation_data.prod_isteps #total simulation time according to the simulation script in [ps]
wait_time = simulation_data.wait_time #[s] waiting time until a simulation started after submission
sleep_time = simulation_data.sleep_time #24*60*60 + 10*60 #[s] waiting sime until a simulation finished after starting
start_command = 'sbatch jobscript.sh 0'
continue_command = 'sbatch jobscript.sh 1'
result_directory = simulation_data.result_directory
result_file = simulation_data.result_file

if new_simulation:
    #submit the first/initial step of the simulation
    output_start = subprocess.check_output(start_command, shell=True)
    jobID_start = jobID_finder(output_start)
    
    
    #wait untile the submitted job started to run
    print("submitted job for starting the simulation is in the que...")
    while not (status_finder(jobID_start)=="R"):
        time.sleep(wait_time)
    
    #wait for 24h until the first simulation is done
    print("the job is running...")
    time.sleep(sleep_time)

#read the reached time after the first step is done
t = last_time_finder(result_file)
print(t,"[ps] of the simulation time(",simulation_time,"[ps]) has passed.")

#submitting subsequent jobs
while t<simulation_time:
    #submit the subsequent step of the simulation
    output_continue = subprocess.check_output(continue_command, shell=True)
    jobID_continue = jobID_finder(output_continue)
    
    #wait untile the submitted job started to run
    print("the subsequent submitted job is in the que...")
    while not (status_finder(jobID_continue)=="R"):
        time.sleep(wait_time)
    
    #wait for 24h until the above simulation is done
    print("the job is running...")
    time.sleep(sleep_time)
    
    #read the reached time after the first step is done
    t = last_time_finder(result_file)
    print(t,"[ps] of the simulation time(",simulation_time,"[ps]) has passed.")


#after finishing the simulation, the h5 file will be converted to a xtc file
os.system("module load gromacs/2018.8")
os.system("python3 h5_to_xtc.py")
os.system("gmx trjconv -f " + simulation_data.xtc_file + " -s " + simulation_data.grofile + " -pbc nojump -center -o " + simulation_data.unwrapped_xtc_file)
