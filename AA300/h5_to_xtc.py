import h5py
import numpy as np
import os
import simulation_data

#os.system("module load gromacs/2018.8")

def last_frame_finder(file, time_path='energy/time'):

    with h5py.File(file, 'r') as h5:
        last_time = np.max(np.array(h5[time_path]))
        h5.close()

    last_frame = int((last_time/simulation_data.timestep)/simulation_data.prod_isteps)

    return last_frame



def g96_writer(filename, position, index, box, time, step):

    name_list_AMIM = ["NA1", "NA2", "HC1", "HC2", "HC3", "HC4", "HC5", "HC6", "HC7", "HA1", "HA2", "HA3", "H11", "H12", "H13", "H14", "H15", "CW1", "CW2", "CT1", "CS1", "CR1", "C21", "C11", "C12"]
    name_list_PF6 = ["P1", "F1", "F2", "F3", "F4", "F5", "F6"]

    out_stream = open(filename, 'w')
    out_stream.write(f"TITLE\n")
    out_stream.write(f"AA AMIM-PF6 t={time:10.5f} step= {step} \n")
    out_stream.write(f"END\n")
    out_stream.write(f"POSITION\n")

    n_particles = len(position[:,0])

    for i in range(n_particles):

        if index[i,1]==1:
            molecule_name = "AMIM"
            atom_name = name_list_AMIM[int(index[i,2])-1]
        elif index[i,1]==-1:
            molecule_name = "PF6"
            atom_name = name_list_PF6[int(index[i,2])-1]

        out_stream.write(f"{index[i,0]:>5} {molecule_name:<5} {atom_name:<5} "
                         f"{i+1:>6}{position[i,0]:15.9f}{position[i,1]:15.9f}{position[i,2]:15.9f}\n")

    out_stream.write(f"END\n")
    out_stream.write(f"BOX\n")

    shearbox = 0
    out_stream.write(f"{box[0]:15.9f}{box[1]:15.9f}{box[2]:15.9f}"
                     f"{0:15.9f}{0:15.9f}{0:15.9f}"
                     f"{0:15.9f}{shearbox:15.9f}{0:15.9f}\n")

    out_stream.write(f"END\n")





with h5py.File(simulation_data.result_file, "r+") as h5md:
    position = h5md["particles/position"][:,:,:] #[frame number, particle ID, [x, y, z]]
    velocity = h5md["particles/velocity"][:,:,:] #[frame number, particle ID, [vx, vy, vz]]
    time = h5md["particles/time"][:] #[time]
    step = h5md["particles/step"][:] #[step number]
    index = h5md["particles/index"][:,:,:] #[frame number, particle ID, [molecule number, molecule name, atom ID in molecule]]
    temperature = h5md["observables/temperature"][:] #[temperature]
    energy = h5md["energy/value"][:,:] #[frame number, [E_total, E_kinetic, E_potential, E_coulomb, E_bond, E_angle, E_dihedral]]
    box = h5md["box/size"][:] #[Lx, Ly, Lz]

    h5md.close()

frame_number = last_frame_finder(simulation_data.result_file) #len(position[:,0,0])

new_directory = simulation_data.convert_directory
os.system("mkdir -p "+new_directory)

for i in range(frame_number):
    print(i)
    #create the g96 file for each frame from the h5 file
    g96_writer(new_directory+str(i)+".g96", position[i,:,:], index[i,:,:], box, time[i], step[i])

    #convert the g96 file to xtc file
    os.system("gmx trjconv -f " + new_directory + str(i) + ".g96 -o " + new_directory + str(i) + ".xtc -ndec 4 -t0 " + str(time[i]) + " -quiet yes")

#concatenate all xtc files into one big one
os.system("gmx trjcat -f " + new_directory + "*.xtc -o " + simulation_data.xtc_file + " -quiet yes -nobackup")

#delete everything in between after conversion
os.system("rm -r " + new_directory)

