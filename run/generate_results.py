import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

parameters = str(sys.argv[1])

sps_path = pyf.catch_parameter(parameters, "SPS") 
rps_path = pyf.catch_parameter(parameters, "RPS") 

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dx = float(pyf.catch_parameter(parameters, "x_spacing"))
dy = float(pyf.catch_parameter(parameters, "y_spacing"))
dz = float(pyf.catch_parameter(parameters, "z_spacing"))

dt = float(pyf.catch_parameter(parameters, "time_spacing"))

slowness_file = pyf.catch_parameter(parameters, "slowness_file")
velocity = 1.0 / pyf.read_binary_volume(nz, nx, ny, slowness_file)

dh = np.array([dz, dy, dx])
slices = np.array([25, 250, 50], dtype = int)

pyf.plot_model_3D(velocity, dh, slices, shots = sps_path, nodes = rps_path, 
                  scale = 4.2, adjx = 0.8, dbar = 1.65, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.savefig("setup.png", dpi = 200)
plt.show()

isnap = int(pyf.catch_parameter(parameters, "beg_snap"))
fsnap = int(pyf.catch_parameter(parameters, "end_snap"))

isnap_rsg_file = f"../outputs/snapshots/triclinic_rsg_snapshot_step{isnap}_{nz}x{nx}x{ny}_shot_1.bin"
fsnap_rsg_file = f"../outputs/snapshots/triclinic_rsg_snapshot_step{fsnap}_{nz}x{nx}x{ny}_shot_1.bin"

eikonal_path = f"../outputs/snapshots/triclinic_eikonal_{nz}x{nx}x{ny}_shot_1.bin"

isnap_rsg = pyf.read_binary_volume(nz, nx, ny, isnap_rsg_file)
fsnap_rsg = pyf.read_binary_volume(nz, nx, ny, fsnap_rsg_file)

isnap_rsg *= 1000 / np.max(np.abs(isnap_rsg))
fsnap_rsg *= 1000 / np.max(np.abs(fsnap_rsg))

times = pyf.read_binary_volume(nz, nx, ny, eikonal_path)

pyf.plot_model_3D(isnap_rsg, dh, slices, eikonal = times, eikonal_levels = [isnap*dt], 
                  eikonal_color = "red",  scale = 4.2, adjx = 0.8, dbar = 1.65, 
                  cmap = "Greys", cblab = "Normalized Amplitude", vmin = -400, vmax = 400)
plt.savefig("rsg_snap500ms.png", dpi = 200)
plt.show()

pyf.plot_model_3D(fsnap_rsg, dh, slices, eikonal = times, eikonal_levels = [fsnap*dt], 
                  eikonal_color = "red",  scale = 4.2, adjx = 0.8, dbar = 1.65, 
                  cmap = "Greys", cblab = "Normalized Amplitude", vmin = -400, vmax = 400)
plt.savefig("rsg_snap1000ms.png", dpi = 200)
plt.show()
