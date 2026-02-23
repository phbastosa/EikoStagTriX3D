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

slowness_file = pyf.catch_parameter(parameters, "slowness_file")
velocity = 1.0 / pyf.read_binary_volume(nz, nx, ny, slowness_file)

dh = np.array([dz, dy, dx])
slices = np.array([0.5*nz, 0.5*ny, 0.5*nx], dtype = int)

pyf.plot_model_3D(velocity, dh, slices, shots = sps_path, nodes = rps_path, 
                  scale = 0.4, adjx = 0.5, dbar = 1.25, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.savefig("setup.png", dpi = 200)
plt.show()

isnap = int(pyf.catch_parameter(parameters, "snapshot_beg"))

snap_rsg_file = f"../outputs/snapshots/triclinic_rsg_snapshot_step{isnap}_{nz}x{nx}x{ny}_shot_1.bin"

snap_rsg = pyf.read_binary_volume(nz, nx, ny, snap_rsg_file)

snap_rsg *= 1000 / np.max(np.abs(snap_rsg))

eikonal_path = f"../outputs/snapshots/triclinic_eikonal_{nz}x{nx}x{ny}_shot_1.bin"

times = pyf.read_binary_volume(nz, nx, ny, eikonal_path)

pyf.plot_model_3D(snap_rsg, dh, slices, eikonal = times, eikonal_levels = [0.3], 
                  eikonal_color = "red",  scale = 0.4, adjx = 0.52, dbar = 1.25, 
                  cmap = "Greys", cblab = "Normalized Amplitude", vmin = -400, vmax = 400)
plt.savefig("rsg.png", dpi = 200)
plt.show()
