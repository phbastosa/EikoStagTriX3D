import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

parameters = str(sys.argv[1])

sps_path = pyf.catch_parameter(parameters,"SPS")
rps_path = pyf.catch_parameter(parameters,"RPS")
xps_path = pyf.catch_parameter(parameters,"XPS")

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dx = float(pyf.catch_parameter(parameters, "x_spacing"))
dy = float(pyf.catch_parameter(parameters, "x_spacing"))
dz = float(pyf.catch_parameter(parameters, "z_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

tId = int(pyf.catch_parameter(parameters, "beg_snap"))

snapshot_folder = pyf.catch_parameter(parameters, "snapshot_folder")
seismogram_folder = pyf.catch_parameter(parameters, "seismogram_folder") 

dh = np.array([dx, dy, dz])
slices = np.array([0.5*nz, 0.5*ny, 0.5*nx], dtype = int)

snap_file = snapshot_folder +f"triclinic_rsg_snapshot_step{tId}_{nz}x{nx}x{ny}_shot_1.bin" 

snapshot = pyf.read_binary_volume(nz, nx, ny, snap_file)

perc = 2000

snapshot *= perc / np.max(np.abs(snapshot))

pyf.plot_model_3D(snapshot, dh, slices, shots = sps_path, scale = 0.4, 
                  adjx = 0.5, dbar = 1.25, cmap = "Greys", cblab = "Amplitude", 
                  vmin = -0.1*perc, vmax = 0.1*perc)
plt.savefig("rsg.png", dpi = 300)
plt.show()