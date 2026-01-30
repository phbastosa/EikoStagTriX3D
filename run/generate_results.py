import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

parameters = str(sys.argv[1])

sps_path = pyf.catch_parameter(parameters,"SPS") 
rps_path = pyf.catch_parameter(parameters,"RPS") 

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dx = float(pyf.catch_parameter(parameters, "x_spacing"))
dy = float(pyf.catch_parameter(parameters, "y_spacing"))
dz = float(pyf.catch_parameter(parameters, "z_spacing"))

slowness_file = pyf.catch_parameter(parameters,"slowness_file")
velocity = 1.0 / pyf.read_binary_volume(nz,nx,ny,slowness_file)

SPS = np.loadtxt(sps_path, dtype = np.float32, delimiter = ",")
RPS = np.loadtxt(rps_path, dtype = np.float32, delimiter = ",")

dh = np.array([dz, dy, dx])
slices = np.array([0.5*nz, 0.5*ny, 0.5*nx], dtype = int)

pyf.plot_model_3D(velocity, dh, slices, shots = sps_path, nodes = rps_path, 
                  scale = 0.4, adjx = 0.5, dbar = 1.25, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.show()


snap_ssg_file = f"../outputs/snapshots/triclinic_ssg_snapshot_step400_{nz}x{nx}x{ny}_shot_1.bin"
snap_rsg_file = f"../outputs/snapshots/triclinic_rsg_snapshot_step400_{nz}x{nx}x{ny}_shot_1.bin"
snap_issg_file = f"../outputs/snapshots/triclinic_issg_snapshot_step400_{nz}x{nx}x{ny}_shot_1.bin"

snap_ssg = pyf.read_binary_volume(nz,nx,ny,snap_ssg_file)
snap_rsg = pyf.read_binary_volume(nz,nx,ny,snap_rsg_file)
snap_issg = pyf.read_binary_volume(nz,nx,ny,snap_issg_file)

snap_ssg[int(0.5*nz)-2:int(0.5*nz)+2,
         int(0.5*nx)-2:int(0.5*nx)+2,
         int(0.5*ny)-2:int(0.5*ny)+2] *= 0.0

snap_rsg[int(0.5*nz)-2:int(0.5*nz)+2,
         int(0.5*nx)-2:int(0.5*nx)+2,
         int(0.5*ny)-2:int(0.5*ny)+2] *= 0.0

snap_issg[int(0.5*nz)-2:int(0.5*nz)+2,
          int(0.5*nx)-2:int(0.5*nx)+2,
          int(0.5*ny)-2:int(0.5*ny)+2] *= 0.0

snap_ssg *= 1000 / np.max(np.abs(snap_ssg))
snap_rsg *= 1000 / np.max(np.abs(snap_rsg))
snap_issg *= 1000 / np.max(np.abs(snap_issg))

pyf.plot_model_3D(snap_ssg, dh, slices,  
                  scale = 0.4, adjx = 0.52, dbar = 1.25, cmap = "Greys",
                  cblab = "Normalized Amplitude", vmin = -400, vmax = 400)
plt.savefig("ssg.png", dpi = 200)
plt.show()

pyf.plot_model_3D(snap_rsg, dh, slices,  
                  scale = 0.4, adjx = 0.52, dbar = 1.25, cmap = "Greys",
                  cblab = "Normalized Amplitude", vmin = -400, vmax = 400)
plt.savefig("rsg.png", dpi = 200)
plt.show()

pyf.plot_model_3D(snap_issg, dh, slices, 
                  scale = 0.4, adjx = 0.52, dbar = 1.25, cmap = "Greys",
                  cblab = "Normalized Amplitude", vmin = -400, vmax = 400)
plt.savefig("issg.png", dpi = 200)
plt.show()
