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

snapshot_folder = pyf.catch_parameter(parameters, "snapshot_folder")
seismogram_folder = pyf.catch_parameter(parameters, "seismogram_folder") 

dh = np.array([dx, dy, dz])
slices = np.array([0.5*nz, 0.5*ny, 0.5*nx], dtype = int)

tId = 400

e = ["T","F","F","T","T","F","T","T","T"]
d = ["F","T","F","T","F","T","T","T","T"]
g = ["F","F","T","F","T","T","T","T","T"]
tx = ["F","F","F","F","F","F","F","T","F"]
ty = ["F","F","F","F","F","F","F","F","T"]

perc = 2000

for i in range(len(e)):

    eiko_file = snapshot_folder +f"anisotropy_eikonal_e{e[i]}_d{d[i]}_g{g[i]}_tx{tx[i]}_ty{ty[i]}.bin"
    snap_file = snapshot_folder +f"anisotropy_snapshot_e{e[i]}_d{d[i]}_g{g[i]}_tx{tx[i]}_ty{ty[i]}.bin" 

    eikonal = pyf.read_binary_volume(nz, nx, ny, eiko_file)
    snapshot = pyf.read_binary_volume(nz, nx, ny, snap_file)

    snapshot *= perc / np.max(np.abs(snapshot))

    ev = 0.1 if e[i] == "T" else 0
    dv = 0.1 if d[i] == "T" else 0
    gv = 0.1 if g[i] == "T" else 0
    txv = 30 if tx[i] == "T" else 0
    tyv = 30 if ty[i] == "T" else 0

    pyf.plot_model_3D(snapshot, dh, slices, shots = sps_path, scale = 0.4, 
                      nodes = rps_path, eikonal = eikonal, eikonal_levels = [tId*dt], 
                      eikonal_colors = ["red"] , adjx = 0.5, dbar = 1.25, cmap = "Greys",
                      cblab = fr"$\epsilon$ = {ev}, $\delta$ = {dv}, $\gamma$ = {gv}, $\theta_x$ = {txv}°, $\theta_y$ = {tyv}°", 
                      vmin = -0.5*perc, vmax = 0.5*perc)
    plt.show()

