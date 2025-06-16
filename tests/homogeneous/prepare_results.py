import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

parameters = str(sys.argv[1])

sps_path = pyf.catch_parameter(parameters,"SPS")
rps_path = pyf.catch_parameter(parameters,"RPS")
xps_path = pyf.catch_parameter(parameters,"XPS")

SPS = np.loadtxt(sps_path, delimiter = ",", dtype = float)
RPS = np.loadtxt(rps_path, delimiter = ",", dtype = float)
XPS = np.loadtxt(xps_path, delimiter = ",", dtype = int)

nr = len(RPS)

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dx = float(pyf.catch_parameter(parameters, "x_spacing"))
dy = float(pyf.catch_parameter(parameters, "x_spacing"))
dz = float(pyf.catch_parameter(parameters, "z_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

model_file = pyf.catch_parameter(parameters, "vp_model_file")
snapshot_folder = pyf.catch_parameter(parameters, "snapshot_folder")
seismogram_folder = pyf.catch_parameter(parameters, "seismogram_folder") 

model = pyf.read_binary_volume(nz, nx, ny, model_file)

dh = np.array([dx, dy, dz])
slices = np.array([0.5*nz, 0.5*ny, 0.5*nx], dtype = int)

pyf.plot_model_3D(model, dh, slices, shots = sps_path, scale = 2.0, 
                  nodes = rps_path, adjx = 0.75, dbar = 1.4,
                  cblab = "P wave velocity [km/s]", 
                  vmin = 1600, vmax = 2000)
plt.savefig("homogeneous_test_vp.png", dpi = 300)
plt.show()

tId = 500

snap_iso_file = snapshot_folder +f"elastic_ani_snapshot_step{tId}_{nz}x{nx}x{ny}_shot_1.bin" 
eiko_iso_file = snapshot_folder +f"elastic_ani_eikonal_{nz}x{nx}x{ny}_shot_1.bin"

snap_ani_file = snapshot_folder +f"elastic_ani_snapshot_step{tId}_{nz}x{nx}x{ny}_shot_1.bin" 
eiko_ani_file = snapshot_folder +f"elastic_ani_eikonal_{nz}x{nx}x{ny}_shot_1.bin"

eikonal_iso = pyf.read_binary_volume(nz, nx, ny, eiko_iso_file)
snapshot_iso = pyf.read_binary_volume(nz, nx, ny, snap_iso_file)

perc = 2000

snapshot_iso *= perc / np.max(np.abs(snapshot_iso))

pyf.plot_model_3D(snapshot_iso, dh, slices, shots = sps_path, scale = 2.0, 
                  nodes = rps_path, eikonal = eikonal_iso, eikonal_levels = [tId*dt], 
                  eikonal_colors = ["red"] , adjx = 0.75, dbar = 1.4, cmap = "Greys",
                  cblab = "Normalized amplitude - Isotropic", vmin = -0.5*perc, vmax = 0.5*perc)
plt.savefig("homogeneous_test_snap_iso.png", dpi = 300)
plt.show()

eikonal_ani = pyf.read_binary_volume(nz, nx, ny, eiko_ani_file)
snapshot_ani = pyf.read_binary_volume(nz, nx, ny, snap_ani_file)

snapshot_ani *= perc / np.max(np.abs(snapshot_ani))

pyf.plot_model_3D(snapshot_ani, dh, slices, shots = sps_path, scale = 2.0, 
                  nodes = rps_path, eikonal = eikonal_ani, eikonal_levels = [tId*dt], 
                  eikonal_colors = ["red"] , adjx = 0.75, dbar = 1.4, cmap = "Greys",
                  cblab = "Normalized amplitude - Anisotropic", vmin = -0.5*perc, vmax = 0.5*perc)
plt.savefig("homogeneous_test_snap_ani.png", dpi = 300)
plt.show()

seis_iso_path = seismogram_folder +f"elastic_iso_nStations{nr}_nSamples{nt}_shot_1.bin"
seis_ani_path = seismogram_folder +f"elastic_ani_nStations{nr}_nSamples{nt}_shot_1.bin"

seismogram_iso = pyf.read_binary_matrix(nt, nr, seis_iso_path)
seismogram_ani = pyf.read_binary_matrix(nt, nr, seis_ani_path)

scale = np.std(seismogram_iso)

fig, ax = plt.subplots(nrows = 2, figsize = (15,8))

ax[0].imshow(seismogram_iso, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax[1].imshow(seismogram_ani, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

fig.tight_layout()
plt.show()

times = np.arange(nt)*dt

trace_iso = seismogram_iso[:,int(0.5*nr)]
trace_ani = seismogram_ani[:,int(0.5*nr)]

freqs = np.fft.fftfreq(nt, dt)

mask = freqs >= 0

fft_iso = np.fft.fft(trace_iso)
fft_ani = np.fft.fft(trace_ani)

fig, ax = plt.subplots(ncols = 2, figsize = (6,8))

ax[0].plot(trace_iso, times)
ax[0].plot(trace_ani, times)
ax[0].set_ylim([0, (nt-1)*dt])
ax[0].invert_yaxis()

ax[1].plot(np.abs(fft_iso[mask]), freqs[mask])
ax[1].plot(np.abs(fft_ani[mask]), freqs[mask])
ax[1].set_ylim([0, 30])
ax[1].invert_yaxis()

fig.tight_layout()
plt.show()

