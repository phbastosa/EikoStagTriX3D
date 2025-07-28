import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

from matplotlib.gridspec import GridSpec

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

tId = 400

snap_iso_file = snapshot_folder +f"elastic_iso_snapshot_step{tId}_{nz}x{nx}x{ny}_shot_1.bin" 
eiko_iso_file = snapshot_folder +f"elastic_iso_eikonal_{nz}x{nx}x{ny}_shot_1.bin"

snap_ani_file = snapshot_folder +f"elastic_ani_snapshot_step{tId}_{nz}x{nx}x{ny}_shot_1.bin" 
eiko_ani_file = snapshot_folder +f"elastic_ani_eikonal_{nz}x{nx}x{ny}_shot_1.bin"

eikonal_iso = pyf.read_binary_volume(nz, nx, ny, eiko_iso_file)
snapshot_iso = pyf.read_binary_volume(nz, nx, ny, snap_iso_file)

perc = 2000

snapshot_iso *= perc / np.max(np.abs(snapshot_iso))

pyf.plot_model_3D(snapshot_iso, dh, slices, shots = sps_path, scale = 0.4, 
                  nodes = rps_path, eikonal = eikonal_iso, eikonal_levels = [tId*dt], 
                  eikonal_colors = ["red"] , adjx = 0.5, dbar = 1.25, cmap = "Greys",
                  cblab = "Normalized amplitude - Isotropic", vmin = -0.5*perc, vmax = 0.5*perc)
plt.show()

eikonal_ani = pyf.read_binary_volume(nz, nx, ny, eiko_ani_file)
snapshot_ani = pyf.read_binary_volume(nz, nx, ny, snap_ani_file)

snapshot_ani *= perc / np.max(np.abs(snapshot_ani))

pyf.plot_model_3D(snapshot_ani, dh, slices, shots = sps_path, scale = 0.4, 
                  nodes = rps_path, eikonal = eikonal_ani, eikonal_levels = [tId*dt], 
                  eikonal_colors = ["red"] , adjx = 0.5, dbar = 1.25, cmap = "Greys",
                  cblab = "Normalized amplitude - Anisotropic", vmin = -0.5*perc, vmax = 0.5*perc)
plt.show()

seis_iso_path = seismogram_folder +f"elastic_iso_Ps_nStations{nr}_nSamples{nt}_shot_1.bin"
seis_ani_path = seismogram_folder +f"elastic_ani_Ps_nStations{nr}_nSamples{nt}_shot_1.bin"

seismogram_iso = pyf.read_binary_matrix(nt, nr, seis_iso_path)
seismogram_ani = pyf.read_binary_matrix(nt, nr, seis_ani_path)

xloc = np.linspace(0, nr-1, 7)
xlab = np.linspace(0, nr, 7, dtype = int)

tloc = np.linspace(0, nt-1, 7)
tlab = np.linspace(0, (nt-1)*dt, 7)

rectangle = np.array([[0.5*nr-3, 0.2], [0.5*nr-3, 1.0], 
                      [0.5*nr+3, 1.0], [0.5*nr+3, 0.2], 
                      [0.5*nr-3, 0.2]])

scale = 5*np.std(seismogram_iso)

times = np.arange(nt)*dt

trace_iso = seismogram_iso[:,int(0.5*nr)]
trace_ani = seismogram_ani[:,int(0.5*nr)]

freqs = np.fft.fftfreq(nt, dt)

mask = freqs >= 0

fft_iso = np.fft.fft(trace_iso)
fft_ani = np.fft.fft(trace_ani)

t = np.arange(nt)*dt

ts = slice(int(0.2/dt), int(1.0/dt))

fig = plt.figure(figsize = (15, 8))

gs = GridSpec(2, 5, figure = fig)

ax1 = fig.add_subplot(gs[:1,:3]) 
ax1.imshow(seismogram_iso, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
ax1.plot(rectangle[:,0], rectangle[:,1]/dt, "--k")
ax1.set_title("Elastic ISO", fontsize = 15)
ax1.set_xlabel("Trace Index", fontsize = 15)
ax1.set_ylabel("Time [s]", fontsize = 15)
ax1.set_xticks(xloc)
ax1.set_yticks(tloc)
ax1.set_xticklabels(xlab)
ax1.set_yticklabels(tlab)

ax2 = fig.add_subplot(gs[1:,:3]) 
ax2.imshow(seismogram_ani, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
ax2.plot(rectangle[:,0], rectangle[:,1]/dt, "--k")
ax2.set_title("Elastic ANI", fontsize = 15)
ax2.set_xlabel("Trace Index", fontsize = 15)
ax2.set_ylabel("Time [s]", fontsize = 15)
ax2.set_xticks(xloc)
ax2.set_yticks(tloc)
ax2.set_xticklabels(xlab)
ax2.set_yticklabels(tlab)

ax3 = fig.add_subplot(gs[:,3:4]) 
ax3.plot(trace_iso[ts], t[ts], label = "Trace ISO")
ax3.plot(trace_ani[ts], t[ts], label = "Trace ANI")
ax3.legend(loc = "lower right", fontsize = 10)
ax3.set_ylim([0.2, 1.0])
ax3.invert_yaxis()
ax3.set_title("Trace", fontsize = 15)
ax3.set_xlabel("Norm. Amp.", fontsize = 15)
ax3.set_ylabel("Time [s]", fontsize = 15)

ax4 = fig.add_subplot(gs[:,4:5]) 
ax4.plot(np.abs(fft_iso[mask]), freqs[mask], label = "Trace ISO")
ax4.plot(np.abs(fft_ani[mask]), freqs[mask], label = "Trace ANI")
ax4.legend(loc = "lower right", fontsize = 10)
ax4.set_ylim([0, 30])
ax4.invert_yaxis()
ax4.set_title("Amp. Spectra", fontsize = 15)
ax4.set_xlabel("Norm. Amp.", fontsize = 15)
ax4.set_ylabel("Frequency [Hz]", fontsize = 15)

fig.tight_layout()
plt.show()

