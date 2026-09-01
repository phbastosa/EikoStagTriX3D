import numpy as np

from scipy.signal import resample

ns = 49
nt = 3001
nr = 10000
dt = 1e-3

new_dt = 4e-3
new_nt = int((nt-1)*dt/new_dt) + 1

folder = "../outputs/seismograms/input_KDM_iso_"

for sId in range(ns):

    print(f"writing file {sId+1} of {ns}")

    prefix = f"triclinic_issg_Ps_nStations{nr}_nSamples{nt}_shot_"

    data = np.fromfile(folder + prefix + f"{sId+1}.bin", dtype = np.float32, count = nt*nr).reshape([nt,nr], order = "F")

    new_data = resample(data, new_nt, axis = 0)

    prefix = f"triclinic_issg_Ps_nStations{nr}_nSamples{new_nt}_shot_"

    new_data.flatten("F").astype(np.float32, order = "F").tofile(folder + prefix + f"{sId+1}.bin")
