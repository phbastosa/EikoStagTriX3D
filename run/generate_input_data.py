import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf

parameters = sys.argv[1]

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

sps_path = pyf.catch_parameter(parameters, "SPS")
rps_path = pyf.catch_parameter(parameters, "RPS")

SPS = np.loadtxt(sps_path, dtype = np.float32, delimiter = ",")
RPS = np.loadtxt(rps_path, dtype = np.float32, delimiter = ",")

ns = len(SPS)
nr = len(RPS)

folder = "../outputs/seismograms/"

gstd = 30
velocity = 1400
time_delay = 0.3

timeId = np.arange(nt)

for sId in range(ns): 

    file = f"triclinic_issg_Ps_nStations{nr}_nSamples{nt}_shot_{sId+1}.bin" 

    data = pyf.read_binary_matrix(nt, nr, folder + file)  

    data *= 1.0 / np.max(np.abs(data)) 

    offset = RPS[:,0] - SPS[sId,0]

    distance = np.sqrt((RPS[:,0] - SPS[sId,0])**2 + (RPS[:,1] - SPS[sId,1])**2)

    tId = np.array((distance/velocity + time_delay) / dt, dtype = int)

    for rId in range(nr):
        
        data[:tId[rId], rId] *= np.exp(-0.5*((timeId - tId[rId]) / gstd)**2)[:tId[rId]]

    data[:700] *= 0.0

    data.flatten("F").astype(np.float32, order = "F").tofile(folder + "input_KDM_iso_" + file)