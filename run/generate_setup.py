import numpy as np

nx = 301
ny = 301
nz = 51

dx = 10.0
dy = 10.0
dz = 10.0

# Acquisition geometry

nsx = 1
nsy = 1

nrx = 117 
nry = 4

ns = nsx*nsy
nr = nrx*nry

sx, sy = 500, 2500 
rx, ry = np.meshgrid(np.linspace(50, 2950, nrx), 
                     np.linspace(50, 2950, nry))

SPS = np.zeros((ns, 3), dtype = float)
RPS = np.zeros((nr, 3), dtype = float)
XPS = np.zeros((ns, 3), dtype = int)

SPS[:,0] = np.reshape(sx, [ns], order = "F")
SPS[:,1] = np.reshape(sy, [ns], order = "F")
SPS[:,2] = np.zeros(ns)

RPS[:,0] = np.reshape(rx, [nr], order = "C")
RPS[:,1] = np.reshape(ry, [nr], order = "C")
RPS[:,2] = np.zeros(nr)

XPS[:, 0] = np.arange(ns)
XPS[:, 1] = np.zeros(ns) 
XPS[:, 2] = np.zeros(ns) + nr 

path_SPS = "../inputs/geometry/EikoStagTriX3D_SPS.txt"
path_RPS = "../inputs/geometry/EikoStagTriX3D_RPS.txt"
path_XPS = "../inputs/geometry/EikoStagTriX3D_XPS.txt"

np.savetxt(path_SPS, SPS, fmt = "%.2f", delimiter = ",")
np.savetxt(path_RPS, RPS, fmt = "%.2f", delimiter = ",")
np.savetxt(path_XPS, XPS, fmt = "%.0f", delimiter = ",")

offset = np.sqrt((SPS[0,0] - RPS[:,0])**2 + (SPS[0,1] - RPS[:,1])**2)

vp = np.array([1500,1600,1800,2000,2500])
vs = np.array([   0, 950,1060,1180,1470])
ro = np.array([1000,2350,2400,2450,2500])
z = np.array([200, 50, 100, 100])

E1 = np.array([0.0, 0.05, 0.10, 0.12, 0.0])
E2 = np.array([0.0, 0.07, 0.08, 0.10, 0.0])
D1 = np.array([0.0, 0.03, 0.05, 0.06, 0.0])
D2 = np.array([0.0, 0.02, 0.03, 0.05, 0.0])
D3 = np.array([0.0, 0.01, 0.02, 0.03, 0.0])
G1 = np.array([0.0, 0.05, 0.02, 0.01, 0.0])
G2 = np.array([0.0, 0.10, 0.08, 0.06, 0.0])

tilt = np.array([0, 10, 15, 20, 0]) * np.pi/180.0
azmt = np.array([0,  5, 10, 30, 0]) * np.pi/180.0

S = np.zeros((nz, nx, ny))
B = np.zeros((nz, nx, ny))

C11 = np.zeros_like(S)
C12 = np.zeros_like(S) 
C13 = np.zeros_like(S) 
C14 = np.zeros_like(S) 
C15 = np.zeros_like(S)
C16 = np.zeros_like(S)

C22 = np.zeros_like(S)
C23 = np.zeros_like(S)
C24 = np.zeros_like(S)
C25 = np.zeros_like(S)
C26 = np.zeros_like(S)

C33 = np.zeros_like(S)
C34 = np.zeros_like(S)
C35 = np.zeros_like(S)
C36 = np.zeros_like(S)

C44 = np.zeros_like(S)
C45 = np.zeros_like(S)
C46 = np.zeros_like(S)

C55 = np.zeros_like(S)
C56 = np.zeros_like(S)

C66 = np.zeros_like(S)

C = np.zeros((6,6))
M = np.zeros((6,6))

c11 = c12 = c13 = c14 = c15 = c16 = 0
c22 = c23 = c24 = c25 = c26 = 0
c33 = c34 = c35 = c36 = 0
c44 = c45 = c46 = 0
c55 = c56 = 0
c66 = 0

SI = 1e9

for i in range(len(vp)):
    
    layer = int(np.sum(z[:i])/dz)

    c33 = ro[i]*vp[i]**2 / SI
    c55 = ro[i]*vs[i]**2 / SI

    c11 = c33*(1.0 + 2.0*E2[i])
    c22 = c33*(1.0 + 2.0*E1[i])

    c66 = c55*(1.0 + 2.0*G1[i])
    c44 = c66/(1.0 + 2.0*G2[i])

    c23 = np.sqrt((c33 - c44)**2 + 2.0*D1[i]*c33*(c33 - c44)) - c44
    c13 = np.sqrt((c33 - c55)**2 + 2.0*D2[i]*c33*(c33 - c55)) - c55
    c12 = np.sqrt((c11 - c66)**2 + 2.0*D3[i]*c11*(c11 - c66)) - c66

    C[0,0] = c11; C[0,1] = c12; C[0,2] = c13; C[0,3] = c14; C[0,4] = c15; C[0,5] = c16  
    C[1,0] = c12; C[1,1] = c22; C[1,2] = c23; C[1,3] = c24; C[1,4] = c25; C[1,5] = c26  
    C[2,0] = c13; C[2,1] = c12; C[2,2] = c33; C[2,3] = c34; C[2,4] = c35; C[2,5] = c36  
    C[3,0] = c14; C[3,1] = c24; C[3,2] = c34; C[3,3] = c44; C[3,4] = c45; C[3,5] = c46  
    C[4,0] = c15; C[4,1] = c25; C[4,2] = c35; C[4,3] = c45; C[4,4] = c55; C[4,5] = c56  
    C[5,0] = c16; C[5,1] = c26; C[5,2] = c36; C[5,3] = c46; C[5,4] = c56; C[5,5] = c66  

    c = np.cos(tilt[i]); s = np.sin(tilt[i])
    Rtilt = np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]])

    c = np.cos(azmt[i]); s = np.sin(azmt[i])
    Razmt = np.array([[c,-s, 0],[s, c, 0],[0, 0, 1]])

    R = Rtilt @ Razmt

    pax = R[0,0]; pay = R[0,1]; paz = R[0,2]
    pbx = R[1,0]; pby = R[1,1]; pbz = R[1,2]
    pcx = R[2,0]; pcy = R[2,1]; pcz = R[2,2]

    M[0,0] = pax*pax; M[1,0] = pbx*pbx; M[2,0] = pcx*pcx
    M[0,1] = pay*pay; M[1,1] = pby*pby; M[2,1] = pcy*pcy
    M[0,2] = paz*paz; M[1,2] = pbz*pbz; M[2,2] = pcz*pcz

    M[3,0] = pbx*pcx; M[4,0] = pcx*pax; M[5,0] = pax*pbx
    M[3,1] = pby*pcy; M[4,1] = pcy*pay; M[5,1] = pay*pby
    M[3,2] = pbz*pcz; M[4,2] = pcz*paz; M[5,2] = paz*pbz

    M[0,3] = 2.0*pay*paz; M[1,3] = 2.0*pby*pbz; M[2,3] = 2.0*pcy*pcz
    M[0,4] = 2.0*paz*pax; M[1,4] = 2.0*pbz*pbx; M[2,4] = 2.0*pcz*pcx
    M[0,5] = 2.0*pax*pay; M[1,5] = 2.0*pbx*pby; M[2,5] = 2.0*pcx*pcy

    M[3,3] = pby*pcz + pbz*pcy; M[4,3] = pay*pcz + paz*pcy; M[5,3] = pay*pbz + paz*pby
    M[3,4] = pbx*pcz + pbz*pcx; M[4,4] = paz*pcx + pax*pcz; M[5,4] = paz*pbx + pax*pbz
    M[3,5] = pby*pcx + pbx*pcy; M[4,5] = pax*pcy + pay*pcx; M[5,5] = pax*pby + pay*pbx

    Cr = (M @ C @ M.T) * SI

    S[layer:] = 1.0 / vp[i]
    B[layer:] = 1.0 / ro[i]

    C11[layer:] = Cr[0,0]; C12[layer:] = Cr[0,1]; C13[layer:] = Cr[0,2]; C14[layer:] = Cr[0,3]
    C15[layer:] = Cr[0,4]; C16[layer:] = Cr[0,5]; C22[layer:] = Cr[1,1]; C23[layer:] = Cr[1,2]
    C24[layer:] = Cr[1,3]; C25[layer:] = Cr[1,4]; C26[layer:] = Cr[1,5]; C33[layer:] = Cr[2,2]
    C34[layer:] = Cr[2,3]; C35[layer:] = Cr[2,4]; C36[layer:] = Cr[2,5]; C44[layer:] = Cr[3,3]
    C45[layer:] = Cr[3,4]; C46[layer:] = Cr[3,5]; C55[layer:] = Cr[4,4]; C56[layer:] = Cr[4,5]
    C66[layer:] = Cr[5,5]; 

S.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_slowness.bin")
B.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_buoyancy.bin")

C11.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C11.bin")
C12.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C12.bin")
C13.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C13.bin")
C14.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C14.bin")
C15.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C15.bin")
C16.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C16.bin")

C22.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C22.bin")
C23.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C23.bin")
C24.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C24.bin")
C25.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C25.bin")
C26.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C26.bin")

C33.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C33.bin")
C34.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C34.bin")
C35.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C35.bin")
C36.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C36.bin")

C44.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C44.bin")
C45.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C45.bin")
C46.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C46.bin")

C55.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C55.bin")
C56.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C56.bin")

C66.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX3D_C66.bin")
