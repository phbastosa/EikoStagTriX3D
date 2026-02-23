## GPU-Accelerated Seismic Wave Simulation in Generally Anisotropic Media Using Eikonal-Guided Domain Clipping and Compressed Stiffness Representation. 

The EikoStagTriX3D/run is a managing directory. From there you can build the model, geometry, compile, execute and generate the results. 

First things first, download the repository and enter in the run folder:

`cd EikoStagTriX3D/run`

## Requirements:

- Ubuntu 24.04 LTS or another linux distribution
- CUDA toolkit 
- openmp 
- numpy 
- matplotlib

## Step One: Create model and geometry

Open the file **generate_setup.py** and set the parameters according the desired model and geometry. Here is an example for a layered model:

```python
# Model aspects

nx = 301              # Samples in x direction 
ny = 301              # Samples in y direction
nz = 51               # Samples in z direction

dx = 10.0             # Spacing in x direction
dy = 10.0             # Spacing in y direction
dz = 10.0             # Spacing in z direction

vp = np.array([1500, 1600, 1800, 2000, 2500]) # P-wave velocity [m/s]
vs = np.array([   0,  950, 1060, 1180, 1470]) # S-wave velocity [m/s]
ro = np.array([1000, 2350, 2400, 2450, 2500]) # Density [kg/m³]
z = np.array([  200,   50,  100,  100,   50]) # Layer thickness [m]

# Tsvanking anisotropic dimensionless parameters

E1 = np.array([0.0, 0.05, 0.10, 0.12, 0.0])  # epsilon 1
E2 = np.array([0.0, 0.07, 0.08, 0.10, 0.0])  # epsilon 2
D1 = np.array([0.0, 0.03, 0.05, 0.06, 0.0])  # delta 1
D2 = np.array([0.0, 0.02, 0.03, 0.05, 0.0])  # delta 2
D3 = np.array([0.0, 0.01, 0.02, 0.03, 0.0])  # delta 3
G1 = np.array([0.0, 0.05, 0.02, 0.01, 0.0])  # gamma 1
G2 = np.array([0.0, 0.10, 0.08, 0.06, 0.0])  # gamma 2

tilt = np.array([0, 10, 15, 20, 0]) * np.pi/180.0 # Dip angle [°] (rotation in y-axis: plane XZ)
azmt = np.array([0,  5, 10, 30, 0]) * np.pi/180.0 # Azimuth angle [°] (rotation in z-axis: plane XY)

# Geometry aspects

nsx = 1        # Number of sources in x direction
nsy = 1        # Number of sources in y direction
 
nrx = 117      # Number of receivers in x direction
nry = 4        # Number of receivers in y direction

sx_beg = 500   # Initial source position in x direction [m]
sx_end = 500   # Final source position in x direction [m]

sy_beg = 2500  # Initial source position in y direction [m]
sy_end = 2500  # Final source position in y direction [m]

rx_beg = 50    # Initial receiver position in x direction [m]
rx_end = 2950  # Final receiver position in x direction [m]

ry_beg = 50    # Initial receiver position in y direction [m]
ry_end = 2950  # Final receiver position in y direction [m]

sz = 0         # Sources depth [m] 
rz = 0         # Receivers depth [m]
```

Make sure you are in the run folder and run the python code as follows: 

`python3 generate_setup.py`

it will generate the files to run the seismic modeling code. 

## Step Two: setting **parameters.txt**

In this file you need to fit the same model parameters and the filenames that the previous code generated. It's already setted for the simple model experiment, but the main parameters are:

```text
x_samples = 301                           # <int>  
y_samples = 301                           # <int>  
z_samples = 51                            # <int>  

x_spacing = 10.0                          # [m] <float> 
y_spacing = 10.0                          # [m] <float> 
z_spacing = 10.0                          # [m] <float> 

.
.
.

modeling_type = 1             # for Rotated Staggered Grid Finite Difference scheme

eikonalClip = true            # to activate the eikonal volume computation
compression = true            # to activate the stiffness data compression

time_samples = 3001           # samples in seismogram 
time_spacing = 1e-3           # time discretization (200 ms in total)
max_frequency = 30.0          # Max frequency in Ricker wavelet

boundary_samples = 50         # for Cerjan Absorbing bondary condition 
boundary_damping = 0.0045     # Damping factor

snapshot = true               # To generate snapshots
beg_snap = 500                # Initial snapshot in samples dimention
end_snap = 1000               # Final snapshot in samples dimention
num_snap = 2                  # Number of snapshots   
```

After set the parameters file you can compile

`./program.sh -compile`

and run the code

`./program.sh -modeling`

## Step Three: generate results

Just run 

`python3 -B generate_results.py parameters.txt`

The following figures will be generated for the current setup:

<img width="1800" height="1560" alt="Image" src="https://github.com/user-attachments/assets/83298ca9-303a-4063-a284-6adfb7effd3b"/>

<img width="1800" height="1560" alt="Image" src="https://github.com/user-attachments/assets/9c278f6a-73d7-4fa2-9df9-f8f76547877b"/>

<img width="1800" height="1560" alt="Image" src="https://github.com/user-attachments/assets/b73396e7-e1c6-4487-8964-4fe41146034a"/>