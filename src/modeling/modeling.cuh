# ifndef MODELING_CUH
# define MODELING_CUH

# include <cuda_runtime.h>

# include "../geometry/geometry.hpp"

# define NSWEEPS 8
# define MESHDIM 3

# define COMPRESS 65535

# define FDM1 6.97545e-4f 
# define FDM2 9.57031e-3f 
# define FDM3 7.97526e-2f 
# define FDM4 1.19628906f 

typedef unsigned short int uintc; 

class Modeling
{
private:

    bool snapshot;
    int snapCount;
    
    std::vector<int> snapId;
    
    float * snapshot_in = nullptr;
    float * snapshot_out = nullptr;

    float * d_seismogram = nullptr;
    float * h_seismogram = nullptr;

    void set_wavelet();
    void set_dampers();
    void set_eikonal();

    int iDivUp(int a, int b);

    void compute_snapshots();
    void compute_seismogram();

    void set_wavefields();
    void initialization();

    void show_time_progress();

protected:

    float dx, dy, dz, dt;
    int nxx, nyy, nzz, volsize;
    int nt, nx, ny, nz, nb, nPoints;
    int tlag, recId, sIdx, sIdy, sIdz;
    int nThreads, sBlocks, nBlocks;
    int nsnap, isnap, fsnap;
    int max_spread, timeId;

    float bd, fmax;

    int total_levels;    
    float dz2i, dx2i, dy2i, dsum;
    float dz2dx2, dz2dy2, dx2dy2;

    int * d_sgnv = nullptr;
    int * d_sgnt = nullptr;

    std::string snapshot_folder;
    std::string seismogram_folder;

    std::string modeling_type;
    std::string modeling_name;

    float * S = nullptr;

    float * d1D = nullptr;
    float * d2D = nullptr;
    float * d3D = nullptr;

    float * d_S = nullptr;

    float * d_T = nullptr;
    float * d_P = nullptr;
    
    float * d_Vx = nullptr;
    float * d_Vy = nullptr;
    float * d_Vz = nullptr;

    float * d_Txx = nullptr;
    float * d_Tyy = nullptr;
    float * d_Tzz = nullptr;

    float * d_Txz = nullptr;
    float * d_Tyz = nullptr;
    float * d_Txy = nullptr;

    int * d_rIdx = nullptr;
    int * d_rIdy = nullptr;
    int * d_rIdz = nullptr;

    float * d_wavelet = nullptr;

    virtual void set_specifications() = 0;
    
    virtual void compute_eikonal() = 0;
    virtual void compute_velocity() = 0;
    virtual void compute_pressure() = 0;

    void eikonal_solver();

    void expand_boundary(float * input, float * output);
    void reduce_boundary(float * input, float * output);
    
    void compression(float * input, uintc * output, int N, float &max_value, float &min_value);

public:

    int srcId;

    std::string parameters;

    Geometry * geometry;

    void set_parameters();
    void show_information();
    void time_propagation();    
    void export_seismogram();
};

__global__ void time_set(float * T, int volsize);

__global__ void time_init(float * T, float * S, float sx, float sy, float sz, float dx, float dy, 
                          float dz, int sIdx, int sIdy, int sIdz, int nxx, int nzz, int nb);

__global__ void inner_sweep(float * S, float * T, int * sgnt, int * sgnv, int sgni, int sgnj, int sgnk, 
                            int level, int xOffset, int yOffset, int xSweepOffset, int ySweepOffset, int zSweepOffset, 
                            int nxx, int nyy, int nzz, float dx, float dy, float dz, float dx2i, float dy2i, float dz2i, 
                            float dz2dx2, float dz2dy2, float dx2dy2, float dsum);

__global__ void compute_seismogram_GPU(float * P, int * rIdx, int * rIdy, int * rIdz, float * seismogram, int spread, int tId, int tlag, int nt, int nxx, int nzz);

__device__ float get_boundary_damper(float * damp1D, float * damp2D, float * damp3D, int i, int j, int k, int nxx, int nyy, int nzz, int nabc);

# endif