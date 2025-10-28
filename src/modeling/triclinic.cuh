# ifndef TRICLINIC_CUH
# define TRICLINIC_CUH

# include <cuda_runtime.h>

# include "../geometry/geometry.hpp"

# define DGS 8

# define NSWEEPS 8
# define MESHDIM 3

# define NTHREADS 256

# define COMPRESS 65535

# define FDM1 6.97545e-4f 
# define FDM2 9.57031e-3f 
# define FDM3 7.97526e-2f 
# define FDM4 1.19628906f 

typedef unsigned short int uintc; 

class Triclinic
{
private:

    bool snapshot;
    int snapCount;
    
    std::vector<int> snapId;
    
    float * snapshot_in = nullptr;
    float * snapshot_out = nullptr;

    float * h_seismogram_Ps = nullptr;
    float * h_seismogram_Vx = nullptr;
    float * h_seismogram_Vy = nullptr;
    float * h_seismogram_Vz = nullptr;

    float * d_seismogram_Ps = nullptr;
    float * d_seismogram_Vx = nullptr;
    float * d_seismogram_Vy = nullptr;
    float * d_seismogram_Vz = nullptr;

    void set_models();
    void set_wavelet();
    void set_dampers();
    void set_eikonal();
    void set_slowness();

    void set_geometry();
    void set_snapshots();
    void set_seismogram();
    void set_wavefields();

    void eikonal_solver();
    void time_propagation();
    void wavefield_refresh();
    void get_shot_position();

    void compute_eikonal();
    void compute_snapshots();
    void compute_seismogram();
    void export_seismograms();
    void export_travelTimes();

    void show_information();
    void show_time_progress();

    void expand_boundary(float * input, float * output);
    void reduce_boundary(float * input, float * output);
    
    void get_compression(float * input, uintc * output, int N, float &max_value, float &min_value);

    int iDivUp(int a, int b);

protected:

    float dx, dy, dz, dt;
    int nxx, nyy, nzz, volsize;
    int nt, nx, ny, nz, nb, nPoints;
    int srcId, recId, sIdx, sIdy, sIdz;
    int tlag, nsnap, isnap, fsnap;
    int max_spread, timeId;
    int sBlocks, nBlocks;

    float bd, fmax;

    int total_levels;    
    float sx, sy, sz;    
    float dz2i, dx2i, dy2i, dsum;
    float dz2dx2, dz2dy2, dx2dy2;

    bool eikonalClip; 
    bool compression;

    int * d_sgnv = nullptr;
    int * d_sgnt = nullptr;

    float * d_skw = nullptr;
    float * d_rkwPs = nullptr;
    float * d_rkwVx = nullptr;
    float * d_rkwVy = nullptr;
    float * d_rkwVz = nullptr;

    int * d_rIdx = nullptr;
    int * d_rIdy = nullptr;
    int * d_rIdz = nullptr;

    float * d_wavelet = nullptr;

    Geometry * geometry;

    std::string snapshot_folder;
    std::string seismogram_folder;

    std::string modeling_type;
    std::string modeling_name;

    float * d1D = nullptr;
    float * d2D = nullptr;
    float * d3D = nullptr;

    float * h_S = nullptr;
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

    float * d_B = nullptr; uintc * dc_B = nullptr; float maxB; float minB;

    float * d_C11 = nullptr; uintc * dc_C11 = nullptr; float maxC11; float minC11;
    float * d_C12 = nullptr; uintc * dc_C12 = nullptr; float maxC12; float minC12;
    float * d_C13 = nullptr; uintc * dc_C13 = nullptr; float maxC13; float minC13;
    float * d_C14 = nullptr; uintc * dc_C14 = nullptr; float maxC14; float minC14;
    float * d_C15 = nullptr; uintc * dc_C15 = nullptr; float maxC15; float minC15;
    float * d_C16 = nullptr; uintc * dc_C16 = nullptr; float maxC16; float minC16;

    float * d_C22 = nullptr; uintc * dc_C22 = nullptr; float maxC22; float minC22;
    float * d_C23 = nullptr; uintc * dc_C23 = nullptr; float maxC23; float minC23;
    float * d_C24 = nullptr; uintc * dc_C24 = nullptr; float maxC24; float minC24;
    float * d_C25 = nullptr; uintc * dc_C25 = nullptr; float maxC25; float minC25;
    float * d_C26 = nullptr; uintc * dc_C26 = nullptr; float maxC26; float minC26;

    float * d_C33 = nullptr; uintc * dc_C33 = nullptr; float maxC33; float minC33;
    float * d_C34 = nullptr; uintc * dc_C34 = nullptr; float maxC34; float minC34;
    float * d_C35 = nullptr; uintc * dc_C35 = nullptr; float maxC35; float minC35;
    float * d_C36 = nullptr; uintc * dc_C36 = nullptr; float maxC36; float minC36;

    float * d_C44 = nullptr; uintc * dc_C44 = nullptr; float maxC44; float minC44;
    float * d_C45 = nullptr; uintc * dc_C45 = nullptr; float maxC45; float minC45;
    float * d_C46 = nullptr; uintc * dc_C46 = nullptr; float maxC46; float minC46;

    float * d_C55 = nullptr; uintc * dc_C55 = nullptr; float maxC55; float minC55;
    float * d_C56 = nullptr; uintc * dc_C56 = nullptr; float maxC56; float minC56;

    float * d_C66 = nullptr; uintc * dc_C66 = nullptr; float maxC66; float minC66;

    virtual void initialization() = 0;
    virtual void compute_velocity() = 0;
    virtual void compute_pressure() = 0;

public:

    std::string parameters;

    void set_parameters();
    void run_wave_propagation();
};

__global__ void time_set(float * T, int volsize);

__global__ void time_init(float * T, float * S, float sx, float sy, float sz, float dx, float dy, 
                          float dz, int sIdx, int sIdy, int sIdz, int nxx, int nzz, int nb);

__global__ void inner_sweep(float * S, float * T, int * sgnt, int * sgnv, int sgni, int sgnj, int sgnk, 
                            int level, int xOffset, int yOffset, int xSweepOffset, int ySweepOffset, int zSweepOffset, 
                            int nxx, int nyy, int nzz, float dx, float dy, float dz, float dx2i, float dy2i, float dz2i, 
                            float dz2dx2, float dz2dy2, float dx2dy2, float dsum);

__global__ void float_quasi_slowness(float * T, float * S, float dx, float dy, float dz, int sIdx, int sIdy, int sIdz, int nxx, int nyy, int nzz, int nb,
                                     float * C11, float * C12, float * C13, float * C14, float * C15, float * C16, float * C22, float * C23, float * C24, float * C25, 
                                     float * C26, float * C33, float * C34, float * C35, float * C36, float * C44, float * C45, float * C46, float * C55, float * C56, 
                                     float * C66);

__global__ void uintc_quasi_slowness(float * T, float * S, float dx, float dy, float dz, int sIdx, int sIdy, int sIdz, int nxx, int nyy, int nzz, int nb,
                                     uintc * C11, uintc * C12, uintc * C13, uintc * C14, uintc * C15, uintc * C16, uintc * C22, uintc * C23, uintc * C24, uintc * C25, 
                                     uintc * C26, uintc * C33, uintc * C34, uintc * C35, uintc * C36, uintc * C44, uintc * C45, uintc * C46, uintc * C55, uintc * C56, 
                                     uintc * C66, float minC11, float maxC11, float minC12, float maxC12, float minC13, float maxC13, float minC14, float maxC14, 
                                     float minC15, float maxC15, float minC16, float maxC16, float minC22, float maxC22, float minC23, float maxC23, float minC24, 
                                     float maxC24, float minC25, float maxC25, float minC26, float maxC26, float minC33, float maxC33, float minC34, float maxC34, 
                                     float minC35, float maxC35, float minC36, float maxC36, float minC44, float maxC44, float minC45, float maxC45, float minC46, 
                                     float maxC46, float minC55, float maxC55, float minC56, float maxC56, float minC66, float maxC66);

__global__ void compute_seismogram_GPU(float * WF, int * rIdx, int * rIdy, int * rIdz, float * rkw, float * seismogram, int spread, int tId, int tlag, int nt, int nxx, int nzz);

__device__ float get_boundary_damper(float * damp1D, float * damp2D, float * damp3D, int i, int j, int k, int nxx, int nyy, int nzz, int nabc);

# endif