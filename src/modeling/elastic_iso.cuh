# ifndef ELASTIC_ISO_CUH
# define ELASTIC_ISO_CUH

# include "modeling.cuh"

class Elastic_ISO : public Modeling
{    
    uintc * d_B = nullptr; float maxB; float minB;

    uintc * d_C13 = nullptr; float maxC13; float minC13;
    uintc * d_C55 = nullptr; float maxC55; float minC55;

    void initialization();
    void compute_eikonal();
    void compute_velocity();
    void compute_pressure();

    void set_specifications();
};

__global__ void compute_velocity_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, uintc * B,
                                     float maxB, float minB, float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, 
                                     int tlag, int sIdx, int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt);

__global__ void compute_pressure_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, uintc * C55, 
                                     uintc * C13, float maxC55, float minC55, float maxC13, float minC13, int tId, int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz);

# endif