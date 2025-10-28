# ifndef TRICLINIC_SSG_CUH
# define TRICLINIC_SSG_CUH

# include "triclinic.cuh"

class Triclinic_SSG : public Triclinic
{    
    void initialization();
    void compute_velocity();
    void compute_pressure();
};

__global__ void float_compute_velocity_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, float * B,
                                           float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, int tlag, int sIdx, 
                                           int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt, bool eikonal);
                                           
__global__ void float_compute_pressure_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, 
                                           float * C11, float * C12, float * C13, float * C14, float * C15, float * C16, float * C22, float * C23, float * C24, float * C25, float * C26, 
                                           float * C33, float * C34, float * C35, float * C36, float * C44, float * C45, float * C46, float * C55, float * C56, float * C66, int tId, 
                                           int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz, bool eikonal);

__global__ void uintc_compute_velocity_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, uintc * B,
                                           float maxB, float minB, float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, 
                                           int tlag, int sIdx, int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt, bool eikonal);

__global__ void uintc_compute_pressure_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, 
                                           uintc * C11, uintc * C12, uintc * C13, uintc * C14, uintc * C15, uintc * C16, uintc * C22, uintc * C23, uintc * C24, uintc * C25, uintc * C26, 
                                           uintc * C33, uintc * C34, uintc * C35, uintc * C36, uintc * C44, uintc * C45, uintc * C46, uintc * C55, uintc * C56, uintc * C66, int tId, 
                                           int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz, float minC11, float maxC11, float minC12, float maxC12, 
                                           float minC13, float maxC13, float minC14, float maxC14, float minC15, float maxC15, float minC16, float maxC16, float minC22, float maxC22, 
                                           float minC23, float maxC23, float minC24, float maxC24, float minC25, float maxC25, float minC26, float maxC26, float minC33, float maxC33, 
                                           float minC34, float maxC34, float minC35, float maxC35, float minC36, float maxC36, float minC44, float maxC44, float minC45, float maxC45, 
                                           float minC46, float maxC46, float minC55, float maxC55, float minC56, float maxC56, float minC66, float maxC66, bool eikonal);

# endif
