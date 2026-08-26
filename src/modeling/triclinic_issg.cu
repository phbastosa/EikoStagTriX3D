# include "triclinic_issg.cuh"

void Triclinic_iSSG::set_modeling_type()
{
    modeling_name = "Triclinic media with interpolated Standard Staggered Grid";
    modeling_type = "triclinic_issg";
}

void Triclinic_iSSG::set_rec_weights()
{
    int * h_rIdx = new int[geometry->nrec]();
    int * h_rIdy = new int[geometry->nrec]();
    int * h_rIdz = new int[geometry->nrec]();

    float * h_rkwPs = new float[DGS*DGS*DGS*geometry->nrec]();

    for (recId = 0; recId < geometry->nrec; recId++)
    {
        float rx = geometry->xrec[recId];
        float ry = geometry->yrec[recId];
        float rz = geometry->zrec[recId];
        
        int rIdx = (int)((rx + 0.5f*dh) / dh);
        int rIdy = (int)((ry + 0.5f*dh) / dh);
        int rIdz = (int)((rz + 0.5f*dh) / dh);
    
        auto rkwPs = kaiser_weights(rx, ry, rz, rIdx, rIdy, rIdz, dh, dh, dh);
        
        for (int zId = 0; zId < DGS; zId++)
            for (int xId = 0; xId < DGS; xId++)
                for (int yId = 0; yId < DGS; yId++)
                    h_rkwPs[zId + xId*DGS + yId*DGS*DGS + recId*DGS*DGS*DGS] = rkwPs[zId][xId][yId];

        h_rIdx[recId] = rIdx + nb;
        h_rIdy[recId] = rIdy + nb;
        h_rIdz[recId] = rIdz + nb;
    }

    cudaMemcpy(d_rkwPs, h_rkwPs, DGS*DGS*DGS*geometry->nrec*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rIdx, h_rIdx, geometry->nrec*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdy, h_rIdy, geometry->nrec*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdz, h_rIdz, geometry->nrec*sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_rkwPs;
    delete[] h_rIdx;
    delete[] h_rIdy;
    delete[] h_rIdz;
}

void Triclinic_iSSG::set_src_weights()
{
    float * h_skw = new float[DGS*DGS*DGS]();

    auto skw = kaiser_weights(sx, sy, sz, sIdx, sIdy, sIdz, dh, dh, dh);

    for (int yId = 0; yId < DGS; yId++)
        for (int xId = 0; xId < DGS; xId++)
            for (int zId = 0; zId < DGS; zId++)
                h_skw[zId + xId*DGS + yId*DGS*DGS] = skw[zId][xId][yId];

    cudaMemcpy(d_skw, h_skw, DGS*DGS*DGS*sizeof(float), cudaMemcpyHostToDevice);
    
    delete[] h_skw;
}

void Triclinic_iSSG::compute_velocity()
{
    if (compression)
    {
        uintc_compute_velocity_issg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_T,dc_B,
                                                          maxB,minB,d1D,d2D,d3D,d_wavelet,dh,dh,dh,dt,timeId,tlag,sIdx, 
                                                          sIdy,sIdz,d_skw,nxx,nyy,nzz,nb,nt,eikonalClip);
    }
    else 
    {
        float_compute_velocity_issg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_T,d_B,
                                                          d1D,d2D,d3D,d_wavelet,dh,dh,dh,dt,timeId,tlag,sIdx,sIdy,sIdz,
                                                          d_skw,nxx,nyy,nzz,nb,nt,eikonalClip);
    }
}

void Triclinic_iSSG::compute_pressure()
{
    if (compression)
    {
        uintc_compute_pressure_issg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_P,d_T, 
                                                          dc_C11,dc_C12,dc_C13,dc_C14,dc_C15,dc_C16,dc_C22,dc_C23,dc_C24,
                                                          dc_C25,dc_C26,dc_C33,dc_C34,dc_C35,dc_C36,dc_C44,dc_C45,dc_C46,
                                                          dc_C55,dc_C56,dc_C66,timeId,tlag,dh,dh,dh,dt,nxx,nyy,nzz,minC11, 
                                                          maxC11,minC12,maxC12,minC13,maxC13,minC14,maxC14,minC15,maxC15,
                                                          minC16,maxC16,minC22,maxC22,minC23,maxC23,minC24,maxC24,minC25,
                                                          maxC25,minC26,maxC26,minC33,maxC33,minC34,maxC34,minC35,maxC35, 
                                                          minC36,maxC36,minC44,maxC44,minC45,maxC45,minC46,maxC46,minC55, 
                                                          maxC55,minC56,maxC56,minC66,maxC66,eikonalClip);
    }
    else 
    {
        float_compute_pressure_issg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_P,d_T, 
                                                          d_C11,d_C12,d_C13,d_C14,d_C15,d_C16,d_C22,d_C23,d_C24,d_C25,
                                                          d_C26,d_C33,d_C34,d_C35,d_C36,d_C44,d_C45,d_C46,d_C55,d_C56, 
                                                          d_C66,timeId,tlag,dh,dh,dh,dt,nxx,nyy,nzz,eikonalClip);
    }
}

__global__ void uintc_compute_velocity_issg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, uintc * B,
                                            float maxB, float minB, float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, 
                                            int tlag, int sIdx, int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    const float FDM1 = 6.97545e-4f; 
    const float FDM2 = 9.57031e-3f; 
    const float FDM3 = 7.97526e-2f; 
    const float FDM4 = 1.19628906f;     

    const float inv_dx = 1.0f / dx;
    const float inv_dy = 1.0f / dy;
    const float inv_dz = 1.0f / dz;

    const size_t nxx_nzz = nxx*nzz;

    int k = (int) (index / nxx_nzz);         
    int j = (int) (index - k*nxx_nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx_nzz); 
    
    size_t bsi = i, bsj = j*nzz, bsk = k*nxx_nzz;  

    size_t ip1 = i+1, ip2 = i+2, ip3 = i+3, ip4 = i+4;
    size_t im1 = i-1, im2 = i-2, im3 = i-3, im4 = i-4;

    size_t jp1 = (j+1)*nzz, jp2 = (j+2)*nzz, jp3 = (j+3)*nzz, jp4 = (j+4)*nzz;
    size_t jm1 = (j-1)*nzz, jm2 = (j-2)*nzz, jm3 = (j-3)*nzz, jm4 = (j-4)*nzz;

    size_t kp1 = (k+1)*nxx_nzz, kp2 = (k+2)*nxx_nzz, kp3 = (k+3)*nxx_nzz, kp4 = (k+4)*nxx_nzz;
    size_t km1 = (k-1)*nxx_nzz, km2 = (k-2)*nxx_nzz, km3 = (k-3)*nxx_nzz, km4 = (k-4)*nxx_nzz;

    T[index] = (eikonal) ? T[index] : 0.0f;

    if((i >= 4) && (i < nzz-4) && (j >= 4) && (j < nxx-4) && (k >= 4) && (k < nyy-4)) 
    {
        if ((T[index] < (float)(tId + tlag)*dt))
        {
            float b  = (minB + (static_cast<float>(B[bsi + bsj + bsk]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float bi = (minB + (static_cast<float>(B[ip1 + bsj + bsk]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float bj = (minB + (static_cast<float>(B[bsi + jp1 + bsk]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float bk = (minB + (static_cast<float>(B[bsi + bsj + kp1]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float dTxx_dx = (FDM1*(Txx[bsi + jm3 + bsk] - Txx[bsi + jp4 + bsk]) +
                             FDM2*(Txx[bsi + jp3 + bsk] - Txx[bsi + jm2 + bsk]) +
                             FDM3*(Txx[bsi + jm1 + bsk] - Txx[bsi + jp2 + bsk]) +
                             FDM4*(Txx[bsi + jp1 + bsk] - Txx[bsi + bsj + bsk])) * inv_dx;

            float dTxy_dy = (FDM1*(Txy[bsi + bsj + km4] - Txy[bsi + bsj + kp3]) +
                             FDM2*(Txy[bsi + bsj + kp2] - Txy[bsi + bsj + km3]) +
                             FDM3*(Txy[bsi + bsj + km2] - Txy[bsi + bsj + kp1]) +
                             FDM4*(Txy[bsi + bsj + bsk] - Txy[bsi + bsj + km1])) * inv_dy;

            float dTxz_dz = (FDM1*(Txz[im4 + bsj + bsk] - Txz[ip3 + bsj + bsk]) +
                             FDM2*(Txz[ip2 + bsj + bsk] - Txz[im3 + bsj + bsk]) +
                             FDM3*(Txz[im2 + bsj + bsk] - Txz[ip1 + bsj + bsk]) +
                             FDM4*(Txz[bsi + bsj + bsk] - Txz[im1 + bsj + bsk])) * inv_dz;

            float dTxy_dx = (FDM1*(Txy[bsi + jm4 + bsk] - Txy[bsi + jp3 + bsk]) +
                             FDM2*(Txy[bsi + jp2 + bsk] - Txy[bsi + jm3 + bsk]) +
                             FDM3*(Txy[bsi + jm2 + bsk] - Txy[bsi + jp1 + bsk]) +
                             FDM4*(Txy[bsi + bsj + bsk] - Txy[bsi + jm1 + bsk])) * inv_dx;

            float dTyy_dy = (FDM1*(Tyy[bsi + bsj + km3] - Tyy[bsi + bsj + kp4]) +
                             FDM2*(Tyy[bsi + bsj + kp3] - Tyy[bsi + bsj + km2]) +
                             FDM3*(Tyy[bsi + bsj + km1] - Tyy[bsi + bsj + kp2]) +
                             FDM4*(Tyy[bsi + bsj + kp1] - Tyy[bsi + bsj + bsk])) * inv_dy;

            float dTyz_dz = (FDM1*(Tyz[im4 + bsj + bsk] - Tyz[ip3 + bsj + bsk]) +
                             FDM2*(Tyz[ip2 + bsj + bsk] - Tyz[im3 + bsj + bsk]) +
                             FDM3*(Tyz[im2 + bsj + bsk] - Tyz[ip1 + bsj + bsk]) +
                             FDM4*(Tyz[bsi + bsj + bsk] - Tyz[im1 + bsj + bsk])) * inv_dz;

            float dTxz_dx = (FDM1*(Txz[bsi + jm4 + bsk] - Txz[bsi + jp3 + bsk]) +
                             FDM2*(Txz[bsi + jp2 + bsk] - Txz[bsi + jm3 + bsk]) +
                             FDM3*(Txz[bsi + jm2 + bsk] - Txz[bsi + jp1 + bsk]) +
                             FDM4*(Txz[bsi + bsj + bsk] - Txz[bsi + jm1 + bsk])) * inv_dx;

            float dTyz_dy = (FDM1*(Tyz[bsi + bsj + km4] - Tyz[bsi + bsj + kp3]) +
                             FDM2*(Tyz[bsi + bsj + kp2] - Tyz[bsi + bsj + km3]) +
                             FDM3*(Tyz[bsi + bsj + km2] - Tyz[bsi + bsj + kp1]) +
                             FDM4*(Tyz[bsi + bsj + bsk] - Tyz[bsi + bsj + km1])) * inv_dy;

            float dTzz_dz = (FDM1*(Tzz[im3 + bsj + bsk] - Tzz[ip4 + bsj + bsk]) +
                             FDM2*(Tzz[ip3 + bsj + bsk] - Tzz[im2 + bsj + bsk]) +
                             FDM3*(Tzz[im1 + bsj + bsk] - Tzz[ip2 + bsj + bsk]) +
                             FDM4*(Tzz[ip1 + bsj + bsk] - Tzz[bsi + bsj + bsk])) * inv_dz;

            float Bx = 0.5f*(b + bj);
            float By = 0.5f*(b + bk);
            float Bz = 0.5f*(b + bi);

            Vx[index] += dt*Bx*(dTxx_dx + dTxy_dy + dTxz_dz); 
            Vy[index] += dt*By*(dTxy_dx + dTyy_dy + dTyz_dz); 
            Vz[index] += dt*Bz*(dTxz_dx + dTyz_dy + dTzz_dz); 
        }

        float damper = get_boundary_damper(damp1D, damp2D, damp3D, i, j, k, nxx, nyy, nzz, nb);

        Vx[index] *= damper;
        Vy[index] *= damper;
        Vz[index] *= damper;

        Txx[index] *= damper;
        Tyy[index] *= damper;
        Tzz[index] *= damper;
        Txz[index] *= damper;
        Tyz[index] *= damper;
        Txy[index] *= damper;
    }
}

__global__ void float_compute_velocity_issg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, float * B,
                                            float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, int tlag, int sIdx, 
                                            int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    const float FDM1 = 6.97545e-4f; 
    const float FDM2 = 9.57031e-3f; 
    const float FDM3 = 7.97526e-2f; 
    const float FDM4 = 1.19628906f;     

    const float inv_dx = 1.0f / dx;
    const float inv_dy = 1.0f / dy;
    const float inv_dz = 1.0f / dz;

    const size_t nxx_nzz = nxx*nzz;

    int k = (int) (index / nxx_nzz);         
    int j = (int) (index - k*nxx_nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx_nzz); 
    
    size_t bsi = i, bsj = j*nzz, bsk = k*nxx_nzz;  

    size_t ip1 = i+1, ip2 = i+2, ip3 = i+3, ip4 = i+4;
    size_t im1 = i-1, im2 = i-2, im3 = i-3, im4 = i-4;

    size_t jp1 = (j+1)*nzz, jp2 = (j+2)*nzz, jp3 = (j+3)*nzz, jp4 = (j+4)*nzz;
    size_t jm1 = (j-1)*nzz, jm2 = (j-2)*nzz, jm3 = (j-3)*nzz, jm4 = (j-4)*nzz;

    size_t kp1 = (k+1)*nxx_nzz, kp2 = (k+2)*nxx_nzz, kp3 = (k+3)*nxx_nzz, kp4 = (k+4)*nxx_nzz;
    size_t km1 = (k-1)*nxx_nzz, km2 = (k-2)*nxx_nzz, km3 = (k-3)*nxx_nzz, km4 = (k-4)*nxx_nzz;

    T[index] = (eikonal) ? T[index] : 0.0f;

    if((i >= 4) && (i < nzz-4) && (j >= 4) && (j < nxx-4) && (k >= 4) && (k < nyy-4)) 
    {
        if ((T[index] < (float)(tId + tlag)*dt))
        {
            float b  = B[bsi + bsj + bsk];
            float bi = B[ip1 + bsj + bsk];
            float bj = B[bsi + jp1 + bsk];
            float bk = B[bsi + bsj + kp1];

            float dTxx_dx = (FDM1*(Txx[bsi + jm3 + bsk] - Txx[bsi + jp4 + bsk]) +
                             FDM2*(Txx[bsi + jp3 + bsk] - Txx[bsi + jm2 + bsk]) +
                             FDM3*(Txx[bsi + jm1 + bsk] - Txx[bsi + jp2 + bsk]) +
                             FDM4*(Txx[bsi + jp1 + bsk] - Txx[bsi + bsj + bsk])) * inv_dx;

            float dTxy_dy = (FDM1*(Txy[bsi + bsj + km4] - Txy[bsi + bsj + kp3]) +
                             FDM2*(Txy[bsi + bsj + kp2] - Txy[bsi + bsj + km3]) +
                             FDM3*(Txy[bsi + bsj + km2] - Txy[bsi + bsj + kp1]) +
                             FDM4*(Txy[bsi + bsj + bsk] - Txy[bsi + bsj + km1])) * inv_dy;

            float dTxz_dz = (FDM1*(Txz[im4 + bsj + bsk] - Txz[ip3 + bsj + bsk]) +
                             FDM2*(Txz[ip2 + bsj + bsk] - Txz[im3 + bsj + bsk]) +
                             FDM3*(Txz[im2 + bsj + bsk] - Txz[ip1 + bsj + bsk]) +
                             FDM4*(Txz[bsi + bsj + bsk] - Txz[im1 + bsj + bsk])) * inv_dz;

            float dTxy_dx = (FDM1*(Txy[bsi + jm4 + bsk] - Txy[bsi + jp3 + bsk]) +
                             FDM2*(Txy[bsi + jp2 + bsk] - Txy[bsi + jm3 + bsk]) +
                             FDM3*(Txy[bsi + jm2 + bsk] - Txy[bsi + jp1 + bsk]) +
                             FDM4*(Txy[bsi + bsj + bsk] - Txy[bsi + jm1 + bsk])) * inv_dx;

            float dTyy_dy = (FDM1*(Tyy[bsi + bsj + km3] - Tyy[bsi + bsj + kp4]) +
                             FDM2*(Tyy[bsi + bsj + kp3] - Tyy[bsi + bsj + km2]) +
                             FDM3*(Tyy[bsi + bsj + km1] - Tyy[bsi + bsj + kp2]) +
                             FDM4*(Tyy[bsi + bsj + kp1] - Tyy[bsi + bsj + bsk])) * inv_dy;

            float dTyz_dz = (FDM1*(Tyz[im4 + bsj + bsk] - Tyz[ip3 + bsj + bsk]) +
                             FDM2*(Tyz[ip2 + bsj + bsk] - Tyz[im3 + bsj + bsk]) +
                             FDM3*(Tyz[im2 + bsj + bsk] - Tyz[ip1 + bsj + bsk]) +
                             FDM4*(Tyz[bsi + bsj + bsk] - Tyz[im1 + bsj + bsk])) * inv_dz;

            float dTxz_dx = (FDM1*(Txz[bsi + jm4 + bsk] - Txz[bsi + jp3 + bsk]) +
                             FDM2*(Txz[bsi + jp2 + bsk] - Txz[bsi + jm3 + bsk]) +
                             FDM3*(Txz[bsi + jm2 + bsk] - Txz[bsi + jp1 + bsk]) +
                             FDM4*(Txz[bsi + bsj + bsk] - Txz[bsi + jm1 + bsk])) * inv_dx;

            float dTyz_dy = (FDM1*(Tyz[bsi + bsj + km4] - Tyz[bsi + bsj + kp3]) +
                             FDM2*(Tyz[bsi + bsj + kp2] - Tyz[bsi + bsj + km3]) +
                             FDM3*(Tyz[bsi + bsj + km2] - Tyz[bsi + bsj + kp1]) +
                             FDM4*(Tyz[bsi + bsj + bsk] - Tyz[bsi + bsj + km1])) * inv_dy;

            float dTzz_dz = (FDM1*(Tzz[im3 + bsj + bsk] - Tzz[ip4 + bsj + bsk]) +
                             FDM2*(Tzz[ip3 + bsj + bsk] - Tzz[im2 + bsj + bsk]) +
                             FDM3*(Tzz[im1 + bsj + bsk] - Tzz[ip2 + bsj + bsk]) +
                             FDM4*(Tzz[ip1 + bsj + bsk] - Tzz[bsi + bsj + bsk])) * inv_dz;

            float Bx = 0.5f*(b + bj);
            float By = 0.5f*(b + bk);
            float Bz = 0.5f*(b + bi);

            Vx[index] += dt*Bx*(dTxx_dx + dTxy_dy + dTxz_dz); 
            Vy[index] += dt*By*(dTxy_dx + dTyy_dy + dTyz_dz); 
            Vz[index] += dt*Bz*(dTxz_dx + dTyz_dy + dTzz_dz);
        }

        float damper = get_boundary_damper(damp1D, damp2D, damp3D, i, j, k, nxx, nyy, nzz, nb);

        Vx[index] *= damper;
        Vy[index] *= damper;
        Vz[index] *= damper;

        Txx[index] *= damper;
        Tyy[index] *= damper;
        Tzz[index] *= damper;
        Txz[index] *= damper;
        Tyz[index] *= damper;
        Txy[index] *= damper;
    }
}

__global__ void uintc_compute_pressure_issg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, 
                                            uintc * C11, uintc * C12, uintc * C13, uintc * C14, uintc * C15, uintc * C16, uintc * C22, uintc * C23, uintc * C24, uintc * C25, uintc * C26, 
                                            uintc * C33, uintc * C34, uintc * C35, uintc * C36, uintc * C44, uintc * C45, uintc * C46, uintc * C55, uintc * C56, uintc * C66, int tId, 
                                            int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz, float minC11, float maxC11, float minC12, float maxC12, 
                                            float minC13, float maxC13, float minC14, float maxC14, float minC15, float maxC15, float minC16, float maxC16, float minC22, float maxC22, 
                                            float minC23, float maxC23, float minC24, float maxC24, float minC25, float maxC25, float minC26, float maxC26, float minC33, float maxC33, 
                                            float minC34, float maxC34, float minC35, float maxC35, float minC36, float maxC36, float minC44, float maxC44, float minC45, float maxC45, 
                                            float minC46, float maxC46, float minC55, float maxC55, float minC56, float maxC56, float minC66, float maxC66, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    const float FDM1 = 6.97545e-4f; 
    const float FDM2 = 9.57031e-3f; 
    const float FDM3 = 7.97526e-2f; 
    const float FDM4 = 1.19628906f;     

    const float CFDM1 = 0.016666666f;
    const float CFDM2 = 0.150000000f;
    const float CFDM3 = 0.750000000f;

    const float inv_dx = 1.0f / dx;
    const float inv_dy = 1.0f / dy;
    const float inv_dz = 1.0f / dz;

    const size_t nxx_nzz = nxx*nzz;

    int k = (int) (index / nxx_nzz);         
    int j = (int) (index - k*nxx_nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx_nzz); 
    
    size_t bsi = i, bsj = j*nzz, bsk = k*nxx_nzz;  

    size_t ip1 = i+1, ip2 = i+2, ip3 = i+3, ip4 = i+4;
    size_t im1 = i-1, im2 = i-2, im3 = i-3, im4 = i-4;

    size_t jp1 = (j+1)*nzz, jp2 = (j+2)*nzz, jp3 = (j+3)*nzz, jp4 = (j+4)*nzz;
    size_t jm1 = (j-1)*nzz, jm2 = (j-2)*nzz, jm3 = (j-3)*nzz, jm4 = (j-4)*nzz;

    size_t kp1 = (k+1)*nxx_nzz, kp2 = (k+2)*nxx_nzz, kp3 = (k+3)*nxx_nzz, kp4 = (k+4)*nxx_nzz;
    size_t km1 = (k-1)*nxx_nzz, km2 = (k-2)*nxx_nzz, km3 = (k-3)*nxx_nzz, km4 = (k-4)*nxx_nzz;

    float dVx_dx, dVx_dx1, dVx_dx2, dVx_dx3, dVx_dx4;
    float dVx_dy, dVx_dy1, dVx_dy2, dVx_dy3, dVx_dy4;
    float dVx_dz, dVx_dz1, dVx_dz2, dVx_dz3, dVx_dz4;

    float dVy_dx, dVy_dx1, dVy_dx2, dVy_dx3, dVy_dx4;
    float dVy_dy, dVy_dy1, dVy_dy2, dVy_dy3, dVy_dy4;
    float dVy_dz, dVy_dz1, dVy_dz2, dVy_dz3, dVy_dz4;

    float dVz_dx, dVz_dx1, dVz_dx2, dVz_dx3, dVz_dx4;
    float dVz_dy, dVz_dy1, dVz_dy2, dVz_dy3, dVz_dy4;
    float dVz_dz, dVz_dz1, dVz_dz2, dVz_dz3, dVz_dz4;

    if((i >= 4) && (i < nzz-4) && (j >= 4) && (j < nxx-4) && (k >= 4) && (k < nyy-4)) 
    {
        T[index] = (eikonal) ? T[index] : 0.0f;

        if ((T[index] < (float)(tId + tlag)*dt))
        {
            float c11 = (minC11 + (static_cast<float>(C11[index]) - 1.0f) * (maxC11 - minC11) / (COMPRESS - 1));
            float c12 = (minC12 + (static_cast<float>(C12[index]) - 1.0f) * (maxC12 - minC12) / (COMPRESS - 1));
            float c13 = (minC13 + (static_cast<float>(C13[index]) - 1.0f) * (maxC13 - minC13) / (COMPRESS - 1));
            float c14 = (minC14 + (static_cast<float>(C14[index]) - 1.0f) * (maxC14 - minC14) / (COMPRESS - 1));
            float c15 = (minC15 + (static_cast<float>(C15[index]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            float c16 = (minC16 + (static_cast<float>(C16[index]) - 1.0f) * (maxC16 - minC16) / (COMPRESS - 1));

            float c22 = (minC22 + (static_cast<float>(C22[index]) - 1.0f) * (maxC22 - minC22) / (COMPRESS - 1));
            float c23 = (minC23 + (static_cast<float>(C23[index]) - 1.0f) * (maxC23 - minC23) / (COMPRESS - 1));
            float c24 = (minC24 + (static_cast<float>(C24[index]) - 1.0f) * (maxC24 - minC24) / (COMPRESS - 1));
            float c25 = (minC25 + (static_cast<float>(C25[index]) - 1.0f) * (maxC25 - minC25) / (COMPRESS - 1));
            float c26 = (minC26 + (static_cast<float>(C26[index]) - 1.0f) * (maxC26 - minC26) / (COMPRESS - 1));

            float c33 = (minC33 + (static_cast<float>(C33[index]) - 1.0f) * (maxC33 - minC33) / (COMPRESS - 1));
            float c34 = (minC34 + (static_cast<float>(C34[index]) - 1.0f) * (maxC34 - minC34) / (COMPRESS - 1));
            float c35 = (minC35 + (static_cast<float>(C35[index]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));
            float c36 = (minC36 + (static_cast<float>(C36[index]) - 1.0f) * (maxC36 - minC36) / (COMPRESS - 1));

            float c44 = (minC44 + (static_cast<float>(C44[index]) - 1.0f) * (maxC44 - minC44) / (COMPRESS - 1));
            float c45 = (minC45 + (static_cast<float>(C45[index]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));
            float c46 = (minC46 + (static_cast<float>(C46[index]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));

            float c55 = (minC55 + (static_cast<float>(C55[index]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            float c56 = (minC56 + (static_cast<float>(C56[index]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));

            float c66 = (minC66 + (static_cast<float>(C66[index]) - 1.0f) * (maxC66 - minC66) / (COMPRESS - 1));

            float aux_Txx = Txx[index], aux_Txz = Txz[index];            
            float aux_Tyy = Tyy[index], aux_Tyz = Tyz[index];            
            float aux_Tzz = Tzz[index], aux_Txy = Txy[index];            

            // dVx_dx ---------------------------------------------------------------------

            dVx_dx = (FDM1*(Vx[bsi + jm4 + bsk] - Vx[bsi + jp3 + bsk]) +
                      FDM2*(Vx[bsi + jp2 + bsk] - Vx[bsi + jm3 + bsk]) +
                      FDM3*(Vx[bsi + jm2 + bsk] - Vx[bsi + jp1 + bsk]) +
                      FDM4*(Vx[bsi + bsj + bsk] - Vx[bsi + jm1 + bsk])) * inv_dx;
 
            // dVx_dy ---------------------------------------------------------------------

            dVx_dy1 = (CFDM1*(Vx[bsi + bsj + kp3] - Vx[bsi + bsj + km3]) +
                       CFDM2*(Vx[bsi + bsj + km2] - Vx[bsi + bsj + kp2]) +
                       CFDM3*(Vx[bsi + bsj + kp1] - Vx[bsi + bsj + km1])) * inv_dy; 
            
            dVx_dy2 = (CFDM1*(Vx[bsi + jm1 + kp3] - Vx[bsi + jm1 + km3]) +
                       CFDM2*(Vx[bsi + jm1 + km2] - Vx[bsi + jm1 + kp2]) +
                       CFDM3*(Vx[bsi + jm1 + kp1] - Vx[bsi + jm1 + km1])) * inv_dy; 
            
            dVx_dy = 0.5f*(dVx_dy1 + dVx_dy2);

            // dVx_dz ---------------------------------------------------------------------

            dVx_dz1 = (CFDM1*(Vx[ip3 + bsj + bsk] - Vx[im3 + bsj + bsk]) +
                       CFDM2*(Vx[im2 + bsj + bsk] - Vx[ip2 + bsj + bsk]) +
                       CFDM3*(Vx[ip1 + bsj + bsk] - Vx[im1 + bsj + bsk])) * inv_dz; 
            
            dVx_dz2 = (CFDM1*(Vx[ip3 + jm1 + bsk] - Vx[im3 + jm1 + bsk]) +
                       CFDM2*(Vx[im2 + jm1 + bsk] - Vx[ip2 + jm1 + bsk]) +
                       CFDM3*(Vx[ip1 + jm1 + bsk] - Vx[im1 + jm1 + bsk])) * inv_dz; 
            
            dVx_dz = 0.5f*(dVx_dz1 + dVx_dz2);

            // dVy_dx ---------------------------------------------------------------------

            dVy_dx1 = (CFDM1*(Vy[bsi + jp3 + bsk] - Vy[bsi + jm3 + bsk]) +
                       CFDM2*(Vy[bsi + jm2 + bsk] - Vy[bsi + jp2 + bsk]) +
                       CFDM3*(Vy[bsi + jp1 + bsk] - Vy[bsi + jm1 + bsk])) * inv_dx; 
            
            dVy_dx2 = (CFDM1*(Vy[bsi + jp3 + km1] - Vy[bsi + jm3 + km1]) +
                       CFDM2*(Vy[bsi + jm2 + km1] - Vy[bsi + jp2 + km1]) +
                       CFDM3*(Vy[bsi + jp1 + km1] - Vy[bsi + jm1 + km1])) * inv_dx; 
            
            dVy_dx = 0.5f*(dVy_dx1 + dVy_dx2);

            // dVy_dy ---------------------------------------------------------------------

            dVy_dy = (FDM1*(Vy[bsi + bsj + km4] - Vy[bsi + bsj + kp3]) +
                      FDM2*(Vy[bsi + bsj + kp2] - Vy[bsi + bsj + km3]) +
                      FDM3*(Vy[bsi + bsj + km2] - Vy[bsi + bsj + kp1]) +
                      FDM4*(Vy[bsi + bsj + bsk] - Vy[bsi + bsj + km1])) * inv_dy;

            // dVy_dz ---------------------------------------------------------------------

            dVy_dz1 = (CFDM1*(Vy[ip3 + bsj + bsk] - Vy[im3 + bsj + bsk]) +
                       CFDM2*(Vy[im2 + bsj + bsk] - Vy[ip2 + bsj + bsk]) +
                       CFDM3*(Vy[ip1 + bsj + bsk] - Vy[im1 + bsj + bsk])) * inv_dz; 
            
            dVy_dz2 = (CFDM1*(Vy[ip3 + bsj + km1] - Vy[im3 + bsj + km1]) +
                       CFDM2*(Vy[im2 + bsj + km1] - Vy[ip2 + bsj + km1]) +
                       CFDM3*(Vy[ip1 + bsj + km1] - Vy[im1 + bsj + km1])) * inv_dz; 

            dVy_dz = 0.5f*(dVy_dz1 + dVy_dz2);

            // dVz_dx ---------------------------------------------------------------------

            dVz_dx1 = (CFDM1*(Vz[bsi + jp3 + bsk] - Vz[bsi + jm3 + bsk]) +
                       CFDM2*(Vz[bsi + jm2 + bsk] - Vz[bsi + jp2 + bsk]) +
                       CFDM3*(Vz[bsi + jp1 + bsk] - Vz[bsi + jm1 + bsk])) * inv_dx; 
            
            dVz_dx2 = (CFDM1*(Vz[im1 + jp3 + bsk] - Vz[im1 + jm3 + bsk]) +
                       CFDM2*(Vz[im1 + jm2 + bsk] - Vz[im1 + jp2 + bsk]) +
                       CFDM3*(Vz[im1 + jp1 + bsk] - Vz[im1 + jm1 + bsk])) * inv_dx; 

            dVz_dx = 0.5f*(dVz_dx1 + dVz_dx2);

            // dVz_dy ---------------------------------------------------------------------

            dVz_dy1 = (CFDM1*(Vz[bsi + bsj + kp3] - Vz[bsi + bsj + km3]) +
                       CFDM2*(Vz[bsi + bsj + km2] - Vz[bsi + bsj + kp2]) +
                       CFDM3*(Vz[bsi + bsj + kp1] - Vz[bsi + bsj + km1])) * inv_dy; 
            
            dVz_dy2 = (CFDM1*(Vz[im1 + bsj + kp3] - Vz[im1 + bsj + km3]) +
                       CFDM2*(Vz[im1 + bsj + km2] - Vz[im1 + bsj + kp2]) +
                       CFDM3*(Vz[im1 + bsj + kp1] - Vz[im1 + bsj + km1])) * inv_dy; 

            dVz_dy = 0.5f*(dVz_dy1 + dVz_dy2);

            // dVz_dz ---------------------------------------------------------------------

            dVz_dz = (FDM1*(Vz[im4 + bsj + bsk] - Vz[ip3 + bsj + bsk]) +
                      FDM2*(Vz[ip2 + bsj + bsk] - Vz[im3 + bsj + bsk]) +
                      FDM3*(Vz[im2 + bsj + bsk] - Vz[ip1 + bsj + bsk]) +
                      FDM4*(Vz[bsi + bsj + bsk] - Vz[im1 + bsj + bsk])) * inv_dz;

            // Equation ---------------------------------------------------------------------

            aux_Txx += dt*(c11*dVx_dx + c16*dVx_dy + c15*dVx_dz +
                           c16*dVy_dx + c12*dVy_dy + c14*dVy_dz +
                           c15*dVz_dx + c14*dVz_dy + c13*dVz_dz);                    
        
            aux_Tyy += dt*(c12*dVx_dx + c26*dVx_dy + c25*dVx_dz +
                           c26*dVy_dx + c22*dVy_dy + c24*dVy_dz +
                           c25*dVz_dx + c24*dVz_dy + c23*dVz_dz);                    
        
            aux_Tzz += dt*(c13*dVx_dx + c36*dVx_dy + c35*dVx_dz +
                           c36*dVy_dx + c23*dVy_dy + c34*dVy_dz +
                           c35*dVz_dx + c34*dVz_dy + c33*dVz_dz);  

            // dVx_dx ---------------------------------------------------------------------

            dVx_dx1 = (CFDM1*(Vx[bsi + jp3 + bsk] - Vx[bsi + jm3 + bsk]) +
                       CFDM2*(Vx[bsi + jm2 + bsk] - Vx[bsi + jp2 + bsk]) +
                       CFDM3*(Vx[bsi + jp1 + bsk] - Vx[bsi + jm1 + bsk])) * inv_dx;
                            
            dVx_dx2 = (CFDM1*(Vx[bsi + jp3 + kp1] - Vx[bsi + jm3 + kp1]) +
                       CFDM2*(Vx[bsi + jm2 + kp1] - Vx[bsi + jp2 + kp1]) +
                       CFDM3*(Vx[bsi + jp1 + kp1] - Vx[bsi + jm1 + kp1])) * inv_dx;

            dVx_dx = 0.5f*(dVx_dx1 + dVx_dx2);

            // dVx_dy ---------------------------------------------------------------------

            dVx_dy = (FDM1*(Vx[bsi + bsj + km3] - Vx[bsi + bsj + kp4]) +
                      FDM2*(Vx[bsi + bsj + kp3] - Vx[bsi + bsj + km2]) +
                      FDM3*(Vx[bsi + bsj + km1] - Vx[bsi + bsj + kp2]) +
                      FDM4*(Vx[bsi + bsj + kp1] - Vx[bsi + bsj + bsk])) * inv_dy;

            // dVx_dz ---------------------------------------------------------------------

            dVx_dz1 = (CFDM1*(Vx[ip3 + bsj + bsk] - Vx[im3 + bsj + bsk]) +
                       CFDM2*(Vx[im2 + bsj + bsk] - Vx[ip2 + bsj + bsk]) +
                       CFDM3*(Vx[ip1 + bsj + bsk] - Vx[im1 + bsj + bsk])) * inv_dz;

            dVx_dz2 = (CFDM1*(Vx[ip3 + bsj + kp1] - Vx[im3 + bsj + kp1]) +
                       CFDM2*(Vx[im2 + bsj + kp1] - Vx[ip2 + bsj + kp1]) +
                       CFDM3*(Vx[ip1 + bsj + kp1] - Vx[im1 + bsj + kp1])) * inv_dz;

            dVx_dz = 0.5f*(dVx_dz1 + dVx_dz2);

            // dVy_dx ---------------------------------------------------------------------

            dVy_dx = (FDM1*(Vy[bsi + jm3 + bsk] - Vy[bsi + jp4 + bsk]) +
                      FDM2*(Vy[bsi + jp3 + bsk] - Vy[bsi + jm2 + bsk]) +
                      FDM3*(Vy[bsi + jm1 + bsk] - Vy[bsi + jp2 + bsk]) +
                      FDM4*(Vy[bsi + jp1 + bsk] - Vy[bsi + bsj + bsk])) * inv_dx;

            // dVy_dy ---------------------------------------------------------------------

            dVy_dy1 = (CFDM1*(Vy[bsi + bsj + kp3] - Vy[bsi + bsj + km3]) +
                       CFDM2*(Vy[bsi + bsj + km2] - Vy[bsi + bsj + kp2]) +
                       CFDM3*(Vy[bsi + bsj + kp1] - Vy[bsi + bsj + km1])) * inv_dy;

            dVy_dy2 = (CFDM1*(Vy[bsi + jp1 + kp3] - Vy[bsi + jp1 + km3]) +
                       CFDM2*(Vy[bsi + jp1 + km2] - Vy[bsi + jp1 + kp2]) +
                       CFDM3*(Vy[bsi + jp1 + kp1] - Vy[bsi + jp1 + km1])) * inv_dy;

            dVy_dy = 0.5f*(dVy_dy1 + dVy_dy2);

            // dVy_dz ---------------------------------------------------------------------

            dVy_dz1 = (CFDM1*(Vy[ip3 + bsj + bsk] - Vy[im3 + bsj + bsk]) +
                       CFDM2*(Vy[im2 + bsj + bsk] - Vy[ip2 + bsj + bsk]) +
                       CFDM3*(Vy[ip1 + bsj + bsk] - Vy[im1 + bsj + bsk])) * inv_dz;

            dVy_dz2 = (CFDM1*(Vy[ip3 + bsj + kp1] - Vy[im3 + bsj + kp1]) +
                       CFDM2*(Vy[im2 + bsj + kp1] - Vy[ip2 + bsj + kp1]) +
                       CFDM3*(Vy[ip1 + bsj + kp1] - Vy[im1 + bsj + kp1])) * inv_dz;

            dVy_dz = 0.5f*(dVy_dz1 + dVy_dz2);

            // dVz_dx ---------------------------------------------------------------------

            dVz_dx1 = (FDM1*(Vz[bsi + jm3 + bsk] - Vz[bsi + jp4 + bsk]) +
                       FDM2*(Vz[bsi + jp3 + bsk] - Vz[bsi + jm2 + bsk]) +
                       FDM3*(Vz[bsi + jm1 + bsk] - Vz[bsi + jp2 + bsk]) +
                       FDM4*(Vz[bsi + jp1 + bsk] - Vz[bsi + bsj + bsk])) * inv_dx;

            dVz_dx2 = (FDM1*(Vz[im1 + jm3 + bsk] - Vz[im1 + jp4 + bsk]) +
                       FDM2*(Vz[im1 + jp3 + bsk] - Vz[im1 + jm2 + bsk]) +
                       FDM3*(Vz[im1 + jm1 + bsk] - Vz[im1 + jp2 + bsk]) +
                       FDM4*(Vz[im1 + jp1 + bsk] - Vz[im1 + bsj + bsk])) * inv_dx;

            dVz_dx3 = (FDM1*(Vz[bsi + jm3 + kp1] - Vz[bsi + jp4 + kp1]) +
                       FDM2*(Vz[bsi + jp3 + kp1] - Vz[bsi + jm2 + kp1]) +
                       FDM3*(Vz[bsi + jm1 + kp1] - Vz[bsi + jp2 + kp1]) +
                       FDM4*(Vz[bsi + jp1 + kp1] - Vz[bsi + bsj + kp1])) * inv_dx;

            dVz_dx4 = (FDM1*(Vz[im1 + jm3 + kp1] - Vz[im1 + jp4 + kp1]) +
                       FDM2*(Vz[im1 + jp3 + kp1] - Vz[im1 + jm2 + kp1]) +
                       FDM3*(Vz[im1 + jm1 + kp1] - Vz[im1 + jp2 + kp1]) +
                       FDM4*(Vz[im1 + jp1 + kp1] - Vz[im1 + bsj + kp1])) * inv_dx;    

            dVz_dx = 0.25f*(dVz_dx1 + dVz_dx2 + dVz_dx3 + dVz_dx4);    

            // dVz_dy ---------------------------------------------------------------------

            dVz_dy1 = (FDM1*(Vz[bsi + bsj + km3] - Vz[bsi + bsj + kp4]) +
                       FDM2*(Vz[bsi + bsj + kp3] - Vz[bsi + bsj + km2]) +
                       FDM3*(Vz[bsi + bsj + km1] - Vz[bsi + bsj + kp2]) +
                       FDM4*(Vz[bsi + bsj + kp1] - Vz[bsi + bsj + bsk])) * inv_dy;

            dVz_dy2 = (FDM1*(Vz[im1 + bsj + km3] - Vz[im1 + bsj + kp4]) +
                       FDM2*(Vz[im1 + bsj + kp3] - Vz[im1 + bsj + km2]) +
                       FDM3*(Vz[im1 + bsj + km1] - Vz[im1 + bsj + kp2]) +
                       FDM4*(Vz[im1 + bsj + kp1] - Vz[im1 + bsj + bsk])) * inv_dy;

            dVz_dy3 = (FDM1*(Vz[bsi + jp1 + km3] - Vz[bsi + jp1 + kp4]) +
                       FDM2*(Vz[bsi + jp1 + kp3] - Vz[bsi + jp1 + km2]) +
                       FDM3*(Vz[bsi + jp1 + km1] - Vz[bsi + jp1 + kp2]) +
                       FDM4*(Vz[bsi + jp1 + kp1] - Vz[bsi + jp1 + bsk])) * inv_dy;    

            dVz_dy4 = (FDM1*(Vz[im1 + jp1 + km3] - Vz[im1 + jp1 + kp4]) +
                       FDM2*(Vz[im1 + jp1 + kp3] - Vz[im1 + jp1 + km2]) +
                       FDM3*(Vz[im1 + jp1 + km1] - Vz[im1 + jp1 + kp2]) +
                       FDM4*(Vz[im1 + jp1 + kp1] - Vz[im1 + jp1 + bsk])) * inv_dy;    

            dVz_dy = 0.25f*(dVz_dy1 + dVz_dy2 + dVz_dy3 + dVz_dy4);    

            // dVz_dz ---------------------------------------------------------------------
            
            dVz_dz1 = (FDM1*(Vz[im4 + bsj + bsk] - Vz[ip3 + bsj + bsk]) +
                       FDM2*(Vz[ip2 + bsj + bsk] - Vz[im3 + bsj + bsk]) +
                       FDM3*(Vz[im2 + bsj + bsk] - Vz[ip1 + bsj + bsk]) +
                       FDM4*(Vz[bsi + bsj + bsk] - Vz[im1 + bsj + bsk])) * inv_dz;    

            dVz_dz2 = (FDM1*(Vz[im4 + jp1 + bsk] - Vz[ip3 + jp1 + bsk]) +
                       FDM2*(Vz[ip2 + jp1 + bsk] - Vz[im3 + jp1 + bsk]) +
                       FDM3*(Vz[im2 + jp1 + bsk] - Vz[ip1 + jp1 + bsk]) +
                       FDM4*(Vz[bsi + jp1 + bsk] - Vz[im1 + jp1 + bsk])) * inv_dz;    

            dVz_dz3 = (FDM1*(Vz[im4 + bsj + kp1] - Vz[ip3 + jp1 + kp1]) +
                       FDM2*(Vz[ip2 + bsj + kp1] - Vz[im3 + jp1 + kp1]) +
                       FDM3*(Vz[im2 + bsj + kp1] - Vz[ip1 + jp1 + kp1]) +
                       FDM4*(Vz[bsi + bsj + kp1] - Vz[im1 + jp1 + kp1])) * inv_dz;    

            dVz_dz4 = (FDM1*(Vz[im4 + jp1 + kp1] - Vz[ip3 + jp1 + kp1]) +
                       FDM2*(Vz[ip2 + jp1 + kp1] - Vz[im3 + jp1 + kp1]) +
                       FDM3*(Vz[im2 + jp1 + kp1] - Vz[ip1 + jp1 + kp1]) +
                       FDM4*(Vz[bsi + jp1 + kp1] - Vz[im1 + jp1 + kp1])) * inv_dz;    

            dVz_dz = 0.25f*(dVz_dz1 + dVz_dz2 + dVz_dz3 + dVz_dz4);    
 
            // Equation ---------------------------------------------------------------------

            aux_Txy += dt*(c16*dVx_dx + c66*dVx_dy + c56*dVx_dz +
                           c66*dVy_dx + c26*dVy_dy + c46*dVy_dz +
                           c56*dVz_dx + c46*dVz_dy + c36*dVz_dz);                    

            // dVx_dx ---------------------------------------------------------------------
            
            dVx_dx1 = (CFDM1*(Vx[bsi + jp3 + bsk] - Vx[bsi + jm3 + bsk]) +
                       CFDM2*(Vx[bsi + jm2 + bsk] - Vx[bsi + jp2 + bsk]) +
                       CFDM3*(Vx[bsi + jp1 + bsk] - Vx[bsi + jm1 + bsk])) * inv_dx;

            dVx_dx2 = (CFDM1*(Vx[ip1 + jp3 + bsk] - Vx[ip1 + jm3 + bsk]) +
                       CFDM2*(Vx[ip1 + jm2 + bsk] - Vx[ip1 + jp2 + bsk]) +
                       CFDM3*(Vx[ip1 + jp1 + bsk] - Vx[ip1 + jm1 + bsk])) * inv_dx;

            dVx_dx = 0.5f*(dVx_dx1 + dVx_dx2);

            // dVx_dy ---------------------------------------------------------------------

            dVx_dy1 = (CFDM1*(Vx[bsi + bsj + kp3] - Vx[bsi + bsj + km3]) +
                       CFDM2*(Vx[bsi + bsj + km2] - Vx[bsi + bsj + kp2]) +
                       CFDM3*(Vx[bsi + bsj + kp1] - Vx[bsi + bsj + km1])) * inv_dy;

            dVx_dy2 = (CFDM1*(Vx[ip1 + bsj + kp3] - Vx[ip1 + bsj + km3]) +
                       CFDM2*(Vx[ip1 + bsj + km2] - Vx[ip1 + bsj + kp2]) +
                       CFDM3*(Vx[ip1 + bsj + kp1] - Vx[ip1 + bsj + km1])) * inv_dy;

            dVx_dy = 0.5f*(dVx_dy1 + dVx_dy2);

            // dVx_dz ---------------------------------------------------------------------

            dVx_dz = (FDM1*(Vx[im3 + bsj + bsk] - Vx[ip4 + bsj + bsk]) +
                      FDM2*(Vx[ip3 + bsj + bsk] - Vx[im2 + bsj + bsk]) +
                      FDM3*(Vx[im1 + bsj + bsk] - Vx[ip2 + bsj + bsk]) +
                      FDM4*(Vx[ip1 + bsj + bsk] - Vx[bsi + bsj + bsk])) * inv_dz;

            // dVy_dx ---------------------------------------------------------------------

            dVy_dx1 = (FDM1*(Vy[bsi + jm3 + bsk] - Vy[bsi + jp4 + bsk]) +
                       FDM2*(Vy[bsi + jp3 + bsk] - Vy[bsi + jm2 + bsk]) +
                       FDM3*(Vy[bsi + jm1 + bsk] - Vy[bsi + jp2 + bsk]) +
                       FDM4*(Vy[bsi + jp1 + bsk] - Vy[bsi + bsj + bsk])) * inv_dx;

            dVy_dx2 = (FDM1*(Vy[bsi + jm3 + km1] - Vy[bsi + jp4 + km1]) +
                       FDM2*(Vy[bsi + jp3 + km1] - Vy[bsi + jm2 + km1]) +
                       FDM3*(Vy[bsi + jm1 + km1] - Vy[bsi + jp2 + km1]) +
                       FDM4*(Vy[bsi + jp1 + km1] - Vy[bsi + bsj + km1])) * inv_dx;

            dVy_dx3 = (FDM1*(Vy[ip1 + jm3 + bsk] - Vy[ip1 + jp4 + bsk]) +
                       FDM2*(Vy[ip1 + jp3 + bsk] - Vy[ip1 + jm2 + bsk]) +
                       FDM3*(Vy[ip1 + jm1 + bsk] - Vy[ip1 + jp2 + bsk]) +
                       FDM4*(Vy[ip1 + jp1 + bsk] - Vy[ip1 + bsj + bsk])) * inv_dx;

            dVy_dx4 = (FDM1*(Vy[ip1 + jm3 + km1] - Vy[ip1 + jp4 + km1]) +
                       FDM2*(Vy[ip1 + jp3 + km1] - Vy[ip1 + jm2 + km1]) +
                       FDM3*(Vy[ip1 + jm1 + km1] - Vy[ip1 + jp2 + km1]) +
                       FDM4*(Vy[ip1 + jp1 + km1] - Vy[ip1 + bsj + km1])) * inv_dx;

            dVy_dx = 0.25f*(dVy_dx1 + dVy_dx2 + dVy_dx3 + dVy_dx4);    

            // dVy_dy ---------------------------------------------------------------------

            dVy_dy1 = (FDM1*(Vy[bsi + bsj + km4] - Vy[bsi + bsj + kp3]) +
                       FDM2*(Vy[bsi + bsj + kp2] - Vy[bsi + bsj + km3]) +
                       FDM3*(Vy[bsi + bsj + km2] - Vy[bsi + bsj + kp1]) +
                       FDM4*(Vy[bsi + bsj + bsk] - Vy[bsi + bsj + km1])) * inv_dy;

            dVy_dy2 = (FDM1*(Vy[bsi + jp1 + km4] - Vy[bsi + jp1 + kp3]) +
                       FDM2*(Vy[bsi + jp1 + kp2] - Vy[bsi + jp1 + km3]) +
                       FDM3*(Vy[bsi + jp1 + km2] - Vy[bsi + jp1 + kp1]) +
                       FDM4*(Vy[bsi + jp1 + bsk] - Vy[bsi + jp1 + km1])) * inv_dy;

            dVy_dy3 = (FDM1*(Vy[ip1 + bsj + km4] - Vy[ip1 + bsj + kp3]) +
                       FDM2*(Vy[ip1 + bsj + kp2] - Vy[ip1 + bsj + km3]) +
                       FDM3*(Vy[ip1 + bsj + km2] - Vy[ip1 + bsj + kp1]) +
                       FDM4*(Vy[ip1 + bsj + bsk] - Vy[ip1 + bsj + km1])) * inv_dy;

            dVy_dy4 = (FDM1*(Vy[ip1 + jp1 + km4] - Vy[ip1 + jp1 + kp3]) +
                       FDM2*(Vy[ip1 + jp1 + kp2] - Vy[ip1 + jp1 + km3]) +
                       FDM3*(Vy[ip1 + jp1 + km2] - Vy[ip1 + jp1 + kp1]) +
                       FDM4*(Vy[ip1 + jp1 + bsk] - Vy[ip1 + jp1 + km1])) * inv_dy;

            dVy_dy = 0.25f*(dVy_dy1 + dVy_dy2 + dVy_dy3 + dVy_dy4);    

            // dVy_dz ---------------------------------------------------------------------

            dVy_dz1 = (FDM1*(Vy[im3 + bsj + bsk] - Vy[ip4 + bsj + bsk]) +
                       FDM2*(Vy[ip3 + bsj + bsk] - Vy[im2 + bsj + bsk]) +
                       FDM3*(Vy[im1 + bsj + bsk] - Vy[ip2 + bsj + bsk]) +
                       FDM4*(Vy[ip1 + bsj + bsk] - Vy[bsi + bsj + bsk])) * inv_dz;

            dVy_dz2 = (FDM1*(Vy[im3 + bsj + km1] - Vy[ip4 + bsj + km1]) +
                       FDM2*(Vy[ip3 + bsj + km1] - Vy[im2 + bsj + km1]) +
                       FDM3*(Vy[im1 + bsj + km1] - Vy[ip2 + bsj + km1]) +
                       FDM4*(Vy[ip1 + bsj + km1] - Vy[bsi + bsj + km1])) * inv_dz;

            dVy_dz3 = (FDM1*(Vy[im3 + jp1 + bsk] - Vy[ip4 + jp1 + bsk]) +
                       FDM2*(Vy[ip3 + jp1 + bsk] - Vy[im2 + jp1 + bsk]) +
                       FDM3*(Vy[im1 + jp1 + bsk] - Vy[ip2 + jp1 + bsk]) +
                       FDM4*(Vy[ip1 + jp1 + bsk] - Vy[bsi + jp1 + bsk])) * inv_dz;

            dVy_dz4 = (FDM1*(Vy[im3 + jp1 + km1] - Vy[ip4 + jp1 + km1]) +
                       FDM2*(Vy[ip3 + jp1 + km1] - Vy[im2 + jp1 + km1]) +
                       FDM3*(Vy[im1 + jp1 + km1] - Vy[ip2 + jp1 + km1]) +
                       FDM4*(Vy[ip1 + jp1 + km1] - Vy[bsi + jp1 + km1])) * inv_dz;

            dVy_dz = 0.25f*(dVy_dz1 + dVy_dz2 + dVy_dz3 + dVy_dz4);    

            // dVz_dx ---------------------------------------------------------------------

            dVz_dx = (FDM1*(Vz[bsi + jm3 + bsk] - Vz[bsi + jp4 + bsk]) +
                      FDM2*(Vz[bsi + jp3 + bsk] - Vz[bsi + jm2 + bsk]) +
                      FDM3*(Vz[bsi + jm1 + bsk] - Vz[bsi + jp2 + bsk]) +
                      FDM4*(Vz[bsi + jp1 + bsk] - Vz[bsi + bsj + bsk])) * inv_dx;

            // dVz_dy ---------------------------------------------------------------------

            dVz_dy1 = (CFDM1*(Vz[bsi + bsj + kp3] - Vz[bsi + bsj + km3]) +
                       CFDM2*(Vz[bsi + bsj + km2] - Vz[bsi + bsj + kp2]) +
                       CFDM3*(Vz[bsi + bsj + kp1] - Vz[bsi + bsj + km1])) * inv_dy;

            dVz_dy2 = (CFDM1*(Vz[bsi + jp1 + kp3] - Vz[bsi + jp1 + km3]) +
                       CFDM2*(Vz[bsi + jp1 + km2] - Vz[bsi + jp1 + kp2]) +
                       CFDM3*(Vz[bsi + jp1 + kp1] - Vz[bsi + jp1 + km1])) * inv_dy;

            dVz_dy = 0.5f*(dVz_dy1 + dVz_dy2);

            // dVz_dz ---------------------------------------------------------------------

            dVz_dz1 = (CFDM1*(Vz[ip3 + bsj + bsk] - Vz[im3 + bsj + bsk]) +
                       CFDM2*(Vz[im2 + bsj + bsk] - Vz[ip2 + bsj + bsk]) +
                       CFDM3*(Vz[ip1 + bsj + bsk] - Vz[im1 + bsj + bsk])) * inv_dz;

            dVz_dz2 = (CFDM1*(Vz[ip3 + jp1 + bsk] - Vz[im3 + jp1 + bsk]) +
                       CFDM2*(Vz[im2 + jp1 + bsk] - Vz[ip2 + jp1 + bsk]) +
                       CFDM3*(Vz[ip1 + jp1 + bsk] - Vz[im1 + jp1 + bsk])) * inv_dz;

            dVz_dz = 0.5f*(dVz_dz1 + dVz_dz2);

            // Equation ---------------------------------------------------------------------

            aux_Txz += dt*(c15*dVx_dx + c56*dVx_dy + c55*dVx_dz +
                           c56*dVy_dx + c25*dVy_dy + c45*dVy_dz +
                           c55*dVz_dx + c45*dVz_dy + c35*dVz_dz);                    

            // dVx_dx ---------------------------------------------------------------------
            
            dVx_dx1 = (FDM1*(Vx[bsi + jm4 + bsk] - Vx[bsi + jp3 + bsk]) +
                       FDM2*(Vx[bsi + jp2 + bsk] - Vx[bsi + jm3 + bsk]) +
                       FDM3*(Vx[bsi + jm2 + bsk] - Vx[bsi + jp1 + bsk]) +
                       FDM4*(Vx[bsi + bsj + bsk] - Vx[bsi + jm1 + bsk])) * inv_dx;

            dVx_dx2 = (FDM1*(Vx[bsi + jm4 + km1] - Vx[bsi + jp3 + km1]) +
                       FDM2*(Vx[bsi + jp2 + km1] - Vx[bsi + jm3 + km1]) +
                       FDM3*(Vx[bsi + jm2 + km1] - Vx[bsi + jp1 + km1]) +
                       FDM4*(Vx[bsi + bsj + km1] - Vx[bsi + jm1 + km1])) * inv_dx; 
        
            dVx_dx3 = (FDM1*(Vx[ip1 + jm4 + bsk] - Vx[ip1 + jp3 + bsk]) +
                       FDM2*(Vx[ip1 + jp2 + bsk] - Vx[ip1 + jm3 + bsk]) +
                       FDM3*(Vx[ip1 + jm2 + bsk] - Vx[ip1 + jp1 + bsk]) +
                       FDM4*(Vx[ip1 + bsj + bsk] - Vx[ip1 + jm1 + bsk])) * inv_dx; 
            
            dVx_dx4 = (FDM1*(Vx[ip1 + jm4 + km1] - Vx[ip1 + jp3 + km1]) +
                       FDM2*(Vx[ip1 + jp2 + km1] - Vx[ip1 + jm3 + km1]) +
                       FDM3*(Vx[ip1 + jm2 + km1] - Vx[ip1 + jp1 + km1]) +
                       FDM4*(Vx[ip1 + bsj + km1] - Vx[ip1 + jm1 + km1])) * inv_dx; 

            dVx_dx = 0.25f*(dVx_dx1 + dVx_dx2 + dVx_dx3 + dVx_dx4);

            // dVx_dy ---------------------------------------------------------------------

            dVx_dy1 = (FDM1*(Vx[bsi + bsj + km3] - Vx[bsi + bsj + kp4]) +
                       FDM2*(Vx[bsi + bsj + kp3] - Vx[bsi + bsj + km2]) +
                       FDM3*(Vx[bsi + bsj + km1] - Vx[bsi + bsj + kp2]) +
                       FDM4*(Vx[bsi + bsj + kp1] - Vx[bsi + bsj + bsk])) * inv_dy;

            dVx_dy2 = (FDM1*(Vx[bsi + jm1 + km3] - Vx[bsi + jm1 + kp4]) +
                       FDM2*(Vx[bsi + jm1 + kp3] - Vx[bsi + jm1 + km2]) +
                       FDM3*(Vx[bsi + jm1 + km1] - Vx[bsi + jm1 + kp2]) +
                       FDM4*(Vx[bsi + jm1 + kp1] - Vx[bsi + jm1 + bsk])) * inv_dy; 
        
            dVx_dy3 = (FDM1*(Vx[ip1 + bsj + km3] - Vx[ip1 + bsj + kp4]) +
                       FDM2*(Vx[ip1 + bsj + kp3] - Vx[ip1 + bsj + km2]) +
                       FDM3*(Vx[ip1 + bsj + km1] - Vx[ip1 + bsj + kp2]) +
                       FDM4*(Vx[ip1 + bsj + kp1] - Vx[ip1 + bsj + bsk])) * inv_dy; 
            
            dVx_dy4 = (FDM1*(Vx[ip1 + jm1 + km3] - Vx[ip1 + jm1 + kp4]) +
                       FDM2*(Vx[ip1 + jm1 + kp3] - Vx[ip1 + jm1 + km2]) +
                       FDM3*(Vx[ip1 + jm1 + km1] - Vx[ip1 + jm1 + kp2]) +
                       FDM4*(Vx[ip1 + jm1 + kp1] - Vx[ip1 + jm1 + bsk])) * inv_dy; 

            dVx_dy = 0.25f*(dVx_dy1 + dVx_dy2 + dVx_dy3 + dVx_dy4);

            // dVx_dz ---------------------------------------------------------------------

            dVx_dz1 = (FDM1*(Vx[im3 + bsj + bsk] - Vx[ip4 + bsj + bsk]) +
                       FDM2*(Vx[ip3 + bsj + bsk] - Vx[im2 + bsj + bsk]) +
                       FDM3*(Vx[im1 + bsj + bsk] - Vx[ip2 + bsj + bsk]) +
                       FDM4*(Vx[ip1 + bsj + bsk] - Vx[bsi + bsj + bsk])) * inv_dz;

            dVx_dz2 = (FDM1*(Vx[im3 + jm1 + bsk] - Vx[ip4 + jm1 + bsk]) +
                       FDM2*(Vx[ip3 + jm1 + bsk] - Vx[im2 + jm1 + bsk]) +
                       FDM3*(Vx[im1 + jm1 + bsk] - Vx[ip2 + jm1 + bsk]) +
                       FDM4*(Vx[ip1 + jm1 + bsk] - Vx[bsi + jm1 + bsk])) * inv_dz; 
        
            dVx_dz3 = (FDM1*(Vx[im3 + bsj + kp1] - Vx[ip4 + bsj + kp1]) +
                       FDM2*(Vx[ip3 + bsj + kp1] - Vx[im2 + bsj + kp1]) +
                       FDM3*(Vx[im1 + bsj + kp1] - Vx[ip2 + bsj + kp1]) +
                       FDM4*(Vx[ip1 + bsj + kp1] - Vx[bsi + bsj + kp1])) * inv_dz; 
            
            dVx_dz4 = (FDM1*(Vx[im3 + jm1 + kp1] - Vx[ip4 + jm1 + kp1]) +
                       FDM2*(Vx[ip3 + jm1 + kp1] - Vx[im2 + jm1 + kp1]) +
                       FDM3*(Vx[im1 + jm1 + kp1] - Vx[ip2 + jm1 + kp1]) +
                       FDM4*(Vx[ip1 + jm1 + kp1] - Vx[bsi + jm1 + kp1])) * inv_dz;  

            dVx_dz = 0.25f*(dVx_dz1 + dVx_dz2 + dVx_dz3 + dVx_dz4);

            // dVy_dx ---------------------------------------------------------------------

            dVy_dx1 = (CFDM1*(Vy[bsi + jp3 + bsk] - Vy[bsi + jm3 + bsk]) +
                       CFDM2*(Vy[bsi + jm2 + bsk] - Vy[bsi + jp2 + bsk]) +
                       CFDM3*(Vy[bsi + jp1 + bsk] - Vy[bsi + jm1 + bsk])) * inv_dx;

            dVy_dx2 = (CFDM1*(Vy[ip1 + jp3 + bsk] - Vy[ip1 + jm3 + bsk]) +
                       CFDM2*(Vy[ip1 + jm2 + bsk] - Vy[ip1 + jp2 + bsk]) +
                       CFDM3*(Vy[ip1 + jp1 + bsk] - Vy[ip1 + jm1 + bsk])) * inv_dx; 
            
            dVy_dx = 0.5f*(dVy_dx1 + dVy_dx2);

            // dVy_dy ---------------------------------------------------------------------

            dVy_dy1 = (CFDM1*(Vy[bsi + bsj + kp3] - Vy[bsi + bsj + km3]) +
                       CFDM2*(Vy[bsi + bsj + km2] - Vy[bsi + bsj + kp2]) +
                       CFDM3*(Vy[bsi + bsj + kp1] - Vy[bsi + bsj + km1])) * inv_dy;

            dVy_dy2 = (CFDM1*(Vy[ip1 + bsj + kp3] - Vy[ip1 + bsj + km3]) +
                       CFDM2*(Vy[ip1 + bsj + km2] - Vy[ip1 + bsj + kp2]) +
                       CFDM3*(Vy[ip1 + bsj + kp1] - Vy[ip1 + bsj + km1])) * inv_dy; 
            
            dVy_dy = 0.5f*(dVy_dy1 + dVy_dy2);

            // dVy_dz ---------------------------------------------------------------------

            dVy_dz = (FDM1*(Vy[im3 + bsj + bsk] - Vy[ip4 + bsj + bsk]) +
                      FDM2*(Vy[ip3 + bsj + bsk] - Vy[im2 + bsj + bsk]) +
                      FDM3*(Vy[im1 + bsj + bsk] - Vy[ip2 + bsj + bsk]) +
                      FDM4*(Vy[ip1 + bsj + bsk] - Vy[bsi + bsj + bsk])) * inv_dz;

            // dVz_dx ---------------------------------------------------------------------

            dVz_dx1 = (CFDM1*(Vz[bsi + jp3 + bsk] - Vz[bsi + jm3 + bsk]) +
                       CFDM2*(Vz[bsi + jm2 + bsk] - Vz[bsi + jp2 + bsk]) +
                       CFDM3*(Vz[bsi + jp1 + bsk] - Vz[bsi + jm1 + bsk])) * inv_dx;

            dVz_dx2 = (CFDM1*(Vz[bsi + jp3 + kp1] - Vz[bsi + jm3 + kp1]) +
                       CFDM2*(Vz[bsi + jm2 + kp1] - Vz[bsi + jp2 + kp1]) +
                       CFDM3*(Vz[bsi + jp1 + kp1] - Vz[bsi + jm1 + kp1])) * inv_dx;
            
            dVz_dx = 0.5f*(dVz_dx1 + dVz_dx2);

            // dVz_dy ---------------------------------------------------------------------

            dVz_dy = (FDM1*(Vz[bsi + bsj + km3] - Vz[bsi + bsj + kp4]) +
                      FDM2*(Vz[bsi + bsj + kp3] - Vz[bsi + bsj + km2]) +
                      FDM3*(Vz[bsi + bsj + km1] - Vz[bsi + bsj + kp2]) +
                      FDM4*(Vz[bsi + bsj + kp1] - Vz[bsi + bsj + bsk])) * inv_dy;

            // dVz_dz ---------------------------------------------------------------------

            dVz_dz1 = (CFDM1*(Vz[ip3 + bsj + bsk] - Vz[im3 + bsj + bsk]) +
                       CFDM2*(Vz[im2 + bsj + bsk] - Vz[ip2 + bsj + bsk]) +
                       CFDM3*(Vz[ip1 + bsj + bsk] - Vz[im1 + bsj + bsk])) * inv_dz;

            dVz_dz2 = (CFDM1*(Vz[ip3 + bsj + kp1] - Vz[im3 + bsj + kp1]) +
                       CFDM2*(Vz[im2 + bsj + kp1] - Vz[ip2 + bsj + kp1]) +
                       CFDM3*(Vz[ip1 + bsj + kp1] - Vz[im1 + bsj + kp1])) * inv_dz; 
            
            dVz_dz = 0.5f*(dVz_dz1 + dVz_dz2);

            // Equation ---------------------------------------------------------------------

            aux_Tyz += dt*(c14*dVx_dx + c46*dVx_dy + c45*dVx_dz +
                           c46*dVy_dx + c24*dVy_dy + c44*dVy_dz +
                           c45*dVz_dx + c44*dVz_dy + c34*dVz_dz); 
        
            P[index] = (aux_Txx + aux_Tyy + aux_Tzz) / 3.0f;

            Txx[index] = aux_Txx;
            Tyy[index] = aux_Tyy;
            Tzz[index] = aux_Tzz;
            Txy[index] = aux_Txy;
            Txz[index] = aux_Txz;
            Tyz[index] = aux_Tyz;
        }
    }
}

__global__ void float_compute_pressure_issg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, 
                                            float * C11, float * C12, float * C13, float * C14, float * C15, float * C16, float * C22, float * C23, float * C24, float * C25, float * C26, 
                                            float * C33, float * C34, float * C35, float * C36, float * C44, float * C45, float * C46, float * C55, float * C56, float * C66, int tId, 
                                            int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    const float FDM1 = 6.97545e-4f; 
    const float FDM2 = 9.57031e-3f; 
    const float FDM3 = 7.97526e-2f; 
    const float FDM4 = 1.19628906f;     

    const float CFDM1 = 0.016666666f;
    const float CFDM2 = 0.150000000f;
    const float CFDM3 = 0.750000000f;

    const float inv_dx = 1.0f / dx;
    const float inv_dy = 1.0f / dy;
    const float inv_dz = 1.0f / dz;

    const size_t nxx_nzz = nxx*nzz;

    int k = (int) (index / nxx_nzz);         
    int j = (int) (index - k*nxx_nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx_nzz); 
    
    size_t bsi = i, bsj = j*nzz, bsk = k*nxx_nzz;  

    size_t ip1 = i+1, ip2 = i+2, ip3 = i+3, ip4 = i+4;
    size_t im1 = i-1, im2 = i-2, im3 = i-3, im4 = i-4;

    size_t jp1 = (j+1)*nzz, jp2 = (j+2)*nzz, jp3 = (j+3)*nzz, jp4 = (j+4)*nzz;
    size_t jm1 = (j-1)*nzz, jm2 = (j-2)*nzz, jm3 = (j-3)*nzz, jm4 = (j-4)*nzz;

    size_t kp1 = (k+1)*nxx_nzz, kp2 = (k+2)*nxx_nzz, kp3 = (k+3)*nxx_nzz, kp4 = (k+4)*nxx_nzz;
    size_t km1 = (k-1)*nxx_nzz, km2 = (k-2)*nxx_nzz, km3 = (k-3)*nxx_nzz, km4 = (k-4)*nxx_nzz;

    float dVx_dx, dVx_dx1, dVx_dx2, dVx_dx3, dVx_dx4;
    float dVx_dy, dVx_dy1, dVx_dy2, dVx_dy3, dVx_dy4;
    float dVx_dz, dVx_dz1, dVx_dz2, dVx_dz3, dVx_dz4;

    float dVy_dx, dVy_dx1, dVy_dx2, dVy_dx3, dVy_dx4;
    float dVy_dy, dVy_dy1, dVy_dy2, dVy_dy3, dVy_dy4;
    float dVy_dz, dVy_dz1, dVy_dz2, dVy_dz3, dVy_dz4;

    float dVz_dx, dVz_dx1, dVz_dx2, dVz_dx3, dVz_dx4;
    float dVz_dy, dVz_dy1, dVz_dy2, dVz_dy3, dVz_dy4;
    float dVz_dz, dVz_dz1, dVz_dz2, dVz_dz3, dVz_dz4;

    T[index] = (eikonal) ? T[index] : 0.0f;

    if((i >= 4) && (i < nzz-4) && (j >= 4) && (j < nxx-4) && (k >= 4) && (k < nyy-4)) 
    {
        if ((T[index] < (float)(tId + tlag)*dt))
        {
            float c11 = C11[index];
            float c12 = C12[index];
            float c13 = C13[index];
            float c14 = C14[index];
            float c15 = C15[index];
            float c16 = C16[index];

            float c22 = C22[index];
            float c23 = C23[index];
            float c24 = C24[index];
            float c25 = C25[index];
            float c26 = C26[index];

            float c33 = C33[index];
            float c34 = C34[index];
            float c35 = C35[index];
            float c36 = C36[index];

            float c44 = C44[index];
            float c45 = C45[index];
            float c46 = C46[index];

            float c55 = C55[index];
            float c56 = C56[index];

            float c66 = C66[index];

            float aux_Txx = Txx[index], aux_Txz = Txz[index];            
            float aux_Tyy = Tyy[index], aux_Tyz = Tyz[index];            
            float aux_Tzz = Tzz[index], aux_Txy = Txy[index];            

            // dVx_dx ---------------------------------------------------------------------

            dVx_dx = (FDM1*(Vx[bsi + jm4 + bsk] - Vx[bsi + jp3 + bsk]) +
                      FDM2*(Vx[bsi + jp2 + bsk] - Vx[bsi + jm3 + bsk]) +
                      FDM3*(Vx[bsi + jm2 + bsk] - Vx[bsi + jp1 + bsk]) +
                      FDM4*(Vx[bsi + bsj + bsk] - Vx[bsi + jm1 + bsk])) * inv_dx;
 
            // dVx_dy ---------------------------------------------------------------------

            dVx_dy1 = (CFDM1*(Vx[bsi + bsj + kp3] - Vx[bsi + bsj + km3]) +
                       CFDM2*(Vx[bsi + bsj + km2] - Vx[bsi + bsj + kp2]) +
                       CFDM3*(Vx[bsi + bsj + kp1] - Vx[bsi + bsj + km1])) * inv_dy; 
            
            dVx_dy2 = (CFDM1*(Vx[bsi + jm1 + kp3] - Vx[bsi + jm1 + km3]) +
                       CFDM2*(Vx[bsi + jm1 + km2] - Vx[bsi + jm1 + kp2]) +
                       CFDM3*(Vx[bsi + jm1 + kp1] - Vx[bsi + jm1 + km1])) * inv_dy; 
            
            dVx_dy = 0.5f*(dVx_dy1 + dVx_dy2);

            // dVx_dz ---------------------------------------------------------------------

            dVx_dz1 = (CFDM1*(Vx[ip3 + bsj + bsk] - Vx[im3 + bsj + bsk]) +
                       CFDM2*(Vx[im2 + bsj + bsk] - Vx[ip2 + bsj + bsk]) +
                       CFDM3*(Vx[ip1 + bsj + bsk] - Vx[im1 + bsj + bsk])) * inv_dz; 
            
            dVx_dz2 = (CFDM1*(Vx[ip3 + jm1 + bsk] - Vx[im3 + jm1 + bsk]) +
                       CFDM2*(Vx[im2 + jm1 + bsk] - Vx[ip2 + jm1 + bsk]) +
                       CFDM3*(Vx[ip1 + jm1 + bsk] - Vx[im1 + jm1 + bsk])) * inv_dz; 
            
            dVx_dz = 0.5f*(dVx_dz1 + dVx_dz2);

            // dVy_dx ---------------------------------------------------------------------

            dVy_dx1 = (CFDM1*(Vy[bsi + jp3 + bsk] - Vy[bsi + jm3 + bsk]) +
                       CFDM2*(Vy[bsi + jm2 + bsk] - Vy[bsi + jp2 + bsk]) +
                       CFDM3*(Vy[bsi + jp1 + bsk] - Vy[bsi + jm1 + bsk])) * inv_dx; 
            
            dVy_dx2 = (CFDM1*(Vy[bsi + jp3 + km1] - Vy[bsi + jm3 + km1]) +
                       CFDM2*(Vy[bsi + jm2 + km1] - Vy[bsi + jp2 + km1]) +
                       CFDM3*(Vy[bsi + jp1 + km1] - Vy[bsi + jm1 + km1])) * inv_dx; 
            
            dVy_dx = 0.5f*(dVy_dx1 + dVy_dx2);

            // dVy_dy ---------------------------------------------------------------------

            dVy_dy = (FDM1*(Vy[bsi + bsj + km4] - Vy[bsi + bsj + kp3]) +
                      FDM2*(Vy[bsi + bsj + kp2] - Vy[bsi + bsj + km3]) +
                      FDM3*(Vy[bsi + bsj + km2] - Vy[bsi + bsj + kp1]) +
                      FDM4*(Vy[bsi + bsj + bsk] - Vy[bsi + bsj + km1])) * inv_dy;

            // dVy_dz ---------------------------------------------------------------------

            dVy_dz1 = (CFDM1*(Vy[ip3 + bsj + bsk] - Vy[im3 + bsj + bsk]) +
                       CFDM2*(Vy[im2 + bsj + bsk] - Vy[ip2 + bsj + bsk]) +
                       CFDM3*(Vy[ip1 + bsj + bsk] - Vy[im1 + bsj + bsk])) * inv_dz; 
            
            dVy_dz2 = (CFDM1*(Vy[ip3 + bsj + km1] - Vy[im3 + bsj + km1]) +
                       CFDM2*(Vy[im2 + bsj + km1] - Vy[ip2 + bsj + km1]) +
                       CFDM3*(Vy[ip1 + bsj + km1] - Vy[im1 + bsj + km1])) * inv_dz; 

            dVy_dz = 0.5f*(dVy_dz1 + dVy_dz2);

            // dVz_dx ---------------------------------------------------------------------

            dVz_dx1 = (CFDM1*(Vz[bsi + jp3 + bsk] - Vz[bsi + jm3 + bsk]) +
                       CFDM2*(Vz[bsi + jm2 + bsk] - Vz[bsi + jp2 + bsk]) +
                       CFDM3*(Vz[bsi + jp1 + bsk] - Vz[bsi + jm1 + bsk])) * inv_dx; 
            
            dVz_dx2 = (CFDM1*(Vz[im1 + jp3 + bsk] - Vz[im1 + jm3 + bsk]) +
                       CFDM2*(Vz[im1 + jm2 + bsk] - Vz[im1 + jp2 + bsk]) +
                       CFDM3*(Vz[im1 + jp1 + bsk] - Vz[im1 + jm1 + bsk])) * inv_dx; 

            dVz_dx = 0.5f*(dVz_dx1 + dVz_dx2);

            // dVz_dy ---------------------------------------------------------------------

            dVz_dy1 = (CFDM1*(Vz[bsi + bsj + kp3] - Vz[bsi + bsj + km3]) +
                       CFDM2*(Vz[bsi + bsj + km2] - Vz[bsi + bsj + kp2]) +
                       CFDM3*(Vz[bsi + bsj + kp1] - Vz[bsi + bsj + km1])) * inv_dy; 
            
            dVz_dy2 = (CFDM1*(Vz[im1 + bsj + kp3] - Vz[im1 + bsj + km3]) +
                       CFDM2*(Vz[im1 + bsj + km2] - Vz[im1 + bsj + kp2]) +
                       CFDM3*(Vz[im1 + bsj + kp1] - Vz[im1 + bsj + km1])) * inv_dy; 

            dVz_dy = 0.5f*(dVz_dy1 + dVz_dy2);

            // dVz_dz ---------------------------------------------------------------------

            dVz_dz = (FDM1*(Vz[im4 + bsj + bsk] - Vz[ip3 + bsj + bsk]) +
                      FDM2*(Vz[ip2 + bsj + bsk] - Vz[im3 + bsj + bsk]) +
                      FDM3*(Vz[im2 + bsj + bsk] - Vz[ip1 + bsj + bsk]) +
                      FDM4*(Vz[bsi + bsj + bsk] - Vz[im1 + bsj + bsk])) * inv_dz;

            // Equation ---------------------------------------------------------------------

            aux_Txx += dt*(c11*dVx_dx + c16*dVx_dy + c15*dVx_dz +
                           c16*dVy_dx + c12*dVy_dy + c14*dVy_dz +
                           c15*dVz_dx + c14*dVz_dy + c13*dVz_dz);                    
        
            aux_Tyy += dt*(c12*dVx_dx + c26*dVx_dy + c25*dVx_dz +
                           c26*dVy_dx + c22*dVy_dy + c24*dVy_dz +
                           c25*dVz_dx + c24*dVz_dy + c23*dVz_dz);                    
        
            aux_Tzz += dt*(c13*dVx_dx + c36*dVx_dy + c35*dVx_dz +
                           c36*dVy_dx + c23*dVy_dy + c34*dVy_dz +
                           c35*dVz_dx + c34*dVz_dy + c33*dVz_dz);  

            // dVx_dx ---------------------------------------------------------------------

            dVx_dx1 = (CFDM1*(Vx[bsi + jp3 + bsk] - Vx[bsi + jm3 + bsk]) +
                       CFDM2*(Vx[bsi + jm2 + bsk] - Vx[bsi + jp2 + bsk]) +
                       CFDM3*(Vx[bsi + jp1 + bsk] - Vx[bsi + jm1 + bsk])) * inv_dx;
                            
            dVx_dx2 = (CFDM1*(Vx[bsi + jp3 + kp1] - Vx[bsi + jm3 + kp1]) +
                       CFDM2*(Vx[bsi + jm2 + kp1] - Vx[bsi + jp2 + kp1]) +
                       CFDM3*(Vx[bsi + jp1 + kp1] - Vx[bsi + jm1 + kp1])) * inv_dx;

            dVx_dx = 0.5f*(dVx_dx1 + dVx_dx2);

            // dVx_dy ---------------------------------------------------------------------

            dVx_dy = (FDM1*(Vx[bsi + bsj + km3] - Vx[bsi + bsj + kp4]) +
                      FDM2*(Vx[bsi + bsj + kp3] - Vx[bsi + bsj + km2]) +
                      FDM3*(Vx[bsi + bsj + km1] - Vx[bsi + bsj + kp2]) +
                      FDM4*(Vx[bsi + bsj + kp1] - Vx[bsi + bsj + bsk])) * inv_dy;

            // dVx_dz ---------------------------------------------------------------------

            dVx_dz1 = (CFDM1*(Vx[ip3 + bsj + bsk] - Vx[im3 + bsj + bsk]) +
                       CFDM2*(Vx[im2 + bsj + bsk] - Vx[ip2 + bsj + bsk]) +
                       CFDM3*(Vx[ip1 + bsj + bsk] - Vx[im1 + bsj + bsk])) * inv_dz;

            dVx_dz2 = (CFDM1*(Vx[ip3 + bsj + kp1] - Vx[im3 + bsj + kp1]) +
                       CFDM2*(Vx[im2 + bsj + kp1] - Vx[ip2 + bsj + kp1]) +
                       CFDM3*(Vx[ip1 + bsj + kp1] - Vx[im1 + bsj + kp1])) * inv_dz;

            dVx_dz = 0.5f*(dVx_dz1 + dVx_dz2);

            // dVy_dx ---------------------------------------------------------------------

            dVy_dx = (FDM1*(Vy[bsi + jm3 + bsk] - Vy[bsi + jp4 + bsk]) +
                      FDM2*(Vy[bsi + jp3 + bsk] - Vy[bsi + jm2 + bsk]) +
                      FDM3*(Vy[bsi + jm1 + bsk] - Vy[bsi + jp2 + bsk]) +
                      FDM4*(Vy[bsi + jp1 + bsk] - Vy[bsi + bsj + bsk])) * inv_dx;

            // dVy_dy ---------------------------------------------------------------------

            dVy_dy1 = (CFDM1*(Vy[bsi + bsj + kp3] - Vy[bsi + bsj + km3]) +
                       CFDM2*(Vy[bsi + bsj + km2] - Vy[bsi + bsj + kp2]) +
                       CFDM3*(Vy[bsi + bsj + kp1] - Vy[bsi + bsj + km1])) * inv_dy;

            dVy_dy2 = (CFDM1*(Vy[bsi + jp1 + kp3] - Vy[bsi + jp1 + km3]) +
                       CFDM2*(Vy[bsi + jp1 + km2] - Vy[bsi + jp1 + kp2]) +
                       CFDM3*(Vy[bsi + jp1 + kp1] - Vy[bsi + jp1 + km1])) * inv_dy;

            dVy_dy = 0.5f*(dVy_dy1 + dVy_dy2);

            // dVy_dz ---------------------------------------------------------------------

            dVy_dz1 = (CFDM1*(Vy[ip3 + bsj + bsk] - Vy[im3 + bsj + bsk]) +
                       CFDM2*(Vy[im2 + bsj + bsk] - Vy[ip2 + bsj + bsk]) +
                       CFDM3*(Vy[ip1 + bsj + bsk] - Vy[im1 + bsj + bsk])) * inv_dz;

            dVy_dz2 = (CFDM1*(Vy[ip3 + bsj + kp1] - Vy[im3 + bsj + kp1]) +
                       CFDM2*(Vy[im2 + bsj + kp1] - Vy[ip2 + bsj + kp1]) +
                       CFDM3*(Vy[ip1 + bsj + kp1] - Vy[im1 + bsj + kp1])) * inv_dz;

            dVy_dz = 0.5f*(dVy_dz1 + dVy_dz2);

            // dVz_dx ---------------------------------------------------------------------

            dVz_dx1 = (FDM1*(Vz[bsi + jm3 + bsk] - Vz[bsi + jp4 + bsk]) +
                       FDM2*(Vz[bsi + jp3 + bsk] - Vz[bsi + jm2 + bsk]) +
                       FDM3*(Vz[bsi + jm1 + bsk] - Vz[bsi + jp2 + bsk]) +
                       FDM4*(Vz[bsi + jp1 + bsk] - Vz[bsi + bsj + bsk])) * inv_dx;

            dVz_dx2 = (FDM1*(Vz[im1 + jm3 + bsk] - Vz[im1 + jp4 + bsk]) +
                       FDM2*(Vz[im1 + jp3 + bsk] - Vz[im1 + jm2 + bsk]) +
                       FDM3*(Vz[im1 + jm1 + bsk] - Vz[im1 + jp2 + bsk]) +
                       FDM4*(Vz[im1 + jp1 + bsk] - Vz[im1 + bsj + bsk])) * inv_dx;

            dVz_dx3 = (FDM1*(Vz[bsi + jm3 + kp1] - Vz[bsi + jp4 + kp1]) +
                       FDM2*(Vz[bsi + jp3 + kp1] - Vz[bsi + jm2 + kp1]) +
                       FDM3*(Vz[bsi + jm1 + kp1] - Vz[bsi + jp2 + kp1]) +
                       FDM4*(Vz[bsi + jp1 + kp1] - Vz[bsi + bsj + kp1])) * inv_dx;

            dVz_dx4 = (FDM1*(Vz[im1 + jm3 + kp1] - Vz[im1 + jp4 + kp1]) +
                       FDM2*(Vz[im1 + jp3 + kp1] - Vz[im1 + jm2 + kp1]) +
                       FDM3*(Vz[im1 + jm1 + kp1] - Vz[im1 + jp2 + kp1]) +
                       FDM4*(Vz[im1 + jp1 + kp1] - Vz[im1 + bsj + kp1])) * inv_dx;    

            dVz_dx = 0.25f*(dVz_dx1 + dVz_dx2 + dVz_dx3 + dVz_dx4);    

            // dVz_dy ---------------------------------------------------------------------

            dVz_dy1 = (FDM1*(Vz[bsi + bsj + km3] - Vz[bsi + bsj + kp4]) +
                       FDM2*(Vz[bsi + bsj + kp3] - Vz[bsi + bsj + km2]) +
                       FDM3*(Vz[bsi + bsj + km1] - Vz[bsi + bsj + kp2]) +
                       FDM4*(Vz[bsi + bsj + kp1] - Vz[bsi + bsj + bsk])) * inv_dy;

            dVz_dy2 = (FDM1*(Vz[im1 + bsj + km3] - Vz[im1 + bsj + kp4]) +
                       FDM2*(Vz[im1 + bsj + kp3] - Vz[im1 + bsj + km2]) +
                       FDM3*(Vz[im1 + bsj + km1] - Vz[im1 + bsj + kp2]) +
                       FDM4*(Vz[im1 + bsj + kp1] - Vz[im1 + bsj + bsk])) * inv_dy;

            dVz_dy3 = (FDM1*(Vz[bsi + jp1 + km3] - Vz[bsi + jp1 + kp4]) +
                       FDM2*(Vz[bsi + jp1 + kp3] - Vz[bsi + jp1 + km2]) +
                       FDM3*(Vz[bsi + jp1 + km1] - Vz[bsi + jp1 + kp2]) +
                       FDM4*(Vz[bsi + jp1 + kp1] - Vz[bsi + jp1 + bsk])) * inv_dy;    

            dVz_dy4 = (FDM1*(Vz[im1 + jp1 + km3] - Vz[im1 + jp1 + kp4]) +
                       FDM2*(Vz[im1 + jp1 + kp3] - Vz[im1 + jp1 + km2]) +
                       FDM3*(Vz[im1 + jp1 + km1] - Vz[im1 + jp1 + kp2]) +
                       FDM4*(Vz[im1 + jp1 + kp1] - Vz[im1 + jp1 + bsk])) * inv_dy;    

            dVz_dy = 0.25f*(dVz_dy1 + dVz_dy2 + dVz_dy3 + dVz_dy4);    

            // dVz_dz ---------------------------------------------------------------------
            
            dVz_dz1 = (FDM1*(Vz[im4 + bsj + bsk] - Vz[ip3 + bsj + bsk]) +
                       FDM2*(Vz[ip2 + bsj + bsk] - Vz[im3 + bsj + bsk]) +
                       FDM3*(Vz[im2 + bsj + bsk] - Vz[ip1 + bsj + bsk]) +
                       FDM4*(Vz[bsi + bsj + bsk] - Vz[im1 + bsj + bsk])) * inv_dz;    

            dVz_dz2 = (FDM1*(Vz[im4 + jp1 + bsk] - Vz[ip3 + jp1 + bsk]) +
                       FDM2*(Vz[ip2 + jp1 + bsk] - Vz[im3 + jp1 + bsk]) +
                       FDM3*(Vz[im2 + jp1 + bsk] - Vz[ip1 + jp1 + bsk]) +
                       FDM4*(Vz[bsi + jp1 + bsk] - Vz[im1 + jp1 + bsk])) * inv_dz;    

            dVz_dz3 = (FDM1*(Vz[im4 + bsj + kp1] - Vz[ip3 + jp1 + kp1]) +
                       FDM2*(Vz[ip2 + bsj + kp1] - Vz[im3 + jp1 + kp1]) +
                       FDM3*(Vz[im2 + bsj + kp1] - Vz[ip1 + jp1 + kp1]) +
                       FDM4*(Vz[bsi + bsj + kp1] - Vz[im1 + jp1 + kp1])) * inv_dz;    

            dVz_dz4 = (FDM1*(Vz[im4 + jp1 + kp1] - Vz[ip3 + jp1 + kp1]) +
                       FDM2*(Vz[ip2 + jp1 + kp1] - Vz[im3 + jp1 + kp1]) +
                       FDM3*(Vz[im2 + jp1 + kp1] - Vz[ip1 + jp1 + kp1]) +
                       FDM4*(Vz[bsi + jp1 + kp1] - Vz[im1 + jp1 + kp1])) * inv_dz;    

            dVz_dz = 0.25f*(dVz_dz1 + dVz_dz2 + dVz_dz3 + dVz_dz4);    
 
            // Equation ---------------------------------------------------------------------

            aux_Txy += dt*(c16*dVx_dx + c66*dVx_dy + c56*dVx_dz +
                           c66*dVy_dx + c26*dVy_dy + c46*dVy_dz +
                           c56*dVz_dx + c46*dVz_dy + c36*dVz_dz);                    

            // dVx_dx ---------------------------------------------------------------------
            
            dVx_dx1 = (CFDM1*(Vx[bsi + jp3 + bsk] - Vx[bsi + jm3 + bsk]) +
                       CFDM2*(Vx[bsi + jm2 + bsk] - Vx[bsi + jp2 + bsk]) +
                       CFDM3*(Vx[bsi + jp1 + bsk] - Vx[bsi + jm1 + bsk])) * inv_dx;

            dVx_dx2 = (CFDM1*(Vx[ip1 + jp3 + bsk] - Vx[ip1 + jm3 + bsk]) +
                       CFDM2*(Vx[ip1 + jm2 + bsk] - Vx[ip1 + jp2 + bsk]) +
                       CFDM3*(Vx[ip1 + jp1 + bsk] - Vx[ip1 + jm1 + bsk])) * inv_dx;

            dVx_dx = 0.5f*(dVx_dx1 + dVx_dx2);

            // dVx_dy ---------------------------------------------------------------------

            dVx_dy1 = (CFDM1*(Vx[bsi + bsj + kp3] - Vx[bsi + bsj + km3]) +
                       CFDM2*(Vx[bsi + bsj + km2] - Vx[bsi + bsj + kp2]) +
                       CFDM3*(Vx[bsi + bsj + kp1] - Vx[bsi + bsj + km1])) * inv_dy;

            dVx_dy2 = (CFDM1*(Vx[ip1 + bsj + kp3] - Vx[ip1 + bsj + km3]) +
                       CFDM2*(Vx[ip1 + bsj + km2] - Vx[ip1 + bsj + kp2]) +
                       CFDM3*(Vx[ip1 + bsj + kp1] - Vx[ip1 + bsj + km1])) * inv_dy;

            dVx_dy = 0.5f*(dVx_dy1 + dVx_dy2);

            // dVx_dz ---------------------------------------------------------------------

            dVx_dz = (FDM1*(Vx[im3 + bsj + bsk] - Vx[ip4 + bsj + bsk]) +
                      FDM2*(Vx[ip3 + bsj + bsk] - Vx[im2 + bsj + bsk]) +
                      FDM3*(Vx[im1 + bsj + bsk] - Vx[ip2 + bsj + bsk]) +
                      FDM4*(Vx[ip1 + bsj + bsk] - Vx[bsi + bsj + bsk])) * inv_dz;

            // dVy_dx ---------------------------------------------------------------------

            dVy_dx1 = (FDM1*(Vy[bsi + jm3 + bsk] - Vy[bsi + jp4 + bsk]) +
                       FDM2*(Vy[bsi + jp3 + bsk] - Vy[bsi + jm2 + bsk]) +
                       FDM3*(Vy[bsi + jm1 + bsk] - Vy[bsi + jp2 + bsk]) +
                       FDM4*(Vy[bsi + jp1 + bsk] - Vy[bsi + bsj + bsk])) * inv_dx;

            dVy_dx2 = (FDM1*(Vy[bsi + jm3 + km1] - Vy[bsi + jp4 + km1]) +
                       FDM2*(Vy[bsi + jp3 + km1] - Vy[bsi + jm2 + km1]) +
                       FDM3*(Vy[bsi + jm1 + km1] - Vy[bsi + jp2 + km1]) +
                       FDM4*(Vy[bsi + jp1 + km1] - Vy[bsi + bsj + km1])) * inv_dx;

            dVy_dx3 = (FDM1*(Vy[ip1 + jm3 + bsk] - Vy[ip1 + jp4 + bsk]) +
                       FDM2*(Vy[ip1 + jp3 + bsk] - Vy[ip1 + jm2 + bsk]) +
                       FDM3*(Vy[ip1 + jm1 + bsk] - Vy[ip1 + jp2 + bsk]) +
                       FDM4*(Vy[ip1 + jp1 + bsk] - Vy[ip1 + bsj + bsk])) * inv_dx;

            dVy_dx4 = (FDM1*(Vy[ip1 + jm3 + km1] - Vy[ip1 + jp4 + km1]) +
                       FDM2*(Vy[ip1 + jp3 + km1] - Vy[ip1 + jm2 + km1]) +
                       FDM3*(Vy[ip1 + jm1 + km1] - Vy[ip1 + jp2 + km1]) +
                       FDM4*(Vy[ip1 + jp1 + km1] - Vy[ip1 + bsj + km1])) * inv_dx;

            dVy_dx = 0.25f*(dVy_dx1 + dVy_dx2 + dVy_dx3 + dVy_dx4);    

            // dVy_dy ---------------------------------------------------------------------

            dVy_dy1 = (FDM1*(Vy[bsi + bsj + km4] - Vy[bsi + bsj + kp3]) +
                       FDM2*(Vy[bsi + bsj + kp2] - Vy[bsi + bsj + km3]) +
                       FDM3*(Vy[bsi + bsj + km2] - Vy[bsi + bsj + kp1]) +
                       FDM4*(Vy[bsi + bsj + bsk] - Vy[bsi + bsj + km1])) * inv_dy;

            dVy_dy2 = (FDM1*(Vy[bsi + jp1 + km4] - Vy[bsi + jp1 + kp3]) +
                       FDM2*(Vy[bsi + jp1 + kp2] - Vy[bsi + jp1 + km3]) +
                       FDM3*(Vy[bsi + jp1 + km2] - Vy[bsi + jp1 + kp1]) +
                       FDM4*(Vy[bsi + jp1 + bsk] - Vy[bsi + jp1 + km1])) * inv_dy;

            dVy_dy3 = (FDM1*(Vy[ip1 + bsj + km4] - Vy[ip1 + bsj + kp3]) +
                       FDM2*(Vy[ip1 + bsj + kp2] - Vy[ip1 + bsj + km3]) +
                       FDM3*(Vy[ip1 + bsj + km2] - Vy[ip1 + bsj + kp1]) +
                       FDM4*(Vy[ip1 + bsj + bsk] - Vy[ip1 + bsj + km1])) * inv_dy;

            dVy_dy4 = (FDM1*(Vy[ip1 + jp1 + km4] - Vy[ip1 + jp1 + kp3]) +
                       FDM2*(Vy[ip1 + jp1 + kp2] - Vy[ip1 + jp1 + km3]) +
                       FDM3*(Vy[ip1 + jp1 + km2] - Vy[ip1 + jp1 + kp1]) +
                       FDM4*(Vy[ip1 + jp1 + bsk] - Vy[ip1 + jp1 + km1])) * inv_dy;

            dVy_dy = 0.25f*(dVy_dy1 + dVy_dy2 + dVy_dy3 + dVy_dy4);    

            // dVy_dz ---------------------------------------------------------------------

            dVy_dz1 = (FDM1*(Vy[im3 + bsj + bsk] - Vy[ip4 + bsj + bsk]) +
                       FDM2*(Vy[ip3 + bsj + bsk] - Vy[im2 + bsj + bsk]) +
                       FDM3*(Vy[im1 + bsj + bsk] - Vy[ip2 + bsj + bsk]) +
                       FDM4*(Vy[ip1 + bsj + bsk] - Vy[bsi + bsj + bsk])) * inv_dz;

            dVy_dz2 = (FDM1*(Vy[im3 + bsj + km1] - Vy[ip4 + bsj + km1]) +
                       FDM2*(Vy[ip3 + bsj + km1] - Vy[im2 + bsj + km1]) +
                       FDM3*(Vy[im1 + bsj + km1] - Vy[ip2 + bsj + km1]) +
                       FDM4*(Vy[ip1 + bsj + km1] - Vy[bsi + bsj + km1])) * inv_dz;

            dVy_dz3 = (FDM1*(Vy[im3 + jp1 + bsk] - Vy[ip4 + jp1 + bsk]) +
                       FDM2*(Vy[ip3 + jp1 + bsk] - Vy[im2 + jp1 + bsk]) +
                       FDM3*(Vy[im1 + jp1 + bsk] - Vy[ip2 + jp1 + bsk]) +
                       FDM4*(Vy[ip1 + jp1 + bsk] - Vy[bsi + jp1 + bsk])) * inv_dz;

            dVy_dz4 = (FDM1*(Vy[im3 + jp1 + km1] - Vy[ip4 + jp1 + km1]) +
                       FDM2*(Vy[ip3 + jp1 + km1] - Vy[im2 + jp1 + km1]) +
                       FDM3*(Vy[im1 + jp1 + km1] - Vy[ip2 + jp1 + km1]) +
                       FDM4*(Vy[ip1 + jp1 + km1] - Vy[bsi + jp1 + km1])) * inv_dz;

            dVy_dz = 0.25f*(dVy_dz1 + dVy_dz2 + dVy_dz3 + dVy_dz4);    

            // dVz_dx ---------------------------------------------------------------------

            dVz_dx = (FDM1*(Vz[bsi + jm3 + bsk] - Vz[bsi + jp4 + bsk]) +
                      FDM2*(Vz[bsi + jp3 + bsk] - Vz[bsi + jm2 + bsk]) +
                      FDM3*(Vz[bsi + jm1 + bsk] - Vz[bsi + jp2 + bsk]) +
                      FDM4*(Vz[bsi + jp1 + bsk] - Vz[bsi + bsj + bsk])) * inv_dx;

            // dVz_dy ---------------------------------------------------------------------

            dVz_dy1 = (CFDM1*(Vz[bsi + bsj + kp3] - Vz[bsi + bsj + km3]) +
                       CFDM2*(Vz[bsi + bsj + km2] - Vz[bsi + bsj + kp2]) +
                       CFDM3*(Vz[bsi + bsj + kp1] - Vz[bsi + bsj + km1])) * inv_dy;

            dVz_dy2 = (CFDM1*(Vz[bsi + jp1 + kp3] - Vz[bsi + jp1 + km3]) +
                       CFDM2*(Vz[bsi + jp1 + km2] - Vz[bsi + jp1 + kp2]) +
                       CFDM3*(Vz[bsi + jp1 + kp1] - Vz[bsi + jp1 + km1])) * inv_dy;

            dVz_dy = 0.5f*(dVz_dy1 + dVz_dy2);

            // dVz_dz ---------------------------------------------------------------------

            dVz_dz1 = (CFDM1*(Vz[ip3 + bsj + bsk] - Vz[im3 + bsj + bsk]) +
                       CFDM2*(Vz[im2 + bsj + bsk] - Vz[ip2 + bsj + bsk]) +
                       CFDM3*(Vz[ip1 + bsj + bsk] - Vz[im1 + bsj + bsk])) * inv_dz;

            dVz_dz2 = (CFDM1*(Vz[ip3 + jp1 + bsk] - Vz[im3 + jp1 + bsk]) +
                       CFDM2*(Vz[im2 + jp1 + bsk] - Vz[ip2 + jp1 + bsk]) +
                       CFDM3*(Vz[ip1 + jp1 + bsk] - Vz[im1 + jp1 + bsk])) * inv_dz;

            dVz_dz = 0.5f*(dVz_dz1 + dVz_dz2);

            // Equation ---------------------------------------------------------------------

            aux_Txz += dt*(c15*dVx_dx + c56*dVx_dy + c55*dVx_dz +
                           c56*dVy_dx + c25*dVy_dy + c45*dVy_dz +
                           c55*dVz_dx + c45*dVz_dy + c35*dVz_dz);                    

            // dVx_dx ---------------------------------------------------------------------
            
            dVx_dx1 = (FDM1*(Vx[bsi + jm4 + bsk] - Vx[bsi + jp3 + bsk]) +
                       FDM2*(Vx[bsi + jp2 + bsk] - Vx[bsi + jm3 + bsk]) +
                       FDM3*(Vx[bsi + jm2 + bsk] - Vx[bsi + jp1 + bsk]) +
                       FDM4*(Vx[bsi + bsj + bsk] - Vx[bsi + jm1 + bsk])) * inv_dx;

            dVx_dx2 = (FDM1*(Vx[bsi + jm4 + km1] - Vx[bsi + jp3 + km1]) +
                       FDM2*(Vx[bsi + jp2 + km1] - Vx[bsi + jm3 + km1]) +
                       FDM3*(Vx[bsi + jm2 + km1] - Vx[bsi + jp1 + km1]) +
                       FDM4*(Vx[bsi + bsj + km1] - Vx[bsi + jm1 + km1])) * inv_dx; 
        
            dVx_dx3 = (FDM1*(Vx[ip1 + jm4 + bsk] - Vx[ip1 + jp3 + bsk]) +
                       FDM2*(Vx[ip1 + jp2 + bsk] - Vx[ip1 + jm3 + bsk]) +
                       FDM3*(Vx[ip1 + jm2 + bsk] - Vx[ip1 + jp1 + bsk]) +
                       FDM4*(Vx[ip1 + bsj + bsk] - Vx[ip1 + jm1 + bsk])) * inv_dx; 
            
            dVx_dx4 = (FDM1*(Vx[ip1 + jm4 + km1] - Vx[ip1 + jp3 + km1]) +
                       FDM2*(Vx[ip1 + jp2 + km1] - Vx[ip1 + jm3 + km1]) +
                       FDM3*(Vx[ip1 + jm2 + km1] - Vx[ip1 + jp1 + km1]) +
                       FDM4*(Vx[ip1 + bsj + km1] - Vx[ip1 + jm1 + km1])) * inv_dx; 

            dVx_dx = 0.25f*(dVx_dx1 + dVx_dx2 + dVx_dx3 + dVx_dx4);

            // dVx_dy ---------------------------------------------------------------------

            dVx_dy1 = (FDM1*(Vx[bsi + bsj + km3] - Vx[bsi + bsj + kp4]) +
                       FDM2*(Vx[bsi + bsj + kp3] - Vx[bsi + bsj + km2]) +
                       FDM3*(Vx[bsi + bsj + km1] - Vx[bsi + bsj + kp2]) +
                       FDM4*(Vx[bsi + bsj + kp1] - Vx[bsi + bsj + bsk])) * inv_dy;

            dVx_dy2 = (FDM1*(Vx[bsi + jm1 + km3] - Vx[bsi + jm1 + kp4]) +
                       FDM2*(Vx[bsi + jm1 + kp3] - Vx[bsi + jm1 + km2]) +
                       FDM3*(Vx[bsi + jm1 + km1] - Vx[bsi + jm1 + kp2]) +
                       FDM4*(Vx[bsi + jm1 + kp1] - Vx[bsi + jm1 + bsk])) * inv_dy; 
        
            dVx_dy3 = (FDM1*(Vx[ip1 + bsj + km3] - Vx[ip1 + bsj + kp4]) +
                       FDM2*(Vx[ip1 + bsj + kp3] - Vx[ip1 + bsj + km2]) +
                       FDM3*(Vx[ip1 + bsj + km1] - Vx[ip1 + bsj + kp2]) +
                       FDM4*(Vx[ip1 + bsj + kp1] - Vx[ip1 + bsj + bsk])) * inv_dy; 
            
            dVx_dy4 = (FDM1*(Vx[ip1 + jm1 + km3] - Vx[ip1 + jm1 + kp4]) +
                       FDM2*(Vx[ip1 + jm1 + kp3] - Vx[ip1 + jm1 + km2]) +
                       FDM3*(Vx[ip1 + jm1 + km1] - Vx[ip1 + jm1 + kp2]) +
                       FDM4*(Vx[ip1 + jm1 + kp1] - Vx[ip1 + jm1 + bsk])) * inv_dy; 

            dVx_dy = 0.25f*(dVx_dy1 + dVx_dy2 + dVx_dy3 + dVx_dy4);

            // dVx_dz ---------------------------------------------------------------------

            dVx_dz1 = (FDM1*(Vx[im3 + bsj + bsk] - Vx[ip4 + bsj + bsk]) +
                       FDM2*(Vx[ip3 + bsj + bsk] - Vx[im2 + bsj + bsk]) +
                       FDM3*(Vx[im1 + bsj + bsk] - Vx[ip2 + bsj + bsk]) +
                       FDM4*(Vx[ip1 + bsj + bsk] - Vx[bsi + bsj + bsk])) * inv_dz;

            dVx_dz2 = (FDM1*(Vx[im3 + jm1 + bsk] - Vx[ip4 + jm1 + bsk]) +
                       FDM2*(Vx[ip3 + jm1 + bsk] - Vx[im2 + jm1 + bsk]) +
                       FDM3*(Vx[im1 + jm1 + bsk] - Vx[ip2 + jm1 + bsk]) +
                       FDM4*(Vx[ip1 + jm1 + bsk] - Vx[bsi + jm1 + bsk])) * inv_dz; 
        
            dVx_dz3 = (FDM1*(Vx[im3 + bsj + kp1] - Vx[ip4 + bsj + kp1]) +
                       FDM2*(Vx[ip3 + bsj + kp1] - Vx[im2 + bsj + kp1]) +
                       FDM3*(Vx[im1 + bsj + kp1] - Vx[ip2 + bsj + kp1]) +
                       FDM4*(Vx[ip1 + bsj + kp1] - Vx[bsi + bsj + kp1])) * inv_dz; 
            
            dVx_dz4 = (FDM1*(Vx[im3 + jm1 + kp1] - Vx[ip4 + jm1 + kp1]) +
                       FDM2*(Vx[ip3 + jm1 + kp1] - Vx[im2 + jm1 + kp1]) +
                       FDM3*(Vx[im1 + jm1 + kp1] - Vx[ip2 + jm1 + kp1]) +
                       FDM4*(Vx[ip1 + jm1 + kp1] - Vx[bsi + jm1 + kp1])) * inv_dz;  

            dVx_dz = 0.25f*(dVx_dz1 + dVx_dz2 + dVx_dz3 + dVx_dz4);

            // dVy_dx ---------------------------------------------------------------------

            dVy_dx1 = (CFDM1*(Vy[bsi + jp3 + bsk] - Vy[bsi + jm3 + bsk]) +
                       CFDM2*(Vy[bsi + jm2 + bsk] - Vy[bsi + jp2 + bsk]) +
                       CFDM3*(Vy[bsi + jp1 + bsk] - Vy[bsi + jm1 + bsk])) * inv_dx;

            dVy_dx2 = (CFDM1*(Vy[ip1 + jp3 + bsk] - Vy[ip1 + jm3 + bsk]) +
                       CFDM2*(Vy[ip1 + jm2 + bsk] - Vy[ip1 + jp2 + bsk]) +
                       CFDM3*(Vy[ip1 + jp1 + bsk] - Vy[ip1 + jm1 + bsk])) * inv_dx; 
            
            dVy_dx = 0.5f*(dVy_dx1 + dVy_dx2);

            // dVy_dy ---------------------------------------------------------------------

            dVy_dy1 = (CFDM1*(Vy[bsi + bsj + kp3] - Vy[bsi + bsj + km3]) +
                       CFDM2*(Vy[bsi + bsj + km2] - Vy[bsi + bsj + kp2]) +
                       CFDM3*(Vy[bsi + bsj + kp1] - Vy[bsi + bsj + km1])) * inv_dy;

            dVy_dy2 = (CFDM1*(Vy[ip1 + bsj + kp3] - Vy[ip1 + bsj + km3]) +
                       CFDM2*(Vy[ip1 + bsj + km2] - Vy[ip1 + bsj + kp2]) +
                       CFDM3*(Vy[ip1 + bsj + kp1] - Vy[ip1 + bsj + km1])) * inv_dy; 
            
            dVy_dy = 0.5f*(dVy_dy1 + dVy_dy2);

            // dVy_dz ---------------------------------------------------------------------

            dVy_dz = (FDM1*(Vy[im3 + bsj + bsk] - Vy[ip4 + bsj + bsk]) +
                      FDM2*(Vy[ip3 + bsj + bsk] - Vy[im2 + bsj + bsk]) +
                      FDM3*(Vy[im1 + bsj + bsk] - Vy[ip2 + bsj + bsk]) +
                      FDM4*(Vy[ip1 + bsj + bsk] - Vy[bsi + bsj + bsk])) * inv_dz;

            // dVz_dx ---------------------------------------------------------------------

            dVz_dx1 = (CFDM1*(Vz[bsi + jp3 + bsk] - Vz[bsi + jm3 + bsk]) +
                       CFDM2*(Vz[bsi + jm2 + bsk] - Vz[bsi + jp2 + bsk]) +
                       CFDM3*(Vz[bsi + jp1 + bsk] - Vz[bsi + jm1 + bsk])) * inv_dx;

            dVz_dx2 = (CFDM1*(Vz[bsi + jp3 + kp1] - Vz[bsi + jm3 + kp1]) +
                       CFDM2*(Vz[bsi + jm2 + kp1] - Vz[bsi + jp2 + kp1]) +
                       CFDM3*(Vz[bsi + jp1 + kp1] - Vz[bsi + jm1 + kp1])) * inv_dx;
            
            dVz_dx = 0.5f*(dVz_dx1 + dVz_dx2);

            // dVz_dy ---------------------------------------------------------------------

            dVz_dy = (FDM1*(Vz[bsi + bsj + km3] - Vz[bsi + bsj + kp4]) +
                      FDM2*(Vz[bsi + bsj + kp3] - Vz[bsi + bsj + km2]) +
                      FDM3*(Vz[bsi + bsj + km1] - Vz[bsi + bsj + kp2]) +
                      FDM4*(Vz[bsi + bsj + kp1] - Vz[bsi + bsj + bsk])) * inv_dy;

            // dVz_dz ---------------------------------------------------------------------

            dVz_dz1 = (CFDM1*(Vz[ip3 + bsj + bsk] - Vz[im3 + bsj + bsk]) +
                       CFDM2*(Vz[im2 + bsj + bsk] - Vz[ip2 + bsj + bsk]) +
                       CFDM3*(Vz[ip1 + bsj + bsk] - Vz[im1 + bsj + bsk])) * inv_dz;

            dVz_dz2 = (CFDM1*(Vz[ip3 + bsj + kp1] - Vz[im3 + bsj + kp1]) +
                       CFDM2*(Vz[im2 + bsj + kp1] - Vz[ip2 + bsj + kp1]) +
                       CFDM3*(Vz[ip1 + bsj + kp1] - Vz[im1 + bsj + kp1])) * inv_dz; 
            
            dVz_dz = 0.5f*(dVz_dz1 + dVz_dz2);

            // Equation ---------------------------------------------------------------------

            aux_Tyz += dt*(c14*dVx_dx + c46*dVx_dy + c45*dVx_dz +
                           c46*dVy_dx + c24*dVy_dy + c44*dVy_dz +
                           c45*dVz_dx + c44*dVz_dy + c34*dVz_dz); 
        
            P[index] = (aux_Txx + aux_Tyy + aux_Tzz) / 3.0f;

            Txx[index] = aux_Txx;
            Tyy[index] = aux_Tyy;
            Tzz[index] = aux_Tzz;
            Txy[index] = aux_Txy;
            Txz[index] = aux_Txz;
            Tyz[index] = aux_Tyz;
        }
    }
}
