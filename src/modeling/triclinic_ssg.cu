# include "triclinic_ssg.cuh"

void Triclinic_SSG::initialization()
{
    modeling_name = "Triclinic media with Standard Staggered Grid";
    modeling_type = "triclinic_ssg";

    float * h_skw = new float[DGS*DGS*DGS]();

    auto skw = kaiser_weights(sx, sy, sz, sIdx, sIdy, sIdz, dx, dy, dz);

    for (int yId = 0; yId < DGS; yId++)
        for (int xId = 0; xId < DGS; xId++)
            for (int zId = 0; zId < DGS; zId++)
                h_skw[zId + xId*DGS + yId*DGS*DGS] = skw[zId][xId][yId];

    sIdx += nb; 
    sIdy += nb; 
    sIdz += nb;

    int * h_rIdx = new int[max_spread]();
    int * h_rIdy = new int[max_spread]();
    int * h_rIdz = new int[max_spread]();

    float * h_rkwPs = new float[DGS*DGS*DGS*max_spread]();
    float * h_rkwVx = new float[DGS*DGS*DGS*max_spread]();
    float * h_rkwVy = new float[DGS*DGS*DGS*max_spread]();
    float * h_rkwVz = new float[DGS*DGS*DGS*max_spread]();

    int spreadId = 0;

    for (recId = geometry->iRec[srcId]; recId < geometry->fRec[srcId]; recId++)
    {
        float rx = geometry->xrec[recId];
        float ry = geometry->yrec[recId];
        float rz = geometry->zrec[recId];
        
        int rIdx = (int)((rx + 0.5f*dx) / dx);
        int rIdy = (int)((ry + 0.5f*dy) / dy);
        int rIdz = (int)((rz + 0.5f*dz) / dz);
    
        auto rkwPs = kaiser_weights(rx, ry, rz, rIdx, rIdy, rIdz, dx, dy, dz);
        auto rkwVx = kaiser_weights(rx + 0.5f*dx, ry, rz, rIdx, rIdy, rIdz, dx, dy, dz);
        auto rkwVy = kaiser_weights(rx, ry + 0.5f*dy, rz, rIdx, rIdy, rIdz, dx, dy, dz);
        auto rkwVz = kaiser_weights(rx, ry, rz + 0.5f*dz, rIdx, rIdy, rIdz, dx, dy, dz);
        
        for (int zId = 0; zId < DGS; zId++)
        {
            for (int xId = 0; xId < DGS; xId++)
            {
                for (int yId = 0; yId < DGS; yId++)
                {
                    h_rkwPs[zId + xId*DGS + yId*DGS*DGS + spreadId*DGS*DGS*DGS] = rkwPs[zId][xId][yId];
                    h_rkwVx[zId + xId*DGS + yId*DGS*DGS + spreadId*DGS*DGS*DGS] = rkwVx[zId][xId][yId];
                    h_rkwVy[zId + xId*DGS + yId*DGS*DGS + spreadId*DGS*DGS*DGS] = rkwVy[zId][xId][yId];
                    h_rkwVz[zId + xId*DGS + yId*DGS*DGS + spreadId*DGS*DGS*DGS] = rkwVz[zId][xId][yId];
                }
            }
        }

        h_rIdx[spreadId] = rIdx + nb;
        h_rIdy[spreadId] = rIdy + nb;
        h_rIdz[spreadId] = rIdz + nb;

        ++spreadId;
    }

    cudaMemcpy(d_skw, h_skw, DGS*DGS*DGS*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rkwPs, h_rkwPs, DGS*DGS*DGS*max_spread*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rkwVx, h_rkwVx, DGS*DGS*DGS*max_spread*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rkwVy, h_rkwVy, DGS*DGS*DGS*max_spread*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rkwVz, h_rkwVz, DGS*DGS*DGS*max_spread*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rIdx, h_rIdx, max_spread*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdy, h_rIdy, max_spread*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdz, h_rIdz, max_spread*sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_skw;
    delete[] h_rkwPs;
    delete[] h_rkwVx;
    delete[] h_rkwVz;
    delete[] h_rIdx;
    delete[] h_rIdz;
}

void Triclinic_SSG::compute_velocity()
{
    if (compression)
    {
        uintc_compute_velocity_ssg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_T,dc_B,
                                                         maxB,minB,d1D,d2D,d3D,d_wavelet,dx,dy,dz,dt,timeId,tlag,sIdx, 
                                                         sIdy,sIdz,d_skw,nxx,nyy,nzz,nb,nt,eikonalClip);
    }
    else 
    {
        float_compute_velocity_ssg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_T,d_B,
                                                         d1D,d2D,d3D,d_wavelet,dx,dy,dz,dt,timeId,tlag,sIdx,sIdy,sIdz,
                                                         d_skw,nxx,nyy,nzz,nb,nt,eikonalClip);
    }
}

void Triclinic_SSG::compute_pressure()
{
    if (compression)
    {
        uintc_compute_pressure_ssg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_P,d_T, 
                                                         dc_C11,dc_C12,dc_C13,dc_C14,dc_C15,dc_C16,dc_C22,dc_C23,dc_C24,
                                                         dc_C25,dc_C26,dc_C33,dc_C34,dc_C35,dc_C36,dc_C44,dc_C45,dc_C46,
                                                         dc_C55,dc_C56,dc_C66,timeId,tlag,dx,dy,dz,dt,nxx,nyy,nzz,minC11, 
                                                         maxC11,minC12,maxC12,minC13,maxC13,minC14,maxC14,minC15,maxC15,
                                                         minC16,maxC16,minC22,maxC22,minC23,maxC23,minC24,maxC24,minC25,
                                                         maxC25,minC26,maxC26,minC33,maxC33,minC34,maxC34,minC35,maxC35, 
                                                         minC36,maxC36,minC44,maxC44,minC45,maxC45,minC46,maxC46,minC55, 
                                                         maxC55,minC56,maxC56,minC66,maxC66,eikonalClip);
    }
    else 
    {
        float_compute_pressure_ssg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_P,d_T, 
                                                         d_C11,d_C12,d_C13,d_C14,d_C15,d_C16,d_C22,d_C23,d_C24,d_C25,
                                                         d_C26,d_C33,d_C34,d_C35,d_C36,d_C44,d_C45,d_C46,d_C55,d_C56, 
                                                         d_C66,timeId,tlag,dx,dy,dz,dt,nxx,nyy,nzz,eikonalClip);
    }
}

__global__ void uintc_compute_velocity_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, uintc * B,
                                           float maxB, float minB, float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, 
                                           int tlag, int sIdx, int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    float Bn, Bm;

    if ((index == 0) && (tId < nt))
    {   
        for (int k = 0; k < DGS; k++)
        {
            int yi = sIdy + k - 3;
            
            for (int j = 0; j < DGS; j++)
            {
                int xi = sIdx + j - 3;
    
                for (int i = 0; i < DGS; i++)
                {
                    int zi = sIdz + i - 3;
            
                    Txx[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);
                    Tyy[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);
                    Tzz[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);           
                }
            }
        }
    }
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        Bn = (minB + (static_cast<float>(B[index]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

        if((i >= 3) && (i < nzz-4) && (j > 3) && (j < nxx-3) && (k >= 3) && (k < nyy-4)) 
        {
            float dTxx_dx = (FDM1*(Txx[i + (j-4)*nzz + k*nxx*nzz] - Txx[i + (j+3)*nzz + k*nxx*nzz]) +
                             FDM2*(Txx[i + (j+2)*nzz + k*nxx*nzz] - Txx[i + (j-3)*nzz + k*nxx*nzz]) +
                             FDM3*(Txx[i + (j-2)*nzz + k*nxx*nzz] - Txx[i + (j+1)*nzz + k*nxx*nzz]) +
                             FDM4*(Txx[i + j*nzz + k*nxx*nzz]     - Txx[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dTxy_dy = (FDM1*(Txy[i + j*nzz + (k-3)*nxx*nzz] - Txy[i + j*nzz + (k+4)*nxx*nzz]) +
                             FDM2*(Txy[i + j*nzz + (k+3)*nxx*nzz] - Txy[i + j*nzz + (k-2)*nxx*nzz]) +
                             FDM3*(Txy[i + j*nzz + (k-1)*nxx*nzz] - Txy[i + j*nzz + (k+2)*nxx*nzz]) +
                             FDM4*(Txy[i + j*nzz + (k+1)*nxx*nzz] - Txy[i + j*nzz + k*nxx*nzz])) / dy;

            float dTxz_dz = (FDM1*(Txz[(i-3) + j*nzz + k*nxx*nzz] - Txz[(i+4) + j*nzz + k*nxx*nzz]) +
                             FDM2*(Txz[(i+3) + j*nzz + k*nxx*nzz] - Txz[(i-2) + j*nzz + k*nxx*nzz]) +
                             FDM3*(Txz[(i-1) + j*nzz + k*nxx*nzz] - Txz[(i+2) + j*nzz + k*nxx*nzz]) +
                             FDM4*(Txz[(i+1) + j*nzz + k*nxx*nzz] - Txz[i + j*nzz + k*nxx*nzz])) / dz;

            Bm = (minB + (static_cast<float>(B[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float Bx = 0.5f*(Bn + Bm);

            Vx[index] += dt*Bx*(dTxx_dx + dTxy_dy + dTxz_dz); 
        }

        if((i >= 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4) && (k > 3) && (k < nyy-3)) 
        {
            float dTxy_dx = (FDM1*(Txy[i + (j-3)*nzz + k*nxx*nzz] - Txy[i + (j+4)*nzz + k*nxx*nzz]) +
                             FDM2*(Txy[i + (j+3)*nzz + k*nxx*nzz] - Txy[i + (j-2)*nzz + k*nxx*nzz]) +
                             FDM3*(Txy[i + (j-1)*nzz + k*nxx*nzz] - Txy[i + (j+2)*nzz + k*nxx*nzz]) +
                             FDM4*(Txy[i + (j+1)*nzz + k*nxx*nzz] - Txy[i + j*nzz + k*nxx*nzz])) / dx;

            float dTyy_dy = (FDM1*(Tyy[i + j*nzz + (k-4)*nxx*nzz] - Tyy[i + j*nzz + (k+3)*nxx*nzz]) +
                             FDM2*(Tyy[i + j*nzz + (k+2)*nxx*nzz] - Tyy[i + j*nzz + (k-3)*nxx*nzz]) +
                             FDM3*(Tyy[i + j*nzz + (k-2)*nxx*nzz] - Tyy[i + j*nzz + (k+1)*nxx*nzz]) +
                             FDM4*(Tyy[i + j*nzz + k*nxx*nzz]     - Tyy[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dTyz_dz = (FDM1*(Tyz[(i-3) + j*nzz + k*nxx*nzz] - Tyz[(i+4) + j*nzz + k*nxx*nzz]) +
                             FDM2*(Tyz[(i+3) + j*nzz + k*nxx*nzz] - Tyz[(i-2) + j*nzz + k*nxx*nzz]) +
                             FDM3*(Tyz[(i-1) + j*nzz + k*nxx*nzz] - Tyz[(i+2) + j*nzz + k*nxx*nzz]) +
                             FDM4*(Tyz[(i+1) + j*nzz + k*nxx*nzz] - Tyz[i + j*nzz + k*nxx*nzz])) / dz;

            Bm = (minB + (static_cast<float>(B[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float By = 0.5f*(Bn + Bm);

            Vy[index] += dt*By*(dTxy_dx + dTyy_dy + dTyz_dz); 
        }    

        if((i > 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4) && (k >= 3) && (k < nyy-4)) 
        {
            float dTxz_dx = (FDM1*(Txz[i + (j-3)*nzz + k*nxx*nzz] - Txz[i + (j+4)*nzz + k*nxx*nzz]) +
                             FDM2*(Txz[i + (j+3)*nzz + k*nxx*nzz] - Txz[i + (j-2)*nzz + k*nxx*nzz]) +
                             FDM3*(Txz[i + (j-1)*nzz + k*nxx*nzz] - Txz[i + (j+2)*nzz + k*nxx*nzz]) +
                             FDM4*(Txz[i + (j+1)*nzz + k*nxx*nzz] - Txz[i + j*nzz + k*nxx*nzz])) / dx;

            float dTyz_dy = (FDM1*(Tyz[i + j*nzz + (k-3)*nxx*nzz] - Tyz[i + j*nzz + (k+4)*nxx*nzz]) +
                             FDM2*(Tyz[i + j*nzz + (k+3)*nxx*nzz] - Tyz[i + j*nzz + (k-2)*nxx*nzz]) +
                             FDM3*(Tyz[i + j*nzz + (k-1)*nxx*nzz] - Tyz[i + j*nzz + (k+2)*nxx*nzz]) +
                             FDM4*(Tyz[i + j*nzz + (k+1)*nxx*nzz] - Tyz[i + j*nzz + k*nxx*nzz])) / dy;

            float dTzz_dz = (FDM1*(Tzz[(i-4) + j*nzz + k*nxx*nzz] - Tzz[(i+3) + j*nzz + k*nxx*nzz]) +
                             FDM2*(Tzz[(i+2) + j*nzz + k*nxx*nzz] - Tzz[(i-3) + j*nzz + k*nxx*nzz]) +
                             FDM3*(Tzz[(i-2) + j*nzz + k*nxx*nzz] - Tzz[(i+1) + j*nzz + k*nxx*nzz]) +
                             FDM4*(Tzz[i + j*nzz + k*nxx*nzz]     - Tzz[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            Bm = (minB + (static_cast<float>(B[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float Bz = 0.5f*(Bn + Bm);

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

__global__ void float_compute_velocity_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, float * B,
                                           float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, int tlag, int sIdx, 
                                           int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    if ((index == 0) && (tId < nt))
    {   
        for (int k = 0; k < DGS; k++)
        {
            int yi = sIdy + k - 3;
            
            for (int j = 0; j < DGS; j++)
            {
                int xi = sIdx + j - 3;
    
                for (int i = 0; i < DGS; i++)
                {
                    int zi = sIdz + i - 3;
            
                    Txx[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);
                    Tyy[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);
                    Tzz[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);           
                }
            }
        }
    }
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        if((i >= 3) && (i < nzz-4) && (j > 3) && (j < nxx-3) && (k >= 3) && (k < nyy-4)) 
        {
            float dTxx_dx = (FDM1*(Txx[i + (j-4)*nzz + k*nxx*nzz] - Txx[i + (j+3)*nzz + k*nxx*nzz]) +
                             FDM2*(Txx[i + (j+2)*nzz + k*nxx*nzz] - Txx[i + (j-3)*nzz + k*nxx*nzz]) +
                             FDM3*(Txx[i + (j-2)*nzz + k*nxx*nzz] - Txx[i + (j+1)*nzz + k*nxx*nzz]) +
                             FDM4*(Txx[i + j*nzz + k*nxx*nzz]     - Txx[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dTxy_dy = (FDM1*(Txy[i + j*nzz + (k-3)*nxx*nzz] - Txy[i + j*nzz + (k+4)*nxx*nzz]) +
                             FDM2*(Txy[i + j*nzz + (k+3)*nxx*nzz] - Txy[i + j*nzz + (k-2)*nxx*nzz]) +
                             FDM3*(Txy[i + j*nzz + (k-1)*nxx*nzz] - Txy[i + j*nzz + (k+2)*nxx*nzz]) +
                             FDM4*(Txy[i + j*nzz + (k+1)*nxx*nzz] - Txy[i + j*nzz + k*nxx*nzz])) / dy;

            float dTxz_dz = (FDM1*(Txz[(i-3) + j*nzz + k*nxx*nzz] - Txz[(i+4) + j*nzz + k*nxx*nzz]) +
                             FDM2*(Txz[(i+3) + j*nzz + k*nxx*nzz] - Txz[(i-2) + j*nzz + k*nxx*nzz]) +
                             FDM3*(Txz[(i-1) + j*nzz + k*nxx*nzz] - Txz[(i+2) + j*nzz + k*nxx*nzz]) +
                             FDM4*(Txz[(i+1) + j*nzz + k*nxx*nzz] - Txz[i + j*nzz + k*nxx*nzz])) / dz;

            float Bx = 0.5f*(B[i + (j+1)*nzz + k*nxx*nzz] + B[i + j*nzz + k*nxx*nzz]);

            Vx[index] += dt*Bx*(dTxx_dx + dTxy_dy + dTxz_dz); 
        }

        if((i >= 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4) && (k > 3) && (k < nyy-3)) 
        {
            float dTxy_dx = (FDM1*(Txy[i + (j-3)*nzz + k*nxx*nzz] - Txy[i + (j+4)*nzz + k*nxx*nzz]) +
                             FDM2*(Txy[i + (j+3)*nzz + k*nxx*nzz] - Txy[i + (j-2)*nzz + k*nxx*nzz]) +
                             FDM3*(Txy[i + (j-1)*nzz + k*nxx*nzz] - Txy[i + (j+2)*nzz + k*nxx*nzz]) +
                             FDM4*(Txy[i + (j+1)*nzz + k*nxx*nzz] - Txy[i + j*nzz + k*nxx*nzz])) / dx;

            float dTyy_dy = (FDM1*(Tyy[i + j*nzz + (k-4)*nxx*nzz] - Tyy[i + j*nzz + (k+3)*nxx*nzz]) +
                             FDM2*(Tyy[i + j*nzz + (k+2)*nxx*nzz] - Tyy[i + j*nzz + (k-3)*nxx*nzz]) +
                             FDM3*(Tyy[i + j*nzz + (k-2)*nxx*nzz] - Tyy[i + j*nzz + (k+1)*nxx*nzz]) +
                             FDM4*(Tyy[i + j*nzz + k*nxx*nzz]     - Tyy[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dTyz_dz = (FDM1*(Tyz[(i-3) + j*nzz + k*nxx*nzz] - Tyz[(i+4) + j*nzz + k*nxx*nzz]) +
                             FDM2*(Tyz[(i+3) + j*nzz + k*nxx*nzz] - Tyz[(i-2) + j*nzz + k*nxx*nzz]) +
                             FDM3*(Tyz[(i-1) + j*nzz + k*nxx*nzz] - Tyz[(i+2) + j*nzz + k*nxx*nzz]) +
                             FDM4*(Tyz[(i+1) + j*nzz + k*nxx*nzz] - Tyz[i + j*nzz + k*nxx*nzz])) / dz;

            float By = 0.5f*(B[i + j*nzz + (k+1)*nxx*nzz] + B[i + j*nzz + k*nxx*nzz]);

            Vy[index] += dt*By*(dTxy_dx + dTyy_dy + dTyz_dz); 
        }    

        if((i > 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4) && (k >= 3) && (k < nyy-4)) 
        {
            float dTxz_dx = (FDM1*(Txz[i + (j-3)*nzz + k*nxx*nzz] - Txz[i + (j+4)*nzz + k*nxx*nzz]) +
                             FDM2*(Txz[i + (j+3)*nzz + k*nxx*nzz] - Txz[i + (j-2)*nzz + k*nxx*nzz]) +
                             FDM3*(Txz[i + (j-1)*nzz + k*nxx*nzz] - Txz[i + (j+2)*nzz + k*nxx*nzz]) +
                             FDM4*(Txz[i + (j+1)*nzz + k*nxx*nzz] - Txz[i + j*nzz + k*nxx*nzz])) / dx;

            float dTyz_dy = (FDM1*(Tyz[i + j*nzz + (k-3)*nxx*nzz] - Tyz[i + j*nzz + (k+4)*nxx*nzz]) +
                             FDM2*(Tyz[i + j*nzz + (k+3)*nxx*nzz] - Tyz[i + j*nzz + (k-2)*nxx*nzz]) +
                             FDM3*(Tyz[i + j*nzz + (k-1)*nxx*nzz] - Tyz[i + j*nzz + (k+2)*nxx*nzz]) +
                             FDM4*(Tyz[i + j*nzz + (k+1)*nxx*nzz] - Tyz[i + j*nzz + k*nxx*nzz])) / dy;

            float dTzz_dz = (FDM1*(Tzz[(i-4) + j*nzz + k*nxx*nzz] - Tzz[(i+3) + j*nzz + k*nxx*nzz]) +
                             FDM2*(Tzz[(i+2) + j*nzz + k*nxx*nzz] - Tzz[(i-3) + j*nzz + k*nxx*nzz]) +
                             FDM3*(Tzz[(i-2) + j*nzz + k*nxx*nzz] - Tzz[(i+1) + j*nzz + k*nxx*nzz]) +
                             FDM4*(Tzz[i + j*nzz + k*nxx*nzz]     - Tzz[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float Bz = 0.5f*(B[(i+1) + j*nzz + k*nxx*nzz] + B[i + j*nzz + k*nxx*nzz]);

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

__global__ void uintc_compute_pressure_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, 
                                           uintc * C11, uintc * C12, uintc * C13, uintc * C14, uintc * C15, uintc * C16, uintc * C22, uintc * C23, uintc * C24, uintc * C25, uintc * C26, 
                                           uintc * C33, uintc * C34, uintc * C35, uintc * C36, uintc * C44, uintc * C45, uintc * C46, uintc * C55, uintc * C56, uintc * C66, int tId, 
                                           int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz, float minC11, float maxC11, float minC12, float maxC12, 
                                           float minC13, float maxC13, float minC14, float maxC14, float minC15, float maxC15, float minC16, float maxC16, float minC22, float maxC22, 
                                           float minC23, float maxC23, float minC24, float maxC24, float minC25, float maxC25, float minC26, float maxC26, float minC33, float maxC33, 
                                           float minC34, float maxC34, float minC35, float maxC35, float minC36, float maxC36, float minC44, float maxC44, float minC45, float maxC45, 
                                           float minC46, float maxC46, float minC55, float maxC55, float minC56, float maxC56, float minC66, float maxC66, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    float c14_1, c14_2, c14_3, c14_4;
    float c24_1, c24_2, c24_3, c24_4;
    float c34_1, c34_2, c34_3, c34_4;
    float c44_1, c44_2, c44_3, c44_4;

    float c15_1, c15_2, c15_3, c15_4;
    float c25_1, c25_2, c25_3, c25_4;
    float c35_1, c35_2, c35_3, c35_4;
    float c45_1, c45_2, c45_3, c45_4;
    float c55_1, c55_2, c55_3, c55_4;

    float c16_1, c16_2, c16_3, c16_4;
    float c26_1, c26_2, c26_3, c26_4;
    float c36_1, c36_2, c36_3, c36_4;
    float c46_1, c46_2, c46_3, c46_4;
    float c56_1, c56_2, c56_3, c56_4;
    float c66_1, c66_2, c66_3, c66_4;
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4) && (k >= 3) && (k < nyy-4)) 
        {    
            float dVx_dx = (FDM1*(Vx[i + (j-3)*nzz + k*nxx*nzz] - Vx[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+3)*nzz + k*nxx*nzz] - Vx[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-1)*nzz + k*nxx*nzz] - Vx[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + (j+1)*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dx;

            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-3)*nxx*nzz] - Vx[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+3)*nxx*nzz] - Vx[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-1)*nxx*nzz] - Vx[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + (k+1)*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dy;

            float dVx_dz = (FDM1*(Vx[(i-3) + j*nzz + k*nxx*nzz] - Vx[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+3) + j*nzz + k*nxx*nzz] - Vx[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-1) + j*nzz + k*nxx*nzz] - Vx[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[(i+1) + j*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dz;

            float dVy_dx = (FDM1*(Vy[i + (j-3)*nzz + k*nxx*nzz] - Vy[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+3)*nzz + k*nxx*nzz] - Vy[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-1)*nzz + k*nxx*nzz] - Vy[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + (j+1)*nzz + k*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-3)*nxx*nzz] - Vy[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+3)*nxx*nzz] - Vy[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-1)*nxx*nzz] - Vy[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + (k+1)*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dy;

            float dVy_dz = (FDM1*(Vy[(i-3) + j*nzz + k*nxx*nzz] - Vy[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+3) + j*nzz + k*nxx*nzz] - Vy[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-1) + j*nzz + k*nxx*nzz] - Vy[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[(i+1) + j*nzz + k*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-3)*nzz + k*nxx*nzz] - Vz[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+3)*nzz + k*nxx*nzz] - Vz[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-1)*nzz + k*nxx*nzz] - Vz[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + (j+1)*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dx;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-3)*nxx*nzz] - Vz[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+3)*nxx*nzz] - Vz[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-1)*nxx*nzz] - Vz[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + (k+1)*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-3) + j*nzz + k*nxx*nzz] - Vz[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+3) + j*nzz + k*nxx*nzz] - Vz[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-1) + j*nzz + k*nxx*nzz] - Vz[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[(i+1) + j*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dz;
            
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

            Txx[index] += dt*(c11*dVx_dx + c16*dVx_dy + c15*dVx_dz +
                              c16*dVy_dx + c12*dVy_dy + c14*dVy_dz +
                              c15*dVz_dx + c14*dVz_dy + c13*dVz_dz);                    
        
            Tyy[index] += dt*(c12*dVx_dx + c26*dVx_dy + c25*dVx_dz +
                              c26*dVy_dx + c22*dVy_dy + c24*dVy_dz +
                              c25*dVz_dx + c24*dVz_dy + c23*dVz_dz);                    
        
            Tzz[index] += dt*(c13*dVx_dx + c36*dVx_dy + c35*dVx_dz +
                              c36*dVy_dx + c23*dVy_dy + c34*dVy_dz +
                              c35*dVz_dx + c34*dVz_dy + c33*dVz_dz);  

            P[index] = (Txx[index] + Tyy[index] + Tzz[index]) / 3.0f;
        }

        if((i >= 3) && (i < nzz-4) && (j > 3) && (j < nxx-3) && (k > 3) && (k < nyy-3)) 
        {
            float dVx_dx = (FDM1*(Vx[i + (j-4)*nzz + k*nxx*nzz] - Vx[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+2)*nzz + k*nxx*nzz] - Vx[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-2)*nzz + k*nxx*nzz] - Vx[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-4)*nxx*nzz] - Vx[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+2)*nxx*nzz] - Vx[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-2)*nxx*nzz] - Vx[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVx_dz = (FDM1*(Vx[(i-3) + j*nzz + k*nxx*nzz] - Vx[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+3) + j*nzz + k*nxx*nzz] - Vx[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-1) + j*nzz + k*nxx*nzz] - Vx[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[(i+1) + j*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dz;

            float dVy_dx = (FDM1*(Vy[i + (j-4)*nzz + k*nxx*nzz] - Vy[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+2)*nzz + k*nxx*nzz] - Vy[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-2)*nzz + k*nxx*nzz] - Vy[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-4)*nxx*nzz] - Vy[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+2)*nxx*nzz] - Vy[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-2)*nxx*nzz] - Vy[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVy_dz = (FDM1*(Vy[(i-3) + j*nzz + k*nxx*nzz] - Vy[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+3) + j*nzz + k*nxx*nzz] - Vy[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-1) + j*nzz + k*nxx*nzz] - Vy[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[(i+1) + j*nzz + k*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-4)*nzz + k*nxx*nzz] - Vz[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+2)*nzz + k*nxx*nzz] - Vz[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-2)*nzz + k*nxx*nzz] - Vz[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-4)*nxx*nzz] - Vz[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+2)*nxx*nzz] - Vz[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-2)*nxx*nzz] - Vz[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-3) + j*nzz + k*nxx*nzz] - Vz[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+3) + j*nzz + k*nxx*nzz] - Vz[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-1) + j*nzz + k*nxx*nzz] - Vz[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[(i+1) + j*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dz;

            c16_1 = (minC16 + (static_cast<float>(C16[i + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC16 - minC16) / (COMPRESS - 1));
            c16_2 = (minC16 + (static_cast<float>(C16[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC16 - minC16) / (COMPRESS - 1));
            c16_3 = (minC16 + (static_cast<float>(C16[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC16 - minC16) / (COMPRESS - 1));
            c16_4 = (minC16 + (static_cast<float>(C16[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC16 - minC16) / (COMPRESS - 1));

            c26_1 = (minC26 + (static_cast<float>(C26[i + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC26 - minC26) / (COMPRESS - 1));
            c26_2 = (minC26 + (static_cast<float>(C26[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC26 - minC26) / (COMPRESS - 1));
            c26_3 = (minC26 + (static_cast<float>(C26[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC26 - minC26) / (COMPRESS - 1));
            c26_4 = (minC26 + (static_cast<float>(C26[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC26 - minC26) / (COMPRESS - 1));

            c36_1 = (minC36 + (static_cast<float>(C36[i + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC36 - minC36) / (COMPRESS - 1));
            c36_2 = (minC36 + (static_cast<float>(C36[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC36 - minC36) / (COMPRESS - 1));
            c36_3 = (minC36 + (static_cast<float>(C36[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC36 - minC36) / (COMPRESS - 1));
            c36_4 = (minC36 + (static_cast<float>(C36[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC36 - minC36) / (COMPRESS - 1));

            c46_1 = (minC46 + (static_cast<float>(C46[i + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));
            c46_2 = (minC46 + (static_cast<float>(C46[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));
            c46_3 = (minC46 + (static_cast<float>(C46[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));
            c46_4 = (minC46 + (static_cast<float>(C46[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));

            c56_1 = (minC56 + (static_cast<float>(C56[i + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));
            c56_2 = (minC56 + (static_cast<float>(C56[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));
            c56_3 = (minC56 + (static_cast<float>(C56[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));
            c56_4 = (minC56 + (static_cast<float>(C56[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));

            c66_1 = (minC66 + (static_cast<float>(C66[i + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC66 - minC66) / (COMPRESS - 1));
            c66_2 = (minC66 + (static_cast<float>(C66[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC66 - minC66) / (COMPRESS - 1));
            c66_3 = (minC66 + (static_cast<float>(C66[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC66 - minC66) / (COMPRESS - 1));
            c66_4 = (minC66 + (static_cast<float>(C66[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC66 - minC66) / (COMPRESS - 1));

            float c16 = powf(0.25f*(1.0f/c16_1 + 1.0f/c16_2 + 1.0f/c16_3 + 1.0f/c16_4),-1.0f);
            float c26 = powf(0.25f*(1.0f/c26_1 + 1.0f/c26_2 + 1.0f/c26_3 + 1.0f/c26_4),-1.0f);
            float c36 = powf(0.25f*(1.0f/c36_1 + 1.0f/c36_2 + 1.0f/c36_3 + 1.0f/c36_4),-1.0f);
            float c46 = powf(0.25f*(1.0f/c46_1 + 1.0f/c46_2 + 1.0f/c46_3 + 1.0f/c46_4),-1.0f);
            float c56 = powf(0.25f*(1.0f/c56_1 + 1.0f/c56_2 + 1.0f/c56_3 + 1.0f/c56_4),-1.0f);
            float c66 = powf(0.25f*(1.0f/c66_1 + 1.0f/c66_2 + 1.0f/c66_3 + 1.0f/c66_4),-1.0f);

            Txy[index] += dt*(c16*dVx_dx + c66*dVx_dy + c56*dVx_dz +
                              c66*dVy_dx + c26*dVy_dy + c46*dVy_dz +
                              c56*dVz_dx + c46*dVz_dy + c36*dVz_dz);                    
        }

        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3) && (k >= 3) && (k < nyy-4)) 
        {
            float dVx_dx = (FDM1*(Vx[i + (j-4)*nzz + k*nxx*nzz] - Vx[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+2)*nzz + k*nxx*nzz] - Vx[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-2)*nzz + k*nxx*nzz] - Vx[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-3)*nxx*nzz] - Vx[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+3)*nxx*nzz] - Vx[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-1)*nxx*nzz] - Vx[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + (k+1)*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dy;

            float dVx_dz = (FDM1*(Vx[(i-4) + j*nzz + k*nxx*nzz] - Vx[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+2) + j*nzz + k*nxx*nzz] - Vx[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-2) + j*nzz + k*nxx*nzz] - Vx[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVy_dx = (FDM1*(Vy[i + (j-4)*nzz + k*nxx*nzz] - Vy[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+2)*nzz + k*nxx*nzz] - Vy[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-2)*nzz + k*nxx*nzz] - Vy[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-3)*nxx*nzz] - Vy[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+3)*nxx*nzz] - Vy[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-1)*nxx*nzz] - Vy[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + (k+1)*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dy;

            float dVy_dz = (FDM1*(Vy[(i-4) + j*nzz + k*nxx*nzz] - Vy[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+2) + j*nzz + k*nxx*nzz] - Vy[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-2) + j*nzz + k*nxx*nzz] - Vy[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-4)*nzz + k*nxx*nzz] - Vz[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+2)*nzz + k*nxx*nzz] - Vz[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-2)*nzz + k*nxx*nzz] - Vz[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-3)*nxx*nzz] - Vz[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+3)*nxx*nzz] - Vz[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-1)*nxx*nzz] - Vz[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + (k+1)*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-4) + j*nzz + k*nxx*nzz] - Vz[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+2) + j*nzz + k*nxx*nzz] - Vz[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-2) + j*nzz + k*nxx*nzz] - Vz[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            c15_1 = (minC15 + (static_cast<float>(C15[(i+1) + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            c15_2 = (minC15 + (static_cast<float>(C15[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            c15_3 = (minC15 + (static_cast<float>(C15[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            c15_4 = (minC15 + (static_cast<float>(C15[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));

            c25_1 = (minC25 + (static_cast<float>(C25[(i+1) + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC25 - minC25) / (COMPRESS - 1));
            c25_2 = (minC25 + (static_cast<float>(C25[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC25 - minC25) / (COMPRESS - 1));
            c25_3 = (minC25 + (static_cast<float>(C25[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC25 - minC25) / (COMPRESS - 1));
            c25_4 = (minC25 + (static_cast<float>(C25[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC25 - minC25) / (COMPRESS - 1));

            c35_1 = (minC35 + (static_cast<float>(C35[(i+1) + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));
            c35_2 = (minC35 + (static_cast<float>(C35[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));
            c35_3 = (minC35 + (static_cast<float>(C35[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));
            c35_4 = (minC35 + (static_cast<float>(C35[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));

            c45_1 = (minC45 + (static_cast<float>(C45[(i+1) + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));
            c45_2 = (minC45 + (static_cast<float>(C45[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));
            c45_3 = (minC45 + (static_cast<float>(C45[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));
            c45_4 = (minC45 + (static_cast<float>(C45[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));

            c55_1 = (minC55 + (static_cast<float>(C55[(i+1) + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_2 = (minC55 + (static_cast<float>(C55[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_3 = (minC55 + (static_cast<float>(C55[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_4 = (minC55 + (static_cast<float>(C55[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            c56_1 = (minC56 + (static_cast<float>(C56[(i+1) + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));
            c56_2 = (minC56 + (static_cast<float>(C56[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));
            c56_3 = (minC56 + (static_cast<float>(C56[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));
            c56_4 = (minC56 + (static_cast<float>(C56[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC56 - minC56) / (COMPRESS - 1));

            float c15 = powf(0.25f*(1.0f/c15_1 + 1.0f/c15_2 + 1.0f/c15_3 + 1.0f/c15_4),-1.0f);
            float c25 = powf(0.25f*(1.0f/c25_1 + 1.0f/c25_2 + 1.0f/c25_3 + 1.0f/c25_4),-1.0f);
            float c35 = powf(0.25f*(1.0f/c35_1 + 1.0f/c35_2 + 1.0f/c35_3 + 1.0f/c35_4),-1.0f);
            float c45 = powf(0.25f*(1.0f/c45_1 + 1.0f/c45_2 + 1.0f/c45_3 + 1.0f/c45_4),-1.0f);
            float c55 = powf(0.25f*(1.0f/c55_1 + 1.0f/c55_2 + 1.0f/c55_3 + 1.0f/c55_4),-1.0f);
            float c56 = powf(0.25f*(1.0f/c56_1 + 1.0f/c56_2 + 1.0f/c56_3 + 1.0f/c56_4),-1.0f);

            Txz[index] += dt*(c15*dVx_dx + c56*dVx_dy + c55*dVx_dz +
                              c56*dVy_dx + c25*dVy_dy + c45*dVy_dz +
                              c55*dVz_dx + c45*dVz_dy + c35*dVz_dz);                    
        }

        if((i > 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4) && (k > 3) && (k < nyy-3)) 
        {
            float dVx_dx = (FDM1*(Vx[i + (j-3)*nzz + k*nxx*nzz] - Vx[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+3)*nzz + k*nxx*nzz] - Vx[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-1)*nzz + k*nxx*nzz] - Vx[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + (j+1)*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dx;

            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-4)*nxx*nzz] - Vx[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+2)*nxx*nzz] - Vx[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-2)*nxx*nzz] - Vx[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVx_dz = (FDM1*(Vx[(i-4) + j*nzz + k*nxx*nzz] - Vx[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+2) + j*nzz + k*nxx*nzz] - Vx[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-2) + j*nzz + k*nxx*nzz] - Vx[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVy_dx = (FDM1*(Vy[i + (j-3)*nzz + k*nxx*nzz] - Vy[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+3)*nzz + k*nxx*nzz] - Vy[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-1)*nzz + k*nxx*nzz] - Vy[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + (j+1)*nzz + k*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-4)*nxx*nzz] - Vy[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+2)*nxx*nzz] - Vy[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-2)*nxx*nzz] - Vy[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVy_dz = (FDM1*(Vy[(i-4) + j*nzz + k*nxx*nzz] - Vy[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+2) + j*nzz + k*nxx*nzz] - Vy[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-2) + j*nzz + k*nxx*nzz] - Vy[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-3)*nzz + k*nxx*nzz] - Vz[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+3)*nzz + k*nxx*nzz] - Vz[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-1)*nzz + k*nxx*nzz] - Vz[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + (j+1)*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dx;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-4)*nxx*nzz] - Vz[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+2)*nxx*nzz] - Vz[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-2)*nxx*nzz] - Vz[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-4) + j*nzz + k*nxx*nzz] - Vz[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+2) + j*nzz + k*nxx*nzz] - Vz[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-2) + j*nzz + k*nxx*nzz] - Vz[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[(i-1) + j*nzz + k*nxx*nzz])) / dz;
                            
            c14_1 = (minC14 + (static_cast<float>(C14[(i+1) + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC14 - minC14) / (COMPRESS - 1));
            c14_2 = (minC14 + (static_cast<float>(C14[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC14 - minC14) / (COMPRESS - 1));
            c14_3 = (minC14 + (static_cast<float>(C14[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC14 - minC14) / (COMPRESS - 1));
            c14_4 = (minC14 + (static_cast<float>(C14[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC14 - minC14) / (COMPRESS - 1));

            c24_1 = (minC24 + (static_cast<float>(C24[(i+1) + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC24 - minC24) / (COMPRESS - 1));
            c24_2 = (minC24 + (static_cast<float>(C24[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC24 - minC24) / (COMPRESS - 1));
            c24_3 = (minC24 + (static_cast<float>(C24[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC24 - minC24) / (COMPRESS - 1));
            c24_4 = (minC24 + (static_cast<float>(C24[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC24 - minC24) / (COMPRESS - 1));

            c34_1 = (minC34 + (static_cast<float>(C34[(i+1) + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC34 - minC34) / (COMPRESS - 1));
            c34_2 = (minC34 + (static_cast<float>(C34[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC34 - minC34) / (COMPRESS - 1));
            c34_3 = (minC34 + (static_cast<float>(C34[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC34 - minC34) / (COMPRESS - 1));
            c34_4 = (minC34 + (static_cast<float>(C34[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC34 - minC34) / (COMPRESS - 1));

            c44_1 = (minC44 + (static_cast<float>(C44[(i+1) + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC44 - minC44) / (COMPRESS - 1));
            c44_2 = (minC44 + (static_cast<float>(C44[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC44 - minC44) / (COMPRESS - 1));
            c44_3 = (minC44 + (static_cast<float>(C44[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC44 - minC44) / (COMPRESS - 1));
            c44_4 = (minC44 + (static_cast<float>(C44[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC44 - minC44) / (COMPRESS - 1));

            c45_1 = (minC45 + (static_cast<float>(C45[(i+1) + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));
            c45_2 = (minC45 + (static_cast<float>(C45[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));
            c45_3 = (minC45 + (static_cast<float>(C45[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));
            c45_4 = (minC45 + (static_cast<float>(C45[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC45 - minC45) / (COMPRESS - 1));

            c46_1 = (minC46 + (static_cast<float>(C46[(i+1) + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));
            c46_2 = (minC46 + (static_cast<float>(C46[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));
            c46_3 = (minC46 + (static_cast<float>(C46[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));
            c46_4 = (minC46 + (static_cast<float>(C46[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC46 - minC46) / (COMPRESS - 1));

            float c14 = powf(0.25f*(1.0f/c14_1 + 1.0f/c14_2 + 1.0f/c14_3 + 1.0f/c14_4),-1.0f);
            float c24 = powf(0.25f*(1.0f/c24_1 + 1.0f/c24_2 + 1.0f/c24_3 + 1.0f/c24_4),-1.0f);
            float c34 = powf(0.25f*(1.0f/c34_1 + 1.0f/c34_2 + 1.0f/c34_3 + 1.0f/c34_4),-1.0f);
            float c44 = powf(0.25f*(1.0f/c44_1 + 1.0f/c44_2 + 1.0f/c44_3 + 1.0f/c44_4),-1.0f);
            float c45 = powf(0.25f*(1.0f/c45_1 + 1.0f/c45_2 + 1.0f/c45_3 + 1.0f/c45_4),-1.0f);
            float c46 = powf(0.25f*(1.0f/c46_1 + 1.0f/c46_2 + 1.0f/c46_3 + 1.0f/c46_4),-1.0f);

            Tyz[index] += dt*(c14*dVx_dx + c46*dVx_dy + c45*dVx_dz +
                              c46*dVy_dx + c24*dVy_dy + c44*dVy_dz +
                              c45*dVz_dx + c44*dVz_dy + c34*dVz_dz); 
        }
    }
}

__global__ void float_compute_pressure_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, 
                                           float * C11, float * C12, float * C13, float * C14, float * C15, float * C16, float * C22, float * C23, float * C24, float * C25, float * C26, 
                                           float * C33, float * C34, float * C35, float * C36, float * C44, float * C45, float * C46, float * C55, float * C56, float * C66, int tId, 
                                           int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    float c14_1, c14_2, c14_3, c14_4;
    float c24_1, c24_2, c24_3, c24_4;
    float c34_1, c34_2, c34_3, c34_4;
    float c44_1, c44_2, c44_3, c44_4;

    float c15_1, c15_2, c15_3, c15_4;
    float c25_1, c25_2, c25_3, c25_4;
    float c35_1, c35_2, c35_3, c35_4;
    float c45_1, c45_2, c45_3, c45_4;
    float c55_1, c55_2, c55_3, c55_4;

    float c16_1, c16_2, c16_3, c16_4;
    float c26_1, c26_2, c26_3, c26_4;
    float c36_1, c36_2, c36_3, c36_4;
    float c46_1, c46_2, c46_3, c46_4;
    float c56_1, c56_2, c56_3, c56_4;
    float c66_1, c66_2, c66_3, c66_4;
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4) && (k >= 3) && (k < nyy-4)) 
        {    
            float dVx_dx = (FDM1*(Vx[i + (j-3)*nzz + k*nxx*nzz] - Vx[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+3)*nzz + k*nxx*nzz] - Vx[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-1)*nzz + k*nxx*nzz] - Vx[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + (j+1)*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dx;

            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-3)*nxx*nzz] - Vx[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+3)*nxx*nzz] - Vx[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-1)*nxx*nzz] - Vx[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + (k+1)*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dy;

            float dVx_dz = (FDM1*(Vx[(i-3) + j*nzz + k*nxx*nzz] - Vx[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+3) + j*nzz + k*nxx*nzz] - Vx[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-1) + j*nzz + k*nxx*nzz] - Vx[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[(i+1) + j*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dz;

            float dVy_dx = (FDM1*(Vy[i + (j-3)*nzz + k*nxx*nzz] - Vy[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+3)*nzz + k*nxx*nzz] - Vy[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-1)*nzz + k*nxx*nzz] - Vy[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + (j+1)*nzz + k*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-3)*nxx*nzz] - Vy[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+3)*nxx*nzz] - Vy[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-1)*nxx*nzz] - Vy[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + (k+1)*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dy;

            float dVy_dz = (FDM1*(Vy[(i-3) + j*nzz + k*nxx*nzz] - Vy[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+3) + j*nzz + k*nxx*nzz] - Vy[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-1) + j*nzz + k*nxx*nzz] - Vy[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[(i+1) + j*nzz + k*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-3)*nzz + k*nxx*nzz] - Vz[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+3)*nzz + k*nxx*nzz] - Vz[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-1)*nzz + k*nxx*nzz] - Vz[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + (j+1)*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dx;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-3)*nxx*nzz] - Vz[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+3)*nxx*nzz] - Vz[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-1)*nxx*nzz] - Vz[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + (k+1)*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-3) + j*nzz + k*nxx*nzz] - Vz[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+3) + j*nzz + k*nxx*nzz] - Vz[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-1) + j*nzz + k*nxx*nzz] - Vz[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[(i+1) + j*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dz;
            
            Txx[index] += dt*(C11[index]*dVx_dx + C16[index]*dVx_dy + C15[index]*dVx_dz +
                              C16[index]*dVy_dx + C12[index]*dVy_dy + C14[index]*dVy_dz +
                              C15[index]*dVz_dx + C14[index]*dVz_dy + C13[index]*dVz_dz);                    
        
            Tyy[index] += dt*(C12[index]*dVx_dx + C26[index]*dVx_dy + C25[index]*dVx_dz +
                              C26[index]*dVy_dx + C22[index]*dVy_dy + C24[index]*dVy_dz +
                              C25[index]*dVz_dx + C24[index]*dVz_dy + C23[index]*dVz_dz);                    
        
            Tzz[index] += dt*(C13[index]*dVx_dx + C36[index]*dVx_dy + C35[index]*dVx_dz +
                              C36[index]*dVy_dx + C23[index]*dVy_dy + C34[index]*dVy_dz +
                              C35[index]*dVz_dx + C34[index]*dVz_dy + C33[index]*dVz_dz);  
        
            P[index] = (Txx[index] + Tyy[index] + Tzz[index]) / 3.0f;        
        }

        if((i >= 3) && (i < nzz-4) && (j > 3) && (j < nxx-3) && (k > 3) && (k < nyy-3)) 
        {
            float dVx_dx = (FDM1*(Vx[i + (j-4)*nzz + k*nxx*nzz] - Vx[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+2)*nzz + k*nxx*nzz] - Vx[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-2)*nzz + k*nxx*nzz] - Vx[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-4)*nxx*nzz] - Vx[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+2)*nxx*nzz] - Vx[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-2)*nxx*nzz] - Vx[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVx_dz = (FDM1*(Vx[(i-3) + j*nzz + k*nxx*nzz] - Vx[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+3) + j*nzz + k*nxx*nzz] - Vx[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-1) + j*nzz + k*nxx*nzz] - Vx[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[(i+1) + j*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dz;

            float dVy_dx = (FDM1*(Vy[i + (j-4)*nzz + k*nxx*nzz] - Vy[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+2)*nzz + k*nxx*nzz] - Vy[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-2)*nzz + k*nxx*nzz] - Vy[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-4)*nxx*nzz] - Vy[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+2)*nxx*nzz] - Vy[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-2)*nxx*nzz] - Vy[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVy_dz = (FDM1*(Vy[(i-3) + j*nzz + k*nxx*nzz] - Vy[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+3) + j*nzz + k*nxx*nzz] - Vy[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-1) + j*nzz + k*nxx*nzz] - Vy[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[(i+1) + j*nzz + k*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-4)*nzz + k*nxx*nzz] - Vz[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+2)*nzz + k*nxx*nzz] - Vz[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-2)*nzz + k*nxx*nzz] - Vz[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-4)*nxx*nzz] - Vz[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+2)*nxx*nzz] - Vz[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-2)*nxx*nzz] - Vz[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-3) + j*nzz + k*nxx*nzz] - Vz[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+3) + j*nzz + k*nxx*nzz] - Vz[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-1) + j*nzz + k*nxx*nzz] - Vz[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[(i+1) + j*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dz;

            c16_1 = C16[i + (j+1)*nzz + (k+1)*nxx*nzz];
            c16_2 = C16[i + (j+1)*nzz + k*nxx*nzz];
            c16_3 = C16[i + j*nzz + (k+1)*nxx*nzz];
            c16_4 = C16[i + j*nzz + k*nxx*nzz];

            c26_1 = C26[i + (j+1)*nzz + (k+1)*nxx*nzz];
            c26_2 = C26[i + (j+1)*nzz + k*nxx*nzz];
            c26_3 = C26[i + j*nzz + (k+1)*nxx*nzz];
            c26_4 = C26[i + j*nzz + k*nxx*nzz];

            c36_1 = C36[i + (j+1)*nzz + (k+1)*nxx*nzz];
            c36_2 = C36[i + (j+1)*nzz + k*nxx*nzz];
            c36_3 = C36[i + j*nzz + (k+1)*nxx*nzz];
            c36_4 = C36[i + j*nzz + k*nxx*nzz];

            c46_1 = C46[i + (j+1)*nzz + (k+1)*nxx*nzz];
            c46_2 = C46[i + (j+1)*nzz + k*nxx*nzz];
            c46_3 = C46[i + j*nzz + (k+1)*nxx*nzz];
            c46_4 = C46[i + j*nzz + k*nxx*nzz];

            c56_1 = C56[i + (j+1)*nzz + (k+1)*nxx*nzz];
            c56_2 = C56[i + (j+1)*nzz + k*nxx*nzz];
            c56_3 = C56[i + j*nzz + (k+1)*nxx*nzz];
            c56_4 = C56[i + j*nzz + k*nxx*nzz];

            c66_1 = C66[i + (j+1)*nzz + (k+1)*nxx*nzz];
            c66_2 = C66[i + (j+1)*nzz + k*nxx*nzz];
            c66_3 = C66[i + j*nzz + (k+1)*nxx*nzz];
            c66_4 = C66[i + j*nzz + k*nxx*nzz];

            float c16 = powf(0.25f*(1.0f/c16_1 + 1.0f/c16_2 + 1.0f/c16_3 + 1.0f/c16_4),-1.0f);
            float c26 = powf(0.25f*(1.0f/c26_1 + 1.0f/c26_2 + 1.0f/c26_3 + 1.0f/c26_4),-1.0f);
            float c36 = powf(0.25f*(1.0f/c36_1 + 1.0f/c36_2 + 1.0f/c36_3 + 1.0f/c36_4),-1.0f);
            float c46 = powf(0.25f*(1.0f/c46_1 + 1.0f/c46_2 + 1.0f/c46_3 + 1.0f/c46_4),-1.0f);
            float c56 = powf(0.25f*(1.0f/c56_1 + 1.0f/c56_2 + 1.0f/c56_3 + 1.0f/c56_4),-1.0f);
            float c66 = powf(0.25f*(1.0f/c66_1 + 1.0f/c66_2 + 1.0f/c66_3 + 1.0f/c66_4),-1.0f);

            Txy[index] += dt*(c16*dVx_dx + c66*dVx_dy + c56*dVx_dz +
                              c66*dVy_dx + c26*dVy_dy + c46*dVy_dz +
                              c56*dVz_dx + c46*dVz_dy + c36*dVz_dz);                    
        }

        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3) && (k >= 3) && (k < nyy-4)) 
        {
            float dVx_dx = (FDM1*(Vx[i + (j-4)*nzz + k*nxx*nzz] - Vx[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+2)*nzz + k*nxx*nzz] - Vx[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-2)*nzz + k*nxx*nzz] - Vx[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-3)*nxx*nzz] - Vx[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+3)*nxx*nzz] - Vx[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-1)*nxx*nzz] - Vx[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + (k+1)*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dy;

            float dVx_dz = (FDM1*(Vx[(i-4) + j*nzz + k*nxx*nzz] - Vx[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+2) + j*nzz + k*nxx*nzz] - Vx[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-2) + j*nzz + k*nxx*nzz] - Vx[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVy_dx = (FDM1*(Vy[i + (j-4)*nzz + k*nxx*nzz] - Vy[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+2)*nzz + k*nxx*nzz] - Vy[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-2)*nzz + k*nxx*nzz] - Vy[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-3)*nxx*nzz] - Vy[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+3)*nxx*nzz] - Vy[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-1)*nxx*nzz] - Vy[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + (k+1)*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dy;

            float dVy_dz = (FDM1*(Vy[(i-4) + j*nzz + k*nxx*nzz] - Vy[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+2) + j*nzz + k*nxx*nzz] - Vy[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-2) + j*nzz + k*nxx*nzz] - Vy[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-4)*nzz + k*nxx*nzz] - Vz[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+2)*nzz + k*nxx*nzz] - Vz[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-2)*nzz + k*nxx*nzz] - Vz[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-3)*nxx*nzz] - Vz[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+3)*nxx*nzz] - Vz[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-1)*nxx*nzz] - Vz[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + (k+1)*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-4) + j*nzz + k*nxx*nzz] - Vz[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+2) + j*nzz + k*nxx*nzz] - Vz[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-2) + j*nzz + k*nxx*nzz] - Vz[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            c15_1 = C15[(i+1) + (j+1)*nzz + k*nxx*nzz];
            c15_2 = C15[i + (j+1)*nzz + k*nxx*nzz];
            c15_3 = C15[(i+1) + j*nzz + k*nxx*nzz];
            c15_4 = C15[i + j*nzz + k*nxx*nzz];

            c25_1 = C25[(i+1) + (j+1)*nzz + k*nxx*nzz];
            c25_2 = C25[i + (j+1)*nzz + k*nxx*nzz];
            c25_3 = C25[(i+1) + j*nzz + k*nxx*nzz];
            c25_4 = C25[i + j*nzz + k*nxx*nzz];

            c35_1 = C35[(i+1) + (j+1)*nzz + k*nxx*nzz];
            c35_2 = C35[i + (j+1)*nzz + k*nxx*nzz];
            c35_3 = C35[(i+1) + j*nzz + k*nxx*nzz];
            c35_4 = C35[i + j*nzz + k*nxx*nzz];

            c45_1 = C45[(i+1) + (j+1)*nzz + k*nxx*nzz];
            c45_2 = C45[i + (j+1)*nzz + k*nxx*nzz];
            c45_3 = C45[(i+1) + j*nzz + k*nxx*nzz];
            c45_4 = C45[i + j*nzz + k*nxx*nzz];

            c55_1 = C55[(i+1) + (j+1)*nzz + k*nxx*nzz];
            c55_2 = C55[i + (j+1)*nzz + k*nxx*nzz];
            c55_3 = C55[(i+1) + j*nzz + k*nxx*nzz];
            c55_4 = C55[i + j*nzz + k*nxx*nzz];

            c56_1 = C56[(i+1) + (j+1)*nzz + k*nxx*nzz];
            c56_2 = C56[i + (j+1)*nzz + k*nxx*nzz];
            c56_3 = C56[(i+1) + j*nzz + k*nxx*nzz];
            c56_4 = C56[i + j*nzz + k*nxx*nzz];

            float c15 = powf(0.25f*(1.0f/c15_1 + 1.0f/c15_2 + 1.0f/c15_3 + 1.0f/c15_4),-1.0f);
            float c25 = powf(0.25f*(1.0f/c25_1 + 1.0f/c25_2 + 1.0f/c25_3 + 1.0f/c25_4),-1.0f);
            float c35 = powf(0.25f*(1.0f/c35_1 + 1.0f/c35_2 + 1.0f/c35_3 + 1.0f/c35_4),-1.0f);
            float c45 = powf(0.25f*(1.0f/c45_1 + 1.0f/c45_2 + 1.0f/c45_3 + 1.0f/c45_4),-1.0f);
            float c55 = powf(0.25f*(1.0f/c55_1 + 1.0f/c55_2 + 1.0f/c55_3 + 1.0f/c55_4),-1.0f);
            float c56 = powf(0.25f*(1.0f/c56_1 + 1.0f/c56_2 + 1.0f/c56_3 + 1.0f/c56_4),-1.0f);

            Txz[index] += dt*(c15*dVx_dx + c56*dVx_dy + c55*dVx_dz +
                              c56*dVy_dx + c25*dVy_dy + c45*dVy_dz +
                              c55*dVz_dx + c45*dVz_dy + c35*dVz_dz);                    
        }

        if((i > 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4) && (k > 3) && (k < nyy-3)) 
        {
            float dVx_dx = (FDM1*(Vx[i + (j-3)*nzz + k*nxx*nzz] - Vx[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+3)*nzz + k*nxx*nzz] - Vx[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-1)*nzz + k*nxx*nzz] - Vx[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + (j+1)*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dx;

            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-4)*nxx*nzz] - Vx[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+2)*nxx*nzz] - Vx[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-2)*nxx*nzz] - Vx[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVx_dz = (FDM1*(Vx[(i-4) + j*nzz + k*nxx*nzz] - Vx[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+2) + j*nzz + k*nxx*nzz] - Vx[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-2) + j*nzz + k*nxx*nzz] - Vx[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVy_dx = (FDM1*(Vy[i + (j-3)*nzz + k*nxx*nzz] - Vy[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+3)*nzz + k*nxx*nzz] - Vy[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-1)*nzz + k*nxx*nzz] - Vy[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + (j+1)*nzz + k*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-4)*nxx*nzz] - Vy[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+2)*nxx*nzz] - Vy[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-2)*nxx*nzz] - Vy[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVy_dz = (FDM1*(Vy[(i-4) + j*nzz + k*nxx*nzz] - Vy[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+2) + j*nzz + k*nxx*nzz] - Vy[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-2) + j*nzz + k*nxx*nzz] - Vy[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-3)*nzz + k*nxx*nzz] - Vz[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+3)*nzz + k*nxx*nzz] - Vz[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-1)*nzz + k*nxx*nzz] - Vz[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + (j+1)*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dx;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-4)*nxx*nzz] - Vz[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+2)*nxx*nzz] - Vz[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-2)*nxx*nzz] - Vz[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-4) + j*nzz + k*nxx*nzz] - Vz[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+2) + j*nzz + k*nxx*nzz] - Vz[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-2) + j*nzz + k*nxx*nzz] - Vz[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[(i-1) + j*nzz + k*nxx*nzz])) / dz;
                            
            c14_1 = C14[(i+1) + j*nzz + (k+1)*nxx*nzz];
            c14_2 = C14[i + j*nzz + (k+1)*nxx*nzz];
            c14_3 = C14[(i+1) + j*nzz + k*nxx*nzz];
            c14_4 = C14[i + j*nzz + k*nxx*nzz];

            c24_1 = C24[(i+1) + j*nzz + (k+1)*nxx*nzz];
            c24_2 = C24[i + j*nzz + (k+1)*nxx*nzz];
            c24_3 = C24[(i+1) + j*nzz + k*nxx*nzz];
            c24_4 = C24[i + j*nzz + k*nxx*nzz];

            c34_1 = C34[(i+1) + j*nzz + (k+1)*nxx*nzz];
            c34_2 = C34[i + j*nzz + (k+1)*nxx*nzz];
            c34_3 = C34[(i+1) + j*nzz + k*nxx*nzz];
            c34_4 = C34[i + j*nzz + k*nxx*nzz];

            c44_1 = C44[(i+1) + j*nzz + (k+1)*nxx*nzz];
            c44_2 = C44[i + j*nzz + (k+1)*nxx*nzz];
            c44_3 = C44[(i+1) + j*nzz + k*nxx*nzz];
            c44_4 = C44[i + j*nzz + k*nxx*nzz];

            c45_1 = C45[(i+1) + j*nzz + (k+1)*nxx*nzz];
            c45_2 = C45[i + j*nzz + (k+1)*nxx*nzz];
            c45_3 = C45[(i+1) + j*nzz + k*nxx*nzz];
            c45_4 = C45[i + j*nzz + k*nxx*nzz];

            c46_1 = C46[(i+1) + j*nzz + (k+1)*nxx*nzz];
            c46_2 = C46[i + j*nzz + (k+1)*nxx*nzz];
            c46_3 = C46[(i+1) + j*nzz + k*nxx*nzz];
            c46_4 = C46[i + j*nzz + k*nxx*nzz];

            float c14 = powf(0.25f*(1.0f/c14_1 + 1.0f/c14_2 + 1.0f/c14_3 + 1.0f/c14_4),-1.0f);
            float c24 = powf(0.25f*(1.0f/c24_1 + 1.0f/c24_2 + 1.0f/c24_3 + 1.0f/c24_4),-1.0f);
            float c34 = powf(0.25f*(1.0f/c34_1 + 1.0f/c34_2 + 1.0f/c34_3 + 1.0f/c34_4),-1.0f);
            float c44 = powf(0.25f*(1.0f/c44_1 + 1.0f/c44_2 + 1.0f/c44_3 + 1.0f/c44_4),-1.0f);
            float c45 = powf(0.25f*(1.0f/c45_1 + 1.0f/c45_2 + 1.0f/c45_3 + 1.0f/c45_4),-1.0f);
            float c46 = powf(0.25f*(1.0f/c46_1 + 1.0f/c46_2 + 1.0f/c46_3 + 1.0f/c46_4),-1.0f);

            Tyz[index] += dt*(c14*dVx_dx + c46*dVx_dy + c45*dVx_dz +
                              c46*dVy_dx + c24*dVy_dy + c44*dVy_dz +
                              c45*dVz_dx + c44*dVz_dy + c34*dVz_dz); 
        }
    }
}