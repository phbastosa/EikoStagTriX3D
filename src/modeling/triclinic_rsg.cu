# include "triclinic_rsg.cuh"

void Triclinic_RSG::initialization()
{
    modeling_name = "Triclinic media with Rotated Staggered Grid";
    modeling_type = "triclinic_rsg";

    float * h_skw = new float[DGS*DGS*DGS]();

    auto skw = gaussian_weights(sx, sy, sz, sIdx, sIdy, sIdz, dx, dy, dz);

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
        auto rkwVx = kaiser_weights(rx + 0.5f*dx, ry + 0.5f*dy, rz + 0.5f*dz, rIdx, rIdy, rIdz, dx, dy, dz);
        auto rkwVy = kaiser_weights(rx + 0.5f*dx, ry + 0.5f*dy, rz + 0.5f*dz, rIdx, rIdy, rIdz, dx, dy, dz);
        auto rkwVz = kaiser_weights(rx + 0.5f*dx, ry + 0.5f*dy, rz + 0.5f*dz, rIdx, rIdy, rIdz, dx, dy, dz);
        
        for (int zId = 0; zId < DGS; zId++)
        {
            for (int xId = 0; xId < DGS; xId++)
            {
                for (int kId = 0; kId < DGS; kId++)
                {
                    h_rkwPs[zId + xId*DGS + kId*DGS*DGS + spreadId*DGS*DGS*DGS] = rkwPs[zId][xId][kId];
                    h_rkwVx[zId + xId*DGS + kId*DGS*DGS + spreadId*DGS*DGS*DGS] = rkwVx[zId][xId][kId];
                    h_rkwVy[zId + xId*DGS + kId*DGS*DGS + spreadId*DGS*DGS*DGS] = rkwVy[zId][xId][kId];
                    h_rkwVz[zId + xId*DGS + kId*DGS*DGS + spreadId*DGS*DGS*DGS] = rkwVz[zId][xId][kId];
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

void Triclinic_RSG::compute_velocity()
{
    if (compression)
    {
        uintc_compute_velocity_rsg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_T,dc_B,
                                                         maxB,minB,d1D,d2D,d3D,d_wavelet,dx,dy,dz,dt,timeId,tlag,sIdx, 
                                                         sIdy,sIdz,d_skw,nxx,nyy,nzz,nb,nt,eikonalClip);
    }
    else 
    {
        float_compute_velocity_rsg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_T,d_B,
                                                         d1D,d2D,d3D,d_wavelet,dx,dy,dz,dt,timeId,tlag,sIdx,sIdy,sIdz,
                                                         d_skw,nxx,nyy,nzz,nb,nt,eikonalClip);
    }
}

void Triclinic_RSG::compute_pressure()
{
    if (compression)
    {
        uintc_compute_pressure_rsg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_P,d_T, 
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
        float_compute_pressure_rsg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vy,d_Vz,d_Txx,d_Tyy,d_Tzz,d_Txz,d_Tyz,d_Txy,d_P,d_T, 
                                                         d_C11,d_C12,d_C13,d_C14,d_C15,d_C16,d_C22,d_C23,d_C24,d_C25,
                                                         d_C26,d_C33,d_C34,d_C35,d_C36,d_C44,d_C45,d_C46,d_C55,d_C56, 
                                                         d_C66,timeId,tlag,dx,dy,dz,dt,nxx,nyy,nzz,eikonalClip);
    }
}

__global__ void uintc_compute_velocity_rsg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, uintc * B,
                                           float maxB, float minB, float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, 
                                           int tlag, int sIdx, int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt, bool eikonal)
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

    float d1_Txx = 0.0f; float d2_Txx = 0.0f; float d3_Txx = 0.0f; float d4_Txx = 0.0f;
    float d1_Tyy = 0.0f; float d2_Tyy = 0.0f; float d3_Tyy = 0.0f; float d4_Tyy = 0.0f;
    float d1_Tzz = 0.0f; float d2_Tzz = 0.0f; float d3_Tzz = 0.0f; float d4_Tzz = 0.0f;
    float d1_Txy = 0.0f; float d2_Txy = 0.0f; float d3_Txy = 0.0f; float d4_Txy = 0.0f;
    float d1_Txz = 0.0f; float d2_Txz = 0.0f; float d3_Txz = 0.0f; float d4_Txz = 0.0f;
    float d1_Tyz = 0.0f; float d2_Tyz = 0.0f; float d3_Tyz = 0.0f; float d4_Tyz = 0.0f;            
 
    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4) && (k >= 3) && (k < nyy-4)) 
        {   
            # pragma unroll 4 
            for (int rsg = 0; rsg < 4; rsg++)
            {
                d1_Txx += FDM[rsg]*(Txx[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txx[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Tyy += FDM[rsg]*(Tyy[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tyy[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Tzz += FDM[rsg]*(Tzz[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tzz[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Txy += FDM[rsg]*(Txy[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txy[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Txz += FDM[rsg]*(Txz[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txz[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Tyz += FDM[rsg]*(Tyz[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tyz[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);

                d2_Txx += FDM[rsg]*(Txx[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txx[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Tyy += FDM[rsg]*(Tyy[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tyy[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Tzz += FDM[rsg]*(Tzz[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tzz[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Txy += FDM[rsg]*(Txy[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txy[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Txz += FDM[rsg]*(Txz[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txz[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Tyz += FDM[rsg]*(Tyz[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tyz[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);

                d3_Txx += FDM[rsg]*(Txx[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txx[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Tyy += FDM[rsg]*(Tyy[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tyy[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Tzz += FDM[rsg]*(Tzz[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tzz[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Txy += FDM[rsg]*(Txy[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txy[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Txz += FDM[rsg]*(Txz[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txz[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Tyz += FDM[rsg]*(Tyz[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tyz[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);

                d4_Txx += FDM[rsg]*(Txx[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txx[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Tyy += FDM[rsg]*(Tyy[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tyy[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Tzz += FDM[rsg]*(Tzz[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tzz[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Txy += FDM[rsg]*(Txy[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txy[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Txz += FDM[rsg]*(Txz[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txz[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Tyz += FDM[rsg]*(Tyz[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tyz[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
            }
    
            float dTxx_dx = 0.25f*(d1_Txx + d2_Txx + d3_Txx + d4_Txx) / dx;
            float dTxy_dx = 0.25f*(d1_Txy + d2_Txy + d3_Txy + d4_Txy) / dx;
            float dTxz_dx = 0.25f*(d1_Txz + d2_Txz + d3_Txz + d4_Txz) / dx;
        
            float dTxy_dy = 0.25f*(d1_Txy + d2_Txy - d3_Txy - d4_Txy) / dy;
            float dTyy_dy = 0.25f*(d1_Tyy + d2_Tyy - d3_Tyy - d4_Tyy) / dy;
            float dTyz_dy = 0.25f*(d1_Tyz + d2_Tyz - d3_Tyz - d4_Tyz) / dy;
            
            float dTxz_dz = 0.25f*(d1_Txz - d2_Txz + d3_Txz - d4_Txz) / dz;
            float dTyz_dz = 0.25f*(d1_Tyz - d2_Tyz + d3_Tyz - d4_Tyz) / dz;
            float dTzz_dz = 0.25f*(d1_Tzz - d2_Tzz + d3_Tzz - d4_Tzz) / dz;

            float B000 = (minB + (static_cast<float>(B[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float B100 = (minB + (static_cast<float>(B[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float B010 = (minB + (static_cast<float>(B[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float B001 = (minB + (static_cast<float>(B[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float B110 = (minB + (static_cast<float>(B[i + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float B101 = (minB + (static_cast<float>(B[(i+1) + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float B011 = (minB + (static_cast<float>(B[(i+1) + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float B111 = (minB + (static_cast<float>(B[(i+1) + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float Bxyz = 0.125f*(B000 + B100 + B010 + B001 + B110 + B101 + B011 + B111);

            Vx[index] += dt*Bxyz*(dTxx_dx + dTxy_dy + dTxz_dz); 
            Vy[index] += dt*Bxyz*(dTxy_dx + dTyy_dy + dTyz_dz);
            Vz[index] += dt*Bxyz*(dTxz_dx + dTyz_dy + dTzz_dz);    
            
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
}

__global__ void float_compute_velocity_rsg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, float * B,
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

    float d1_Txx = 0.0f; float d2_Txx = 0.0f; float d3_Txx = 0.0f; float d4_Txx = 0.0f;
    float d1_Tyy = 0.0f; float d2_Tyy = 0.0f; float d3_Tyy = 0.0f; float d4_Tyy = 0.0f;
    float d1_Tzz = 0.0f; float d2_Tzz = 0.0f; float d3_Tzz = 0.0f; float d4_Tzz = 0.0f;
    float d1_Txy = 0.0f; float d2_Txy = 0.0f; float d3_Txy = 0.0f; float d4_Txy = 0.0f;
    float d1_Txz = 0.0f; float d2_Txz = 0.0f; float d3_Txz = 0.0f; float d4_Txz = 0.0f;
    float d1_Tyz = 0.0f; float d2_Tyz = 0.0f; float d3_Tyz = 0.0f; float d4_Tyz = 0.0f;            
 
    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4) && (k >= 3) && (k < nyy-4)) 
        {   
            # pragma unroll 4 
            for (int rsg = 0; rsg < 4; rsg++)
            {
                d1_Txx += FDM[rsg]*(Txx[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txx[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Tyy += FDM[rsg]*(Tyy[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tyy[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Tzz += FDM[rsg]*(Tzz[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tzz[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Txy += FDM[rsg]*(Txy[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txy[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Txz += FDM[rsg]*(Txz[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txz[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d1_Tyz += FDM[rsg]*(Tyz[(i+rsg+1) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tyz[(i-rsg) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);

                d2_Txx += FDM[rsg]*(Txx[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txx[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Tyy += FDM[rsg]*(Tyy[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tyy[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Tzz += FDM[rsg]*(Tzz[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tzz[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Txy += FDM[rsg]*(Txy[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txy[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Txz += FDM[rsg]*(Txz[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Txz[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);
                d2_Tyz += FDM[rsg]*(Tyz[(i-rsg) + (j+rsg+1)*nzz + (k+rsg+1)*nxx*nzz] - Tyz[(i+rsg+1) + (j-rsg)*nzz + (k-rsg)*nxx*nzz]);

                d3_Txx += FDM[rsg]*(Txx[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txx[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Tyy += FDM[rsg]*(Tyy[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tyy[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Tzz += FDM[rsg]*(Tzz[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tzz[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Txy += FDM[rsg]*(Txy[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txy[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Txz += FDM[rsg]*(Txz[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txz[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d3_Tyz += FDM[rsg]*(Tyz[(i+rsg+1) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tyz[(i-rsg) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);

                d4_Txx += FDM[rsg]*(Txx[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txx[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Tyy += FDM[rsg]*(Tyy[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tyy[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Tzz += FDM[rsg]*(Tzz[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tzz[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Txy += FDM[rsg]*(Txy[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txy[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Txz += FDM[rsg]*(Txz[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Txz[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
                d4_Tyz += FDM[rsg]*(Tyz[(i-rsg) + (j+rsg+1)*nzz + (k-rsg)*nxx*nzz] - Tyz[(i+rsg+1) + (j-rsg)*nzz + (k+rsg+1)*nxx*nzz]);
            }
    
            float dTxx_dx = 0.25f*(d1_Txx + d2_Txx + d3_Txx + d4_Txx) / dx;
            float dTxy_dx = 0.25f*(d1_Txy + d2_Txy + d3_Txy + d4_Txy) / dx;
            float dTxz_dx = 0.25f*(d1_Txz + d2_Txz + d3_Txz + d4_Txz) / dx;
        
            float dTxy_dy = 0.25f*(d1_Txy + d2_Txy - d3_Txy - d4_Txy) / dy;
            float dTyy_dy = 0.25f*(d1_Tyy + d2_Tyy - d3_Tyy - d4_Tyy) / dy;
            float dTyz_dy = 0.25f*(d1_Tyz + d2_Tyz - d3_Tyz - d4_Tyz) / dy;
            
            float dTxz_dz = 0.25f*(d1_Txz - d2_Txz + d3_Txz - d4_Txz) / dz;
            float dTyz_dz = 0.25f*(d1_Tyz - d2_Tyz + d3_Tyz - d4_Tyz) / dz;
            float dTzz_dz = 0.25f*(d1_Tzz - d2_Tzz + d3_Tzz - d4_Tzz) / dz;

            float B000 = B[i + j*nzz + k*nxx*nzz];
            float B100 = B[i + (j+1)*nzz + k*nxx*nzz];
            float B010 = B[i + j*nzz + (k+1)*nxx*nzz];
            float B001 = B[(i+1) + j*nzz + k*nxx*nzz];
            float B110 = B[i + (j+1)*nzz + (k+1)*nxx*nzz];
            float B101 = B[(i+1) + (j+1)*nzz + k*nxx*nzz];
            float B011 = B[(i+1) + j*nzz + (k+1)*nxx*nzz];
            float B111 = B[(i+1) + (j+1)*nzz + (k+1)*nxx*nzz];

            float Bxyz = 0.125f*(B000 + B100 + B010 + B001 + B110 + B101 + B011 + B111);

            Vx[index] += dt*Bxyz*(dTxx_dx + dTxy_dy + dTxz_dz); 
            Vy[index] += dt*Bxyz*(dTxy_dx + dTyy_dy + dTyz_dz);
            Vz[index] += dt*Bxyz*(dTxz_dx + dTyz_dy + dTzz_dz);    
            
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
}

__global__ void uintc_compute_pressure_rsg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, 
                                           uintc * C11, uintc * C12, uintc * C13, uintc * C14, uintc * C15, uintc * C16, uintc * C22, uintc * C23, uintc * C24, uintc * C25, 
                                           uintc * C26, uintc * C33, uintc * C34, uintc * C35, uintc * C36, uintc * C44, uintc * C45, uintc * C46, uintc * C55, uintc * C56, 
                                           uintc * C66, int tId, int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz, float minC11, float maxC11, 
                                           float minC12, float maxC12, float minC13, float maxC13, float minC14, float maxC14, float minC15, float maxC15, float minC16, float maxC16, float minC22, 
                                           float maxC22, float minC23, float maxC23, float minC24, float maxC24, float minC25, float maxC25, float minC26, float maxC26, float minC33, float maxC33, 
                                           float minC34, float maxC34, float minC35, float maxC35, float minC36, float maxC36, float minC44, float maxC44, float minC45, float maxC45, float minC46, 
                                           float maxC46, float minC55, float maxC55, float minC56, float maxC56, float minC66, float maxC66, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    float d1_Vx = 0.0f; float d2_Vx = 0.0f; float d3_Vx = 0.0f; float d4_Vx = 0.0f;
    float d1_Vy = 0.0f; float d2_Vy = 0.0f; float d3_Vy = 0.0f; float d4_Vy = 0.0f;
    float d1_Vz = 0.0f; float d2_Vz = 0.0f; float d3_Vz = 0.0f; float d4_Vz = 0.0f;

    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};

    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3) && (k > 3) && (k < nyy-3)) 
        {
            # pragma unroll 4
            for (int rsg = 0; rsg < 4; rsg++)
            {       
                d1_Vx += FDM[rsg]*(Vx[(i+rsg) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vx[(i-rsg-1) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
                d1_Vy += FDM[rsg]*(Vy[(i+rsg) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vy[(i-rsg-1) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
                d1_Vz += FDM[rsg]*(Vz[(i+rsg) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vz[(i-rsg-1) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
    
                d2_Vx += FDM[rsg]*(Vx[(i-rsg-1) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vx[(i+rsg) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
                d2_Vy += FDM[rsg]*(Vy[(i-rsg-1) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vy[(i+rsg) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
                d2_Vz += FDM[rsg]*(Vz[(i-rsg-1) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vz[(i+rsg) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
    
                d3_Vx += FDM[rsg]*(Vx[(i+rsg) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vx[(i-rsg-1) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
                d3_Vy += FDM[rsg]*(Vy[(i+rsg) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vy[(i-rsg-1) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
                d3_Vz += FDM[rsg]*(Vz[(i+rsg) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vz[(i-rsg-1) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
    
                d4_Vx += FDM[rsg]*(Vx[(i-rsg-1) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vx[(i+rsg) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
                d4_Vy += FDM[rsg]*(Vy[(i-rsg-1) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vy[(i+rsg) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
                d4_Vz += FDM[rsg]*(Vz[(i-rsg-1) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vz[(i+rsg) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);                          
            }
    
            float dVx_dx = 0.25f*(d1_Vx + d2_Vx + d3_Vx + d4_Vx) / dx;
            float dVy_dx = 0.25f*(d1_Vy + d2_Vy + d3_Vy + d4_Vy) / dx;
            float dVz_dx = 0.25f*(d1_Vz + d2_Vz + d3_Vz + d4_Vz) / dx;
        
            float dVx_dy = 0.25f*(d1_Vx + d2_Vx - d3_Vx - d4_Vx) / dy;
            float dVy_dy = 0.25f*(d1_Vy + d2_Vy - d3_Vy - d4_Vy) / dy;
            float dVz_dy = 0.25f*(d1_Vz + d2_Vz - d3_Vz - d4_Vz) / dy;
        
            float dVx_dz = 0.25f*(d1_Vx - d2_Vx + d3_Vx - d4_Vx) / dz;
            float dVy_dz = 0.25f*(d1_Vy - d2_Vy + d3_Vy - d4_Vy) / dz;
            float dVz_dz = 0.25f*(d1_Vz - d2_Vz + d3_Vz - d4_Vz) / dz;

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
                    
            Txx[index] += dt*(c11*dVx_dx + c16*dVx_dy + c15*dVx_dz +
                              c16*dVy_dx + c12*dVy_dy + c14*dVy_dz +
                              c15*dVz_dx + c14*dVz_dy + c13*dVz_dz);                    
        
            Tyy[index] += dt*(c12*dVx_dx + c26*dVx_dy + c25*dVx_dz +
                              c26*dVy_dx + c22*dVy_dy + c24*dVy_dz +
                              c25*dVz_dx + c24*dVz_dy + c23*dVz_dz);                    
        
            Tzz[index] += dt*(c13*dVx_dx + c36*dVx_dy + c35*dVx_dz +
                              c36*dVy_dx + c23*dVy_dy + c34*dVy_dz +
                              c35*dVz_dx + c34*dVz_dy + c33*dVz_dz);  
        
            Txy[index] += dt*(c16*dVx_dx + c66*dVx_dy + c56*dVx_dz +
                              c66*dVy_dx + c26*dVy_dy + c46*dVy_dz +
                              c56*dVz_dx + c46*dVz_dy + c36*dVz_dz);                    
        
            Txz[index] += dt*(c15*dVx_dx + c56*dVx_dy + c55*dVx_dz +
                              c56*dVy_dx + c25*dVy_dy + c45*dVy_dz +
                              c55*dVz_dx + c45*dVz_dy + c35*dVz_dz);                    
        
            Tyz[index] += dt*(c14*dVx_dx + c46*dVx_dy + c45*dVx_dz +
                              c46*dVy_dx + c24*dVy_dy + c44*dVy_dz +
                              c45*dVz_dx + c44*dVz_dy + c34*dVz_dz); 
    
            P[index] = (Txx[index] + Tyy[index] + Tzz[index]) / 3.0f;
        }
    }
}

__global__ void float_compute_pressure_rsg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, 
                                           float * C11, float * C12, float * C13, float * C14, float * C15, float * C16, float * C22, float * C23, float * C24, float * C25, float * C26, 
                                           float * C33, float * C34, float * C35, float * C36, float * C44, float * C45, float * C46, float * C55, float * C56, float * C66, int tId, 
                                           int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    float d1_Vx = 0.0f; float d2_Vx = 0.0f; float d3_Vx = 0.0f; float d4_Vx = 0.0f;
    float d1_Vy = 0.0f; float d2_Vy = 0.0f; float d3_Vy = 0.0f; float d4_Vy = 0.0f;
    float d1_Vz = 0.0f; float d2_Vz = 0.0f; float d3_Vz = 0.0f; float d4_Vz = 0.0f;

    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};

    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3) && (k > 3) && (k < nyy-3)) 
        {
            # pragma unroll 4
            for (int rsg = 0; rsg < 4; rsg++)
            {       
                d1_Vx += FDM[rsg]*(Vx[(i+rsg) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vx[(i-rsg-1) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
                d1_Vy += FDM[rsg]*(Vy[(i+rsg) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vy[(i-rsg-1) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
                d1_Vz += FDM[rsg]*(Vz[(i+rsg) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vz[(i-rsg-1) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
    
                d2_Vx += FDM[rsg]*(Vx[(i-rsg-1) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vx[(i+rsg) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
                d2_Vy += FDM[rsg]*(Vy[(i-rsg-1) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vy[(i+rsg) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
                d2_Vz += FDM[rsg]*(Vz[(i-rsg-1) + (j+rsg)*nzz + (k+rsg)*nxx*nzz] - Vz[(i+rsg) + (j-rsg-1)*nzz + (k-rsg-1)*nxx*nzz]);      
    
                d3_Vx += FDM[rsg]*(Vx[(i+rsg) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vx[(i-rsg-1) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
                d3_Vy += FDM[rsg]*(Vy[(i+rsg) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vy[(i-rsg-1) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
                d3_Vz += FDM[rsg]*(Vz[(i+rsg) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vz[(i-rsg-1) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
    
                d4_Vx += FDM[rsg]*(Vx[(i-rsg-1) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vx[(i+rsg) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
                d4_Vy += FDM[rsg]*(Vy[(i-rsg-1) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vy[(i+rsg) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);      
                d4_Vz += FDM[rsg]*(Vz[(i-rsg-1) + (j+rsg)*nzz + (k-rsg-1)*nxx*nzz] - Vz[(i+rsg) + (j-rsg-1)*nzz + (k+rsg)*nxx*nzz]);                          
            }
    
            float dVx_dx = 0.25f*(d1_Vx + d2_Vx + d3_Vx + d4_Vx) / dx;
            float dVy_dx = 0.25f*(d1_Vy + d2_Vy + d3_Vy + d4_Vy) / dx;
            float dVz_dx = 0.25f*(d1_Vz + d2_Vz + d3_Vz + d4_Vz) / dx;
        
            float dVx_dy = 0.25f*(d1_Vx + d2_Vx - d3_Vx - d4_Vx) / dy;
            float dVy_dy = 0.25f*(d1_Vy + d2_Vy - d3_Vy - d4_Vy) / dy;
            float dVz_dy = 0.25f*(d1_Vz + d2_Vz - d3_Vz - d4_Vz) / dy;
        
            float dVx_dz = 0.25f*(d1_Vx - d2_Vx + d3_Vx - d4_Vx) / dz;
            float dVy_dz = 0.25f*(d1_Vy - d2_Vy + d3_Vy - d4_Vy) / dz;
            float dVz_dz = 0.25f*(d1_Vz - d2_Vz + d3_Vz - d4_Vz) / dz;
                    
            Txx[index] += dt*(C11[index]*dVx_dx + C16[index]*dVx_dy + C15[index]*dVx_dz +
                              C16[index]*dVy_dx + C12[index]*dVy_dy + C14[index]*dVy_dz +
                              C15[index]*dVz_dx + C14[index]*dVz_dy + C13[index]*dVz_dz);                    
        
            Tyy[index] += dt*(C12[index]*dVx_dx + C26[index]*dVx_dy + C25[index]*dVx_dz +
                              C26[index]*dVy_dx + C22[index]*dVy_dy + C24[index]*dVy_dz +
                              C25[index]*dVz_dx + C24[index]*dVz_dy + C23[index]*dVz_dz);                    
        
            Tzz[index] += dt*(C13[index]*dVx_dx + C36[index]*dVx_dy + C35[index]*dVx_dz +
                              C36[index]*dVy_dx + C23[index]*dVy_dy + C34[index]*dVy_dz +
                              C35[index]*dVz_dx + C34[index]*dVz_dy + C33[index]*dVz_dz);  
        
            Txy[index] += dt*(C16[index]*dVx_dx + C66[index]*dVx_dy + C56[index]*dVx_dz +
                              C66[index]*dVy_dx + C26[index]*dVy_dy + C46[index]*dVy_dz +
                              C56[index]*dVz_dx + C46[index]*dVz_dy + C36[index]*dVz_dz);                    
        
            Txz[index] += dt*(C15[index]*dVx_dx + C56[index]*dVx_dy + C55[index]*dVx_dz +
                              C56[index]*dVy_dx + C25[index]*dVy_dy + C45[index]*dVy_dz +
                              C55[index]*dVz_dx + C45[index]*dVz_dy + C35[index]*dVz_dz);                    
        
            Tyz[index] += dt*(C14[index]*dVx_dx + C46[index]*dVx_dy + C45[index]*dVx_dz +
                              C46[index]*dVy_dx + C24[index]*dVy_dy + C44[index]*dVy_dz +
                              C45[index]*dVz_dx + C44[index]*dVz_dy + C34[index]*dVz_dz); 
    
            P[index] = (Txx[index] + Tyy[index] + Tzz[index]) / 3.0f;
        }
    }
}
