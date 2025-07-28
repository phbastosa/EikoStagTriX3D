# include "elastic_iso.cuh"

void Elastic_ISO::set_specifications()
{
    modeling_type = "elastic_iso";
    modeling_name = "Modeling type: Elastic isotropic solver";

    auto * Cij = new float[nPoints]();

    std::string vp_file = catch_parameter("vp_model_file", parameters);
    std::string ro_file = catch_parameter("ro_model_file", parameters);
    std::string Cijkl_folder = catch_parameter("Cijkl_folder", parameters);

    float * S = new float[volsize]();
    import_binary_float(vp_file, Cij, nPoints);
    expand_boundary(Cij, S);

    # pragma omp parallel for
    for (int index = 0; index < volsize; index++)
        S[index] = 1.0f / S[index];

    cudaMalloc((void**)&(d_S), volsize*sizeof(float));
    cudaMemcpy(d_S, S, volsize*sizeof(float), cudaMemcpyHostToDevice);
    delete[] S;

    auto * B = new float[volsize]();
    auto * uB = new uintc[volsize]();
    import_binary_float(ro_file, Cij, nPoints);
    expand_boundary(Cij, B);

    # pragma omp parallel for
    for (int index = 0; index < volsize; index++)
        B[index] = 1.0f / B[index];

    compression(B, uB, volsize, maxB, minB);    
    cudaMalloc((void**)&(d_B), volsize*sizeof(uintc));
    cudaMemcpy(d_B, uB, volsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] B;
    delete[] uB;

    auto * C13 = new float[volsize]();
    auto * uC13 = new uintc[volsize]();
    import_binary_float(Cijkl_folder + "C13.bin", Cij, nPoints);
    expand_boundary(Cij, C13);
    compression(C13, uC13, volsize, maxC13, minC13);    
    cudaMalloc((void**)&(d_C13), volsize*sizeof(uintc));
    cudaMemcpy(d_C13, uC13, volsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C13;
    delete[] uC13;

    auto * C55 = new float[volsize]();
    auto * uC55 = new uintc[volsize]();
    import_binary_float(Cijkl_folder + "C55.bin", Cij, nPoints);
    expand_boundary(Cij, C55);
    compression(C55, uC55, volsize, maxC55, minC55);    
    cudaMalloc((void**)&(d_C55), volsize*sizeof(uintc));
    cudaMemcpy(d_C55, uC55, volsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C55;
    delete[] uC55;
}

void Elastic_ISO::initialization()
{
    float beta = 5.0f;

    sx = geometry->xsrc[geometry->sInd[srcId]];
    sy = geometry->ysrc[geometry->sInd[srcId]];
    sz = geometry->zsrc[geometry->sInd[srcId]];

    sIdx = (int)((sx + 0.5f*dx) / dx);
    sIdy = (int)((sy + 0.5f*dy) / dy);
    sIdz = (int)((sz + 0.5f*dz) / dz);

    float * h_skw = new float[DGS*DGS*DGS]();

    auto skw = kaiser_weights(sx, sy, sz, sIdx, sIdy, sIdz, dx, dy, dz, beta);

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
    
        auto rkwPs = kaiser_weights(rx, ry, rz, rIdx, rIdy, rIdz, dx, dy, dz, beta);
        auto rkwVx = kaiser_weights(rx + 0.5f*dx, ry, rz, rIdx, rIdy, rIdz, dx, dy, dz, beta);
        auto rkwVy = kaiser_weights(rx, ry + 0.5f*dy, rz, rIdx, rIdy, rIdz, dx, dy, dz, beta);
        auto rkwVz = kaiser_weights(rx, ry, rz + 0.5f*dz, rIdx, rIdy, rIdz, dx, dy, dz, beta);
        
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

void Elastic_ISO::compute_eikonal()
{
    dim3 grid(1,1,1);
    dim3 block(MESHDIM,MESHDIM,MESHDIM);

    time_set<<<nBlocks,NTHREADS>>>(d_T, volsize);
    time_init<<<grid,block>>>(d_T,d_S,sx,sy,sz,dx,dy,dz,sIdx,sIdy,sIdz,nxx,nzz,nb);
    eikonal_solver();
}

void Elastic_ISO::compute_velocity()
{
    compute_velocity_ssg<<<nBlocks,NTHREADS>>>(d_Vx, d_Vy, d_Vz, d_Txx, d_Tyy, d_Tzz, d_Txz, d_Tyz, d_Txy, d_T, d_B, maxB, minB, d1D, d2D, 
                                               d3D, d_wavelet, dx, dy, dz, dt, timeId, tlag, sIdx, sIdy, sIdz, d_skw, nxx, nyy, nzz, nb, nt);
}

void Elastic_ISO::compute_pressure()
{
    compute_pressure_ssg<<<nBlocks,NTHREADS>>>(d_Vx, d_Vy, d_Vz, d_Txx, d_Tyy, d_Tzz, d_Txz, d_Tyz, d_Txy, d_P, d_T, d_C55, d_C13, 
                                               maxC55, minC55, maxC13, minC13, timeId, tlag, dx, dy, dz, dt, nxx, nyy, nzz);    
}

__global__ void compute_velocity_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * T, uintc * B,
                                     float maxB, float minB, float * damp1D, float * damp2D, float * damp3D, float * wavelet, float dx, float dy, float dz, float dt, int tId, 
                                     int tlag, int sIdx, int sIdy, int sIdz, float * skw, int nxx, int nyy, int nzz, int nb, int nt)
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
            int yi = sIdy + k - 2;
            
            for (int j = 0; j < DGS; j++)
            {
                int xi = sIdx + j - 2;
    
                for (int i = 0; i < DGS; i++)
                {
                    int zi = sIdz + i - 2;
            
                    Txx[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);
                    Tyy[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);
                    Tzz[zi + xi*nzz + yi*nxx*nzz] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);           
                }
            }
        }
    }

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

__global__ void compute_pressure_ssg(float * Vx, float * Vy, float * Vz, float * Txx, float * Tyy, float * Tzz, float * Txz, float * Tyz, float * Txy, float * P, float * T, uintc * C55, 
                                     uintc * C13, float maxC55, float minC55, float maxC13, float minC13, int tId, int tlag, float dx, float dy, float dz, float dt, int nxx, int nyy, int nzz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    float c55_1, c55_2, c55_3, c55_4;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nyy*nzz))
    {
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4) && (k >= 3) && (k < nyy-4)) 
        {    
            float dVx_dx = (FDM1*(Vx[i + (j-3)*nzz + k*nxx*nzz] - Vx[i + (j+4)*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[i + (j+3)*nzz + k*nxx*nzz] - Vx[i + (j-2)*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[i + (j-1)*nzz + k*nxx*nzz] - Vx[i + (j+2)*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + (j+1)*nzz + k*nxx*nzz] - Vx[i + j*nzz + k*nxx*nzz])) / dx;

            float dVy_dy = (FDM1*(Vy[i + j*nzz + (k-3)*nxx*nzz] - Vy[i + j*nzz + (k+4)*nxx*nzz]) +
                            FDM2*(Vy[i + j*nzz + (k+3)*nxx*nzz] - Vy[i + j*nzz + (k-2)*nxx*nzz]) +
                            FDM3*(Vy[i + j*nzz + (k-1)*nxx*nzz] - Vy[i + j*nzz + (k+2)*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + (k+1)*nxx*nzz] - Vy[i + j*nzz + k*nxx*nzz])) / dy;

            float dVz_dz = (FDM1*(Vz[(i-3) + j*nzz + k*nxx*nzz] - Vz[(i+4) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[(i+3) + j*nzz + k*nxx*nzz] - Vz[(i-2) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[(i-1) + j*nzz + k*nxx*nzz] - Vz[(i+2) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[(i+1) + j*nzz + k*nxx*nzz] - Vz[i + j*nzz + k*nxx*nzz])) / dz;
            
            float c13 = (minC13 + (static_cast<float>(C13[index]) - 1.0f) * (maxC13 - minC13) / (COMPRESS - 1));
            float c44 = (minC55 + (static_cast<float>(C55[index]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            Txx[index] += dt*((c13 + 2*c44)*dVx_dx + c13*(dVy_dy + dVz_dz));
            Tyy[index] += dt*((c13 + 2*c44)*dVy_dy + c13*(dVx_dx + dVz_dz));
            Tzz[index] += dt*((c13 + 2*c44)*dVz_dz + c13*(dVx_dx + dVy_dy));                    
        }

        if((i >= 3) && (i < nzz-4) && (j > 3) && (j < nxx-3) && (k > 3) && (k < nyy-3)) 
        {
            float dVx_dy = (FDM1*(Vx[i + j*nzz + (k-4)*nxx*nzz] - Vx[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vx[i + j*nzz + (k+2)*nxx*nzz] - Vx[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vx[i + j*nzz + (k-2)*nxx*nzz] - Vx[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[i + j*nzz + (k-1)*nxx*nzz])) / dy;

            float dVy_dx = (FDM1*(Vy[i + (j-4)*nzz + k*nxx*nzz] - Vy[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[i + (j+2)*nzz + k*nxx*nzz] - Vy[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[i + (j-2)*nzz + k*nxx*nzz] - Vy[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            c55_1 = (minC55 + (static_cast<float>(C55[i + (j+1)*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_2 = (minC55 + (static_cast<float>(C55[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_3 = (minC55 + (static_cast<float>(C55[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_4 = (minC55 + (static_cast<float>(C55[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            float Mxy = powf(0.25f*(1.0f/c55_1 + 1.0f/c55_2 + 1.0f/c55_3 + 1.0f/c55_4),-1.0f);

            Txy[index] += dt*Mxy*(dVx_dy + dVy_dx);
        }

        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3) && (k >= 3) && (k < nyy-4)) 
        {
            float dVx_dz = (FDM1*(Vx[(i-4) + j*nzz + k*nxx*nzz] - Vx[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vx[(i+2) + j*nzz + k*nxx*nzz] - Vx[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vx[(i-2) + j*nzz + k*nxx*nzz] - Vx[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vx[i + j*nzz + k*nxx*nzz]     - Vx[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-4)*nzz + k*nxx*nzz] - Vz[i + (j+3)*nzz + k*nxx*nzz]) +
                            FDM2*(Vz[i + (j+2)*nzz + k*nxx*nzz] - Vz[i + (j-3)*nzz + k*nxx*nzz]) +
                            FDM3*(Vz[i + (j-2)*nzz + k*nxx*nzz] - Vz[i + (j+1)*nzz + k*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + (j-1)*nzz + k*nxx*nzz])) / dx;

            c55_1 = (minC55 + (static_cast<float>(C55[(i+1) + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_2 = (minC55 + (static_cast<float>(C55[i + (j+1)*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_3 = (minC55 + (static_cast<float>(C55[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_4 = (minC55 + (static_cast<float>(C55[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            float Mxz = powf(0.25f*(1.0f/c55_1 + 1.0f/c55_2 + 1.0f/c55_3 + 1.0f/c55_4),-1.0f);

            Txz[index] += dt*Mxz*(dVx_dz + dVz_dx);
        }

        if((i > 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4) && (k > 3) && (k < nyy-3)) 
        {
            float dVy_dz = (FDM1*(Vy[(i-4) + j*nzz + k*nxx*nzz] - Vy[(i+3) + j*nzz + k*nxx*nzz]) +
                            FDM2*(Vy[(i+2) + j*nzz + k*nxx*nzz] - Vy[(i-3) + j*nzz + k*nxx*nzz]) +
                            FDM3*(Vy[(i-2) + j*nzz + k*nxx*nzz] - Vy[(i+1) + j*nzz + k*nxx*nzz]) +
                            FDM4*(Vy[i + j*nzz + k*nxx*nzz]     - Vy[(i-1) + j*nzz + k*nxx*nzz])) / dz;

            float dVz_dy = (FDM1*(Vz[i + j*nzz + (k-4)*nxx*nzz] - Vz[i + j*nzz + (k+3)*nxx*nzz]) +
                            FDM2*(Vz[i + j*nzz + (k+2)*nxx*nzz] - Vz[i + j*nzz + (k-3)*nxx*nzz]) +
                            FDM3*(Vz[i + j*nzz + (k-2)*nxx*nzz] - Vz[i + j*nzz + (k+1)*nxx*nzz]) +
                            FDM4*(Vz[i + j*nzz + k*nxx*nzz]     - Vz[i + j*nzz + (k-1)*nxx*nzz])) / dy;
            
            c55_1 = (minC55 + (static_cast<float>(C55[(i+1) + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_2 = (minC55 + (static_cast<float>(C55[i + j*nzz + (k+1)*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_3 = (minC55 + (static_cast<float>(C55[(i+1) + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_4 = (minC55 + (static_cast<float>(C55[i + j*nzz + k*nxx*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            float Myz = powf(0.25f*(1.0f/c55_1 + 1.0f/c55_2 + 1.0f/c55_3 + 1.0f/c55_4),-1.0f);

            Tyz[index] += dt*Myz*(dVy_dz + dVz_dy);
        }

        if ((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4) && (k > 3) && (k < nyy-4))
            P[index] = (Txx[index] + Tyy[index] + Tzz[index]) / 3.0f;
    }
}