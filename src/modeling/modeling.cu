# include "modeling.cuh"

void Modeling::set_parameters()
{
    nx = std::stoi(catch_parameter("x_samples", parameters));
    ny = std::stoi(catch_parameter("y_samples", parameters));
    nz = std::stoi(catch_parameter("z_samples", parameters));

    dx = std::stof(catch_parameter("x_spacing", parameters));
    dy = std::stof(catch_parameter("y_spacing", parameters));
    dz = std::stof(catch_parameter("z_spacing", parameters));

    nt = std::stoi(catch_parameter("time_samples", parameters));
    dt = std::stof(catch_parameter("time_spacing", parameters));
    
    fmax = std::stof(catch_parameter("max_frequency", parameters));

    nb = std::stoi(catch_parameter("boundary_samples", parameters));
    bd = std::stof(catch_parameter("boundary_damping", parameters));
    
    isnap = std::stoi(catch_parameter("beg_snap", parameters));
    fsnap = std::stoi(catch_parameter("end_snap", parameters));
    nsnap = std::stoi(catch_parameter("num_snap", parameters));

    snapshot = str2bool(catch_parameter("snapshot", parameters));

    snapshot_folder = catch_parameter("snapshot_folder", parameters);
    seismogram_folder = catch_parameter("seismogram_folder", parameters);

    geometry = new Geometry();

    geometry->parameters = parameters;
    geometry->set_parameters();

    max_spread = 0;
    for (int index = 0; index < geometry->nrel; index++)
    {   
        if (max_spread < geometry->spread[index])
            max_spread = geometry->spread[index]; 
    }

    nPoints = nx*ny*nz;

    nxx = nx + 2*nb;
    nyy = ny + 2*nb;
    nzz = nz + 2*nb;

    volsize = nxx*nyy*nzz;

    nThreads = 256;
    nBlocks = (int)((volsize + nThreads - 1) / nThreads);
    sBlocks = (int)((max_spread + nThreads - 1) / nThreads); 

    h_seismogram = new float[nt*max_spread]();

    if (snapshot)
    {
        if (nsnap == 1) 
            snapId.push_back(isnap);
        else 
        {
            for (int i = 0; i < nsnap; i++) 
                snapId.push_back(isnap + i * (fsnap - isnap) / (nsnap - 1));
        }
        
        snapshot_in = new float[volsize]();
        snapshot_out = new float[nPoints]();
    }

    set_wavelet();
    set_dampers();
    set_eikonal();

    set_specifications();

    cudaMalloc((void**)&(d_P), volsize*sizeof(float));
    cudaMalloc((void**)&(d_T), volsize*sizeof(float));

    cudaMalloc((void**)&(d_Vx), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Vy), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Vz), volsize*sizeof(float));

    cudaMalloc((void**)&(d_Txx), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Tyy), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Tzz), volsize*sizeof(float));
    
    cudaMalloc((void**)&(d_Txz), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Tyz), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Txy), volsize*sizeof(float));

    cudaMalloc((void**)&(d_rIdx), max_spread*sizeof(int));
    cudaMalloc((void**)&(d_rIdy), max_spread*sizeof(int));
    cudaMalloc((void**)&(d_rIdz), max_spread*sizeof(int));

    cudaMalloc((void**)&(d_seismogram), nt*max_spread*sizeof(float));
}

void Modeling::set_wavelet()
{
    float * signal_aux1 = new float[nt]();
    float * signal_aux2 = new float[nt]();

    float t0 = 2.0f*sqrtf(M_PI) / fmax;
    float fc = fmax / (3.0f * sqrtf(M_PI));

    tlag = (int)((t0 + 0.5f*dt) / dt);

    for (int n = 0; n < nt; n++)
    {
        float td = n*dt - t0;

        float arg = M_PI*M_PI*M_PI*fc*fc*td*td;

        signal_aux1[n] = 1e5f*(1.0f - 2.0f*arg)*expf(-arg);
    }

    for (int n = 0; n < nt; n++)
    {
        float summation = 0;
        for (int i = 0; i < n; i++)
            summation += signal_aux1[i];    
        
        signal_aux2[n] = summation;
    }

    cudaMalloc((void**)&(d_wavelet), nt*sizeof(float));

    cudaMemcpy(d_wavelet, signal_aux2, nt*sizeof(float), cudaMemcpyHostToDevice);

    delete[] signal_aux1;
    delete[] signal_aux2;    
}

void Modeling::set_dampers()
{
    float * damp1D = new float[nb]();
    float * damp2D = new float[nb*nb]();
    float * damp3D = new float[nb*nb*nb]();

    for (int i = 0; i < nb; i++) 
    {
        damp1D[i] = expf(-powf(bd * (nb - i), 2.0f));
    }

    for(int i = 0; i < nb; i++) 
    {
        for (int j = 0; j < nb; j++)
        {   
            damp2D[j + i*nb] += damp1D[i];
            damp2D[i + j*nb] += damp1D[i];
        }
    }

    for (int i  = 0; i < nb; i++)
    {
        for(int j = 0; j < nb; j++)
        {
            for(int k = 0; k < nb; k++)
            {
                damp3D[i + j*nb + k*nb*nb] += damp2D[i + j*nb];
                damp3D[i + j*nb + k*nb*nb] += damp2D[j + k*nb];
                damp3D[i + j*nb + k*nb*nb] += damp2D[i + k*nb];
            }
        }
    }    

    for (int index = 0; index < nb*nb; index++)
        damp2D[index] -= 1.0f;

    for (int index = 0; index < nb*nb*nb; index++)
        damp3D[index] -= 5.0f;    

	cudaMalloc((void**)&(d1D), nb*sizeof(float));
	cudaMalloc((void**)&(d2D), nb*nb*sizeof(float));
	cudaMalloc((void**)&(d3D), nb*nb*nb*sizeof(float));

	cudaMemcpy(d1D, damp1D, nb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d2D, damp2D, nb*nb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d3D, damp3D, nb*nb*nb*sizeof(float), cudaMemcpyHostToDevice);

    delete[] damp1D;
    delete[] damp2D;
    delete[] damp3D;
}

void Modeling::set_eikonal()
{
    dz2i = 1.0f / (dz*dz);
    dx2i = 1.0f / (dx*dx);
    dy2i = 1.0f / (dy*dy);

    dz2dx2 = dz2i * dx2i;
    dz2dy2 = dz2i * dy2i;
    dx2dy2 = dx2i * dy2i;

    dsum = dz2i + dx2i + dy2i;

    total_levels = (nxx - 1) + (nyy - 1) + (nzz - 1);

    std::vector<std::vector<int>> sgnv = {{1,1,1}, {0,1,1}, {1,1,0}, {0,1,0}, {1,0,1}, {0,0,1}, {1,0,0}, {0,0,0}};
    std::vector<std::vector<int>> sgnt = {{1,1,1}, {-1,1,1}, {1,1,-1}, {-1,1,-1}, {1,-1,1}, {-1,-1,1}, {1,-1,-1}, {-1,-1,-1}};

    int * h_sgnv = new int[NSWEEPS * MESHDIM]();
    int * h_sgnt = new int[NSWEEPS * MESHDIM](); 

    for (int index = 0; index < NSWEEPS * MESHDIM; index++)
    {
        int j = index / NSWEEPS;
        int i = index % NSWEEPS;				

	    h_sgnv[i + j * NSWEEPS] = sgnv[i][j];
	    h_sgnt[i + j * NSWEEPS] = sgnt[i][j];
    }

    cudaMalloc((void**)&(d_T), volsize*sizeof(float));

    cudaMalloc((void**)&(d_sgnv), NSWEEPS*MESHDIM*sizeof(int));
    cudaMalloc((void**)&(d_sgnt), NSWEEPS*MESHDIM*sizeof(int));

    cudaMemcpy(d_sgnv, h_sgnv, NSWEEPS*MESHDIM*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sgnt, h_sgnt, NSWEEPS*MESHDIM*sizeof(int), cudaMemcpyHostToDevice);

    std::vector<std::vector<int>>().swap(sgnv);
    std::vector<std::vector<int>>().swap(sgnt);

    delete[] h_sgnt;
    delete[] h_sgnv;
}

void Modeling::time_propagation()
{
    set_wavefields();
    initialization();
    compute_eikonal();

    if (snapshot)
    {
        snapCount = 0;

        cudaMemcpy(snapshot_in, d_T, volsize*sizeof(float), cudaMemcpyDeviceToHost);
        reduce_boundary(snapshot_in, snapshot_out);
        export_binary_float(snapshot_folder + modeling_type + "_eikonal_" + std::to_string(nz) + "x" + std::to_string(nx) + "x" + std::to_string(ny) + "_shot_" + std::to_string(geometry->sInd[srcId]+1) + ".bin", snapshot_out, nPoints);    
    }

    for (timeId = 0; timeId < nt + tlag; timeId++)
    {
        compute_velocity();
        compute_pressure();
        compute_snapshots();
        compute_seismogram();    
        show_time_progress();
    }
}

void Modeling::set_wavefields()
{
    cudaMemset(d_P, 0.0f, volsize*sizeof(float));
    
	cudaMemset(d_Vx, 0.0f, volsize*sizeof(float));
    cudaMemset(d_Vy, 0.0f, volsize*sizeof(float));
    cudaMemset(d_Vz, 0.0f, volsize*sizeof(float));
    
	cudaMemset(d_Txx, 0.0f, volsize*sizeof(float));
	cudaMemset(d_Tyy, 0.0f, volsize*sizeof(float));
    cudaMemset(d_Tzz, 0.0f, volsize*sizeof(float));
    
	cudaMemset(d_Txz, 0.0f, volsize*sizeof(float));
	cudaMemset(d_Tyz, 0.0f, volsize*sizeof(float));
	cudaMemset(d_Txy, 0.0f, volsize*sizeof(float));
}

void Modeling::initialization()
{
    sIdx = (int)((geometry->xsrc[geometry->sInd[srcId]] + 0.5f*dx) / dx) + nb;
    sIdy = (int)((geometry->ysrc[geometry->sInd[srcId]] + 0.5f*dy) / dy) + nb;
    sIdz = (int)((geometry->zsrc[geometry->sInd[srcId]] + 0.5f*dz) / dz) + nb;

    int * h_rIdx = new int[max_spread]();
    int * h_rIdy = new int[max_spread]();
    int * h_rIdz = new int[max_spread]();

    int spread = 0;

    for (recId = geometry->iRec[srcId]; recId < geometry->fRec[srcId]; recId++)
    {
        h_rIdx[spread] = (int)((geometry->xrec[recId] + 0.5f*dx) / dx) + nb;
        h_rIdy[spread] = (int)((geometry->yrec[recId] + 0.5f*dy) / dy) + nb;
        h_rIdz[spread] = (int)((geometry->zrec[recId] + 0.5f*dz) / dz) + nb;
    
        ++spread;
    }

    cudaMemcpy(d_rIdx, h_rIdx, geometry->spread[srcId]*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdy, h_rIdy, geometry->spread[srcId]*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdz, h_rIdz, geometry->spread[srcId]*sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_rIdx;
    delete[] h_rIdy;
    delete[] h_rIdz;
}

void Modeling::eikonal_solver()
{
    for (int sweep = 0; sweep < NSWEEPS; sweep++)
    { 
	    int start = (sweep == 3 || sweep == 5 || sweep == 6 || sweep == 7) ? total_levels : MESHDIM;
	    int end = (start == MESHDIM) ? total_levels + 1 : MESHDIM - 1;
	    int incr = (start == MESHDIM) ? true : false;

	    int xSweepOff = (sweep == 3 || sweep == 4) ? nxx : 0;
	    int ySweepOff = (sweep == 2 || sweep == 5) ? nyy : 0;
	    int zSweepOff = (sweep == 1 || sweep == 6) ? nzz : 0;
		
	    for (int level = start; level != end; level = (incr) ? level + 1 : level - 1)
	    {			
            int xs = max(1, level - (nyy + nzz));	
            int ys = max(1, level - (nxx + nzz));
            
            int xe = min(nxx, level - (MESHDIM - 1));
            int ye = min(nyy, level - (MESHDIM - 1));	
            
            int xr = xe - xs + 1;
            int yr = ye - ys + 1;

            int nThreads = xr * yr;
                
            dim3 bs(16, 16, 1);

            if (nThreads < 32) { bs.x = xr; bs.y = yr; }  

            dim3 gs(iDivUp(xr, bs.x), iDivUp(yr , bs.y), 1);
                
            int sgni = sweep + 0*NSWEEPS;
            int sgnj = sweep + 1*NSWEEPS;
            int sgnk = sweep + 2*NSWEEPS;

            inner_sweep<<<gs, bs>>>(d_S, d_T, d_sgnt, d_sgnv, sgni, sgnj, sgnk, level, xs, ys, 
                                    xSweepOff, ySweepOff, zSweepOff, nxx, nyy, nzz, dx, dy, dz, 
                                    dx2i, dy2i, dz2i, dz2dx2, dz2dy2, dx2dy2, dsum);
	    }
    }
}

void Modeling::compute_snapshots()
{
    if (snapshot)
    {
        if (snapCount < snapId.size())
        {
            if ((timeId-tlag) == snapId[snapCount])
            {
                cudaMemcpy(snapshot_in, d_P, volsize*sizeof(float), cudaMemcpyDeviceToHost);
                reduce_boundary(snapshot_in, snapshot_out);
                export_binary_float(snapshot_folder + modeling_type + "_snapshot_step" + std::to_string(timeId-tlag) + "_" + std::to_string(nz) + "x" + std::to_string(nx) + "x" + std::to_string(ny) + "_shot_" + std::to_string(geometry->sInd[srcId]+1) + ".bin", snapshot_out, nPoints);    
                
                ++snapCount;
            }
        }
    }
}

void Modeling::show_time_progress()
{
    if (timeId > tlag)
    {
        if ((timeId - tlag) % (int)(nt / 10) == 0) 
        {
            show_information();
            
            int percent = (int)floorf((float)(timeId - tlag + 1) / (float)(nt) * 100.0f);  
            
            std::cout << "\nPropagation progress: " << percent << " % \n";
        }   
    }
}

void Modeling::compute_seismogram()
{
    compute_seismogram_GPU<<<sBlocks,nThreads>>>(d_P, d_rIdx, d_rIdy, d_rIdz, d_seismogram, geometry->spread[srcId], timeId, tlag, nt, nxx, nzz);     
}

void Modeling::export_seismogram()
{   
    cudaMemcpy(h_seismogram, d_seismogram, nt*max_spread*sizeof(float), cudaMemcpyDeviceToHost);    
    std::string data_file = seismogram_folder + modeling_type + "_nStations" + std::to_string(geometry->spread[srcId]) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(geometry->sInd[srcId]+1) + ".bin";
    export_binary_float(data_file, h_seismogram, nt*geometry->spread[srcId]);    
}

void Modeling::expand_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int k = (int) (index / (nx*nz));         
        int j = (int) (index - k*nx*nz) / nz;    
        int i = (int) (index - j*nz - k*nx*nz);  

        output[(i + nb) + (j + nb)*nzz + (k + nb)*nxx*nzz] = input[i + j*nz + k*nx*nz];       
    }

    for (int k = nb; k < nyy - nb; k++)
    {   
        for (int j = nb; j < nxx - nb; j++)
        {
            for (int i = 0; i < nb; i++)            
            {
                output[i + j*nzz + k*nxx*nzz] = input[0 + (j - nb)*nz + (k - nb)*nx*nz];
                output[(nzz - i - 1) + j*nzz + k*nxx*nzz] = input[(nz - 1) + (j - nb)*nz + (k - nb)*nx*nz];
            }
        }
    }

    for (int k = 0; k < nyy; k++)
    {   
        for (int j = 0; j < nb; j++)
        {
            for (int i = 0; i < nzz; i++)
            {
                output[i + j*nzz + k*nxx*nzz] = output[i + nb*nzz + k*nxx*nzz];
                output[i + (nxx - j - 1)*nzz + k*nxx*nzz] = output[i + (nxx - nb - 1)*nzz + k*nxx*nzz];
            }
        }
    }

    for (int k = 0; k < nb; k++)
    {   
        for (int j = 0; j < nxx; j++)
        {
            for (int i = 0; i < nzz; i++)
            {
                output[i + j*nzz + k*nxx*nzz] = output[i + j*nzz + nb*nxx*nzz];
                output[i + j*nzz + (nyy - k - 1)*nxx*nzz] = output[i + j*nzz + (nyy - nb - 1)*nxx*nzz];
            }
        }
    }
}

void Modeling::reduce_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int k = (int) (index / (nx*nz));         
        int j = (int) (index - k*nx*nz) / nz;    
        int i = (int) (index - j*nz - k*nx*nz);  

        output[i + j*nz + k*nx*nz] = input[(i + nb) + (j + nb)*nzz + (k + nb)*nxx*nzz];
    }
}

void Modeling::show_information()
{
    auto clear = system("clear");
    
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "                                 \033[34mWASMEM3D\033[0;0m\n";
    std::cout << "-------------------------------------------------------------------------------\n\n";

    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << ", x = " << (nx - 1) * dx <<", y = " << (ny - 1) * dy << ") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << 
                                       ", y = " << geometry->ysrc[geometry->sInd[srcId]] << ") m\n\n";

    std::cout << modeling_name << "\n";
}

void Modeling::compression(float * input, uintc * output, int N, float &max_value, float &min_value)
{
    max_value =-1e20f;
    min_value = 1e20f;
    
    # pragma omp parallel for
    for (int index = 0; index < N; index++)
    {
        min_value = std::min(input[index], min_value);
        max_value = std::max(input[index], max_value);        
    }

    # pragma omp parallel for
    for (int index = 0; index < N; index++)
        output[index] = static_cast<uintc>(1.0f + (COMPRESS - 1)*(input[index] - min_value) / (max_value - min_value));
}

int Modeling::iDivUp(int a, int b) 
{ 
    return ( (a % b) != 0 ) ? (a / b + 1) : (a / b); 
}

__global__ void time_set(float * T, int volsize)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < volsize) T[index] = 1e6f;
}

__global__ void time_init(float * T, float * S, float sx, float sy, float sz, float dx, float dy, 
                          float dz, int sIdx, int sIdy, int sIdz, int nxx, int nzz, int nb)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    int yi = sIdy + (k - 1);
    int xi = sIdx + (j - 1);
    int zi = sIdz + (i - 1);

    int index = zi + xi*nzz + yi*nxx*nzz;

    T[index] = S[index] * sqrtf(powf((xi - nb)*dx - sx, 2.0f) + 
                                powf((yi - nb)*dy - sy, 2.0f) +
                                powf((zi - nb)*dz - sz, 2.0f));
}

__global__ void inner_sweep(float * S, float * T, int * sgnt, int * sgnv, int sgni, int sgnj, int sgnk, 
                            int level, int xOffset, int yOffset, int xSweepOffset, int ySweepOffset, int zSweepOffset, 
                            int nxx, int nyy, int nzz, float dx, float dy, float dz, float dx2i, float dy2i, float dz2i, 
                            float dz2dx2, float dz2dy2, float dx2dy2, float dsum)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) + xOffset;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) + yOffset;

    float ta, tb, tc, t1, t2, t3, Sref;
    float t1D1, t1D2, t1D3, t1D, t2D1, t2D2, t2D3, t2D, t3D;

    if ((x < nxx) && (y < nyy)) 
    {
	    int z = level - (x + y);
		
        if ((z >= 0) && (z < nzz))	
        {
            int i = abs(z - zSweepOffset);
            int j = abs(x - xSweepOffset);
            int k = abs(y - ySweepOffset);

            if ((i > 0) && (i < nzz-1) && (j > 0) && (j < nxx-1) && (k > 0) && (k < nyy-1))
            {		
                int i1 = i - sgnv[sgni];
                int j1 = j - sgnv[sgnj];
                int k1 = k - sgnv[sgnk];

                int ijk = i + j*nzz + k*nxx*nzz;
                        
                float tv = T[(i - sgnt[sgni]) + j*nzz + k*nxx*nzz];
                float te = T[i + (j - sgnt[sgnj])*nzz + k*nxx*nzz];
                float tn = T[i + j*nzz + (k - sgnt[sgnk])*nxx*nzz];

                float tev = T[(i - sgnt[sgni]) + (j - sgnt[sgnj])*nzz + k*nxx*nzz];
                float ten = T[i + (j - sgnt[sgnj])*nzz + (k - sgnt[sgnk])*nxx*nzz];
                float tnv = T[(i - sgnt[sgni]) + j*nzz + (k - sgnt[sgnk])*nxx*nzz];
                        
                float tnve = T[(i - sgnt[sgni]) + (j - sgnt[sgnj])*nzz + (k - sgnt[sgnk])*nxx*nzz];

                t1D1 = tv + dz * min(S[i1 + max(j-1,1)*nzz   + max(k-1,1)*nxx*nzz], 
                                 min(S[i1 + max(j-1,1)*nzz   + min(k,nyy-1)*nxx*nzz], 
                                 min(S[i1 + min(j,nxx-1)*nzz + max(k-1,1)*nxx*nzz],
                                     S[i1 + min(j,nxx-1)*nzz + min(k,nyy-1)*nxx*nzz])));                                     

                t1D2 = te + dx * min(S[max(i-1,1)   + j1*nzz + max(k-1,1)*nxx*nzz], 
                                 min(S[min(i,nzz-1) + j1*nzz + max(k-1,1)*nxx*nzz],
                                 min(S[max(i-1,1)   + j1*nzz + min(k,nyy-1)*nxx*nzz], 
                                     S[min(i,nzz-1) + j1*nzz + min(k,nyy-1)*nxx*nzz])));                    

                t1D3 = tn + dy * min(S[max(i-1,1)   + max(j-1,1)*nzz   + k1*nxx*nzz], 
                                 min(S[max(i-1,1)   + min(j,nxx-1)*nzz + k1*nxx*nzz],
                                 min(S[min(i,nzz-1) + max(j-1,1)*nzz   + k1*nxx*nzz], 
                                     S[min(i,nzz-1) + min(j,nxx-1)*nzz + k1*nxx*nzz])));

                t1D = min(t1D1, min(t1D2, t1D3));

                //------------------- 2D operators - 4 points operator ---------------------------------------------------------------------------------------------------
                t2D1 = 1e6; t2D2 = 1e6; t2D3 = 1e6;

                // XZ plane ----------------------------------------------------------------------------------------------------------------------------------------------
                Sref = min(S[i1 + j1*nzz + max(k-1,1)*nxx*nzz], S[i1 + j1*nzz + min(k, nyy-1)*nxx*nzz]);
                
                if ((tv < te + dx*Sref) && (te < tv + dz*Sref))
                {
                    ta = tev + te - tv;
                    tb = tev - te + tv;

                    t2D1 = ((tb*dz2i + ta*dx2i) + sqrtf(4.0f*Sref*Sref*(dz2i + dx2i) - dz2i*dx2i*(ta - tb)*(ta - tb))) / (dz2i + dx2i);
                }

                // YZ plane -------------------------------------------------------------------------------------------------------------------------------------------------------------
                Sref = min(S[i1 + max(j-1,1)*nzz + k1*nxx*nzz], S[i1 + min(j,nxx-1)*nzz + k1*nxx*nzz]);

                if((tv < tn + dy*Sref) && (tn < tv + dz*Sref))
                {
                    ta = tv - tn + tnv;
                    tb = tn - tv + tnv;
                    
                    t2D2 = ((ta*dz2i + tb*dy2i) + sqrtf(4.0f*Sref*Sref*(dz2i + dy2i) - dz2i*dy2i*(ta - tb)*(ta - tb))) / (dz2i + dy2i); 
                }

                // XY plane -------------------------------------------------------------------------------------------------------------------------------------------------------------
                Sref = min(S[max(i-1,1) + j1*nzz + k1*nxx*nzz],S[min(i,nzz-1) + j1*nzz + k1*nxx*nzz]);

                if((te < tn + dy*Sref) && (tn < te + dx*Sref))
                {
                    ta = te - tn + ten;
                    tb = tn - te + ten;

                    t2D3 = ((ta*dx2i + tb*dy2i) + sqrtf(4.0f*Sref*Sref*(dx2i + dy2i) - dx2i*dy2i*(ta - tb)*(ta - tb))) / (dx2i + dy2i);
                }

                t2D = min(t2D1, min(t2D2, t2D3));

                //------------------- 3D operators - 8 point operator ---------------------------------------------------------------------------------------------------
                t3D = 1e6;

                Sref = S[i1 + j1*nzz + k1*nxx*nzz];

                ta = te - 0.5f*tn + 0.5f*ten - 0.5f*tv + 0.5f*tev - tnv + tnve;
                tb = tv - 0.5f*tn + 0.5f*tnv - 0.5f*te + 0.5f*tev - ten + tnve;
                tc = tn - 0.5f*te + 0.5f*ten - 0.5f*tv + 0.5f*tnv - tev + tnve;

                if (min(t1D, t2D) > max(tv, max(te, tn)))
                {
                    t2 = 9.0f*Sref*Sref*dsum; 
                    
                    t3 = dz2dx2*(ta - tb)*(ta - tb) + dz2dy2*(tb - tc)*(tb - tc) + dx2dy2*(ta - tc)*(ta - tc);
                    
                    if (t2 >= t3)
                    {
                        t1 = tb*dz2i + ta*dx2i + tc*dy2i;        
                        
                        t3D = (t1 + sqrtf(t2 - t3)) / dsum;
                    }
                }

		        T[ijk] = min(T[ijk], min(t1D, min(t2D, t3D)));
            }
        }
    }
}

__global__ void compute_seismogram_GPU(float * P, int * rIdx, int * rIdy, int * rIdz, float * seismogram, int spread, int tId, int tlag, int nt, int nxx, int nzz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < spread) && (tId >= tlag))
        seismogram[(tId - tlag) + index*nt] = P[rIdz[index] + rIdx[index]*nzz + rIdy[index]*nxx*nzz];
}

__device__ float get_boundary_damper(float * damp1D, float * damp2D, float * damp3D, int i, int j, int k, int nxx, int nyy, int nzz, int nabc)
{
    float damper;

    // global case
    if ((i >= nabc) && (i < nzz-nabc) && (j >= nabc) && (j < nxx-nabc) && (k >= nabc) && (k < nyy-nabc))
    {
        damper = 1.0f;
    }

    // 1D damping
    else if((i < nabc) && (j >= nabc) && (j < nxx-nabc) && (k >= nabc) && (k < nyy-nabc)) 
    {
        damper = damp1D[i];
    }         
    else if((i >= nzz-nabc) && (i < nzz) && (j >= nabc) && (j < nxx-nabc) && (k >= nabc) && (k < nyy-nabc)) 
    {
        damper = damp1D[nabc-(i-(nzz-nabc))-1];
    }         
    else if((i >= nabc) && (i < nzz-nabc) && (j >= 0) && (j < nabc) && (k >= nabc) && (k < nyy-nabc)) 
    {
        damper = damp1D[j];
    }
    else if((i >= nabc) && (i < nzz-nabc) && (j >= nxx-nabc) && (j < nxx) && (k >= nabc) && (k < nyy-nabc)) 
    {
        damper = damp1D[nabc-(j-(nxx-nabc))-1];
    }
    else if((i >= nabc) && (i < nzz-nabc) && (j >= nabc) && (j < nxx-nabc) && (k >= 0) && (k < nabc)) 
    {
        damper = damp1D[k];
    }
    else if((i >= nabc) && (i < nzz-nabc) && (j >= nabc) && (j < nxx-nabc) && (k >= nyy-nabc) && (k < nyy)) 
    {
        damper = damp1D[nabc-(k-(nyy-nabc))-1];
    }

    // 2D damping 
    else if((i >= nabc) && (i < nzz-nabc) && (j >= 0) && (j < nabc) && (k >= 0) && (k < nabc))
    {
        damper = damp2D[j + k*nabc];
    }
    else if((i >= nabc) && (i < nzz-nabc) && (j >= nxx-nabc) && (j < nxx) && (k >= 0) && (k < nabc))
    {
        damper = damp2D[nabc-(j-(nxx-nabc))-1 + k*nabc];
    }
    else if((i >= nabc) && (i < nzz-nabc) && (j >= 0) && (j < nabc) && (k >= nyy-nabc) && (k < nyy))
    {
        damper = damp2D[j + (nabc-(k-(nyy-nabc))-1)*nabc];
    }
    else if((i >= nabc) && (i < nzz-nabc) && (j >= nxx-nabc) && (j < nxx) && (k >= nyy-nabc) && (k < nyy))
    {
        damper = damp2D[nabc-(j-(nxx-nabc))-1 + (nabc-(k-(nyy-nabc))-1)*nabc];
    }

    else if((i >= 0) && (i < nabc) && (j >= nabc) && (j < nxx-nabc) && (k >= 0) && (k < nabc))
    {
        damper = damp2D[i + k*nabc];
    }
    else if((i >= nzz-nabc) && (i < nzz) && (j >= nabc) && (j < nxx-nabc) && (k >= 0) && (k < nabc))
    {
        damper = damp2D[nabc-(i-(nzz-nabc))-1 + k*nabc];
    }
    else if((i >= 0) && (i < nabc) && (j >= nabc) && (j < nxx-nabc) && (k >= nyy-nabc) && (k < nyy))
    {
        damper = damp2D[i + (nabc-(k-(nyy-nabc))-1)*nabc];
    }
    else if((i >= nzz-nabc) && (i < nzz) && (j >= nabc) && (j < nxx-nabc) && (k >= nyy-nabc) && (k < nyy))
    {
        damper = damp2D[nabc-(i-(nzz-nabc))-1 + (nabc-(k-(nyy-nabc))-1)*nabc];
    }

    else if((i >= 0) && (i < nabc) && (j >= 0) && (j < nabc) && (k >= nabc) && (k < nyy-nabc))
    {
        damper = damp2D[i + j*nabc];
    }
    else if((i >= nzz-nabc) && (i < nzz) && (j >= 0) && (j < nabc) && (k >= nabc) && (k < nyy-nabc))
    {
        damper = damp2D[nabc-(i-(nzz-nabc))-1 + j*nabc];
    }
    else if((i >= 0) && (i < nabc) && (j >= nxx-nabc) && (j < nxx) && (k >= nabc) && (k < nyy-nabc))
    {
        damper = damp2D[i + (nabc-(j-(nxx-nabc))-1)*nabc];
    }
    else if((i >= nzz-nabc) && (i < nzz) && (j >= nxx-nabc) && (j < nxx) && (k >= nabc) && (k < nyy-nabc))
    {
        damper = damp2D[nabc-(i-(nzz-nabc))-1 + (nabc-(j-(nxx-nabc))-1)*nabc];
    }

    // 3D damping
    else if((i >= 0) && (i < nabc) && (j >= 0) && (j < nabc) && (k >= 0) && (k < nabc))
    {
        damper = damp3D[i + j*nabc + k*nabc*nabc];
    }
    else if((i >= nzz-nabc) && (i < nzz) && (j >= 0) && (j < nabc) && (k >= 0) && (k < nabc))
    {
        damper = damp3D[nabc-(i-(nzz-nabc))-1 + j*nabc + k*nabc*nabc];
    }
    else if((i >= 0) && (i < nabc) && (j >= nxx-nabc) && (j < nxx) && (k >= 0) && (k < nabc))
    {
        damper = damp3D[i + (nabc-(j-(nxx-nabc))-1)*nabc + k*nabc*nabc];
    }
    else if((i >= 0) && (i < nabc) && (j >= 0) && (j < nabc) && (k >= nyy-nabc) && (k < nyy))
    {
        damper = damp3D[i + j*nabc + (nabc-(k-(nyy-nabc))-1)*nabc*nabc];
    }
    else if((i >= nzz-nabc) && (i < nzz) && (j >= nxx-nabc) && (j < nxx) && (k >= 0) && (k < nabc))
    {
        damper = damp3D[nabc-(i-(nzz-nabc))-1 + (nabc-(j-(nxx-nabc))-1)*nabc + k*nabc*nabc];
    }
    else if((i >= nzz-nabc) && (i < nzz) && (j >= 0) && (j < nabc) && (k >= nyy-nabc) && (k < nyy))
    {
        damper = damp3D[nabc-(i-(nzz-nabc))-1 + j*nabc + (nabc-(k-(nyy-nabc))-1)*nabc*nabc];
    }
    else if((i >= 0) && (i < nabc) && (j >= nxx-nabc) && (j < nxx) && (k >= nyy-nabc) && (k < nyy))
    {
        damper = damp3D[i + (nabc-(j-(nxx-nabc))-1)*nabc + (nabc-(k-(nyy-nabc))-1)*nabc*nabc];
    }
    else if((i >= nzz-nabc) && (i < nzz) && (j >= nxx-nabc) && (j < nxx) && (k >= nyy-nabc) && (k < nyy))
    {
        damper = damp3D[nabc-(i-(nzz-nabc))-1 + (nabc-(j-(nxx-nabc))-1)*nabc + (nabc-(k-(nyy-nabc))-1)*nabc*nabc];
    }

    return damper;
}