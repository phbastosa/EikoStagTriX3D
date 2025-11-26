# include "triclinic.cuh"

void Triclinic::set_parameters()
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

    eikonalClip = str2bool(catch_parameter("eikonalClip", parameters));
    compression = str2bool(catch_parameter("compression", parameters));

    snapshot_folder = catch_parameter("snapshot_folder", parameters);
    seismogram_folder = catch_parameter("seismogram_folder", parameters);

    nPoints = nx*ny*nz;

    nxx = nx + 2*nb;
    nyy = ny + 2*nb;
    nzz = nz + 2*nb;

    volsize = nxx*nyy*nzz;

    nBlocks = (int)((volsize + NTHREADS - 1) / NTHREADS);

    set_models();
    set_wavelet();
    set_dampers();
    set_eikonal();
    set_geometry();
    set_snapshots();
    set_seismogram();
    set_wavefields();

    set_modeling_type();
}

void Triclinic::set_uintc_element(std::string element_path, uintc *&dCij, float &max, float &min)
{   
    auto * iModel = new float[nPoints]();
    auto * xModel = new float[volsize]();
    auto * uModel = new uintc[volsize]();
    
    import_binary_float(element_path, iModel, nPoints);
    expand_boundary(iModel, xModel);
    get_compression(xModel, uModel, volsize, max, min);        
    cudaMalloc((void**)&(dCij), volsize*sizeof(uintc));
    cudaMemcpy(dCij, uModel, volsize*sizeof(uintc), cudaMemcpyHostToDevice);

    delete[] iModel;   
    delete[] xModel;
    delete[] uModel;    
}

void Triclinic::set_float_element(std::string element_path, float *&dCij)
{   
    auto * iModel = new float[nPoints]();
    auto * xModel = new float[volsize]();
    
    import_binary_float(element_path, iModel, nPoints);
    expand_boundary(iModel, xModel);
    cudaMalloc((void**)&(dCij), volsize*sizeof(float));
    cudaMemcpy(dCij, xModel, volsize*sizeof(float), cudaMemcpyHostToDevice);

    delete[] iModel;   
    delete[] xModel;    
}

void Triclinic::set_models()
{
    std::string slowness_file = catch_parameter("slowness_file", parameters);
    std::string buoyancy_file = catch_parameter("buoyancy_file", parameters);
    std::string Cijkl_folder = catch_parameter("Cijkl_folder", parameters);
    
    set_float_element(slowness_file, d_S);

    if (compression)
    {
        set_uintc_element(buoyancy_file, dc_B, maxB, minB);

        set_uintc_element(Cijkl_folder + "C11.bin", dc_C11, maxC11, minC11);
        set_uintc_element(Cijkl_folder + "C12.bin", dc_C12, maxC12, minC12);
        set_uintc_element(Cijkl_folder + "C13.bin", dc_C13, maxC13, minC13);
        set_uintc_element(Cijkl_folder + "C14.bin", dc_C14, maxC14, minC14);
        set_uintc_element(Cijkl_folder + "C15.bin", dc_C15, maxC15, minC15);
        set_uintc_element(Cijkl_folder + "C16.bin", dc_C16, maxC16, minC16);
        set_uintc_element(Cijkl_folder + "C22.bin", dc_C22, maxC22, minC22);
        set_uintc_element(Cijkl_folder + "C23.bin", dc_C23, maxC23, minC23);
        set_uintc_element(Cijkl_folder + "C24.bin", dc_C24, maxC24, minC24);
        set_uintc_element(Cijkl_folder + "C25.bin", dc_C25, maxC25, minC25);
        set_uintc_element(Cijkl_folder + "C26.bin", dc_C26, maxC26, minC26);
        set_uintc_element(Cijkl_folder + "C33.bin", dc_C33, maxC33, minC33);
        set_uintc_element(Cijkl_folder + "C34.bin", dc_C34, maxC34, minC34);
        set_uintc_element(Cijkl_folder + "C35.bin", dc_C35, maxC35, minC35);
        set_uintc_element(Cijkl_folder + "C36.bin", dc_C36, maxC36, minC36);
        set_uintc_element(Cijkl_folder + "C44.bin", dc_C44, maxC44, minC44);
        set_uintc_element(Cijkl_folder + "C45.bin", dc_C45, maxC45, minC45);
        set_uintc_element(Cijkl_folder + "C46.bin", dc_C46, maxC46, minC46);
        set_uintc_element(Cijkl_folder + "C55.bin", dc_C55, maxC55, minC55);
        set_uintc_element(Cijkl_folder + "C56.bin", dc_C56, maxC56, minC56);
        set_uintc_element(Cijkl_folder + "C66.bin", dc_C66, maxC66, minC66);
    }
    else 
    {
        set_float_element(buoyancy_file, d_B);

        set_float_element(Cijkl_folder + "C11.bin", d_C11);
        set_float_element(Cijkl_folder + "C12.bin", d_C12);
        set_float_element(Cijkl_folder + "C13.bin", d_C13);
        set_float_element(Cijkl_folder + "C14.bin", d_C14);
        set_float_element(Cijkl_folder + "C15.bin", d_C15);
        set_float_element(Cijkl_folder + "C16.bin", d_C16);
        set_float_element(Cijkl_folder + "C22.bin", d_C22);
        set_float_element(Cijkl_folder + "C23.bin", d_C23);
        set_float_element(Cijkl_folder + "C24.bin", d_C24);
        set_float_element(Cijkl_folder + "C25.bin", d_C25);
        set_float_element(Cijkl_folder + "C26.bin", d_C26);
        set_float_element(Cijkl_folder + "C33.bin", d_C33);
        set_float_element(Cijkl_folder + "C34.bin", d_C34);
        set_float_element(Cijkl_folder + "C35.bin", d_C35);
        set_float_element(Cijkl_folder + "C36.bin", d_C36);
        set_float_element(Cijkl_folder + "C44.bin", d_C44);
        set_float_element(Cijkl_folder + "C45.bin", d_C45);
        set_float_element(Cijkl_folder + "C46.bin", d_C46);
        set_float_element(Cijkl_folder + "C55.bin", d_C55);
        set_float_element(Cijkl_folder + "C56.bin", d_C56);
        set_float_element(Cijkl_folder + "C66.bin", d_C66);
    }
}

void Triclinic::expand_boundary(float * input, float * output)
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

void Triclinic::reduce_boundary(float * input, float * output)
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

void Triclinic::get_compression(float * input, uintc * output, int N, float &max_value, float &min_value)
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
        output[index] = static_cast<uintc>(1.0f + (float)(COMPRESS - 1)*(input[index] - min_value) / (max_value - min_value));
}

void Triclinic::set_wavelet()
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

void Triclinic::set_dampers()
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

void Triclinic::set_eikonal()
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

void Triclinic::set_geometry()
{
    geometry = new Geometry();
    geometry->parameters = parameters;
    geometry->set_parameters();

    h_skw = new float[DGS*DGS*DGS*geometry->nsrc]();

    h_rkwPs = new float[DGS*DGS*DGS*geometry->nrec]();
    h_rkwVx = new float[DGS*DGS*DGS*geometry->nrec]();
    h_rkwVy = new float[DGS*DGS*DGS*geometry->nrec]();
    h_rkwVz = new float[DGS*DGS*DGS*geometry->nrec]();

    set_geometry_weights();
    
    cudaMalloc((void**)&(d_skw), DGS*DGS*DGS*sizeof(float));
    
    cudaMalloc((void**)&(d_rkwPs), DGS*DGS*DGS*sizeof(float));
    cudaMalloc((void**)&(d_rkwVx), DGS*DGS*DGS*sizeof(float));
    cudaMalloc((void**)&(d_rkwVy), DGS*DGS*DGS*sizeof(float));
    cudaMalloc((void**)&(d_rkwVz), DGS*DGS*DGS*sizeof(float));
}

void Triclinic::set_snapshots()
{
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
}

void Triclinic::set_seismogram()
{
    h_seismogram_Ps = new float[nt*geometry->nrec]();
    h_seismogram_Vx = new float[nt*geometry->nrec]();
    h_seismogram_Vy = new float[nt*geometry->nrec]();
    h_seismogram_Vz = new float[nt*geometry->nrec]();

    cudaMalloc((void**)&(d_seismogram_Ps), nt*geometry->nrec*sizeof(float));
    cudaMalloc((void**)&(d_seismogram_Vx), nt*geometry->nrec*sizeof(float));
    cudaMalloc((void**)&(d_seismogram_Vy), nt*geometry->nrec*sizeof(float));
    cudaMalloc((void**)&(d_seismogram_Vz), nt*geometry->nrec*sizeof(float));
}

void Triclinic::set_wavefields()
{
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
}

void Triclinic::run_wave_propagation()
{
    for (srcId = 0; srcId < geometry->nsrc; srcId++)
    {
        initialization();                      
        time_propagation();
        show_information();
        export_seismograms();
    }
}

void Triclinic::initialization()
{
    sx = geometry->xsrc[srcId];
    sy = geometry->ysrc[srcId];
    sz = geometry->zsrc[srcId];

    sIdx = (int)((sx + 0.5f*dx) / dx) + nb;
    sIdy = (int)((sy + 0.5f*dy) / dy) + nb;
    sIdz = (int)((sz + 0.5f*dz) / dz) + nb;

    cudaMemcpy(d_skw, h_skw + srcId*DGS*DGS*DGS, DGS*DGS*DGS*sizeof(float), cudaMemcpyHostToDevice);

    compute_eikonal();

    if (snapshot)
    {
        snapCount = 0;

        export_travelTimes();
    }

    wavefield_refresh();
}

void Triclinic::show_information()
{
    auto clear = system("clear");
    
    std::cout << "-----------------------------------------------------------\n";
    std::cout << " \033[34mEikoStagTriX3D\033[0;0m -------------------------------------------\n";
    std::cout << "-----------------------------------------------------------\n\n";

    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << ", x = " << (nx - 1) * dx <<", y = " << (ny - 1) * dy << ") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nsrc << " in total\n\n";

    std::cout << "Current shot position: (z = " << sz << ", x = " << sx << ", y = " << sy << ") m\n\n";

    std::cout << modeling_name << "\n";
}

void Triclinic::time_propagation()
{
    for (timeId = 0; timeId < nt + tlag; timeId++)
    {
        source_injection();
        compute_velocity();
        compute_pressure();
        compute_snapshots();
        compute_seismogram();    
        show_time_progress();
    }
}

void Triclinic::compute_eikonal()
{    
    if (eikonalClip)
    {
        eikonal_solver();
        
        if (compression)
        {
            uintc_quasi_slowness<<<nBlocks,NTHREADS>>>(d_T,d_S,dx,dy,dz,sIdx,sIdy,sIdz,nxx,nyy,nzz,nb,dc_C11,dc_C12,dc_C13,dc_C14,dc_C15,
                                                       dc_C16,dc_C22,dc_C23,dc_C24,dc_C25,dc_C26,dc_C33,dc_C34,dc_C35,dc_C36,dc_C44,dc_C45,
                                                       dc_C46,dc_C55,dc_C56,dc_C66,minC11,maxC11,minC12,maxC12,minC13,maxC13,minC14,maxC14, 
                                                       minC15,maxC15,minC16,maxC16,minC22,maxC22,minC23,maxC23,minC24,maxC24,minC25,maxC25,
                                                       minC26,maxC26,minC33,maxC33,minC34,maxC34,minC35,maxC35,minC36,maxC36,minC44,maxC44,
                                                       minC45,maxC45,minC46,maxC46,minC55,maxC55,minC56,maxC56,minC66,maxC66);        
        }
        else
        {
            float_quasi_slowness<<<nBlocks,NTHREADS>>>(d_T,d_S,dx,dy,dz,sIdx,sIdy,sIdz,nxx,nyy,nzz,nb,d_C11,d_C12,d_C13,d_C14,d_C15,d_C16,d_C22,
                                                       d_C23,d_C24,d_C25,d_C26,d_C33,d_C34,d_C35,d_C36,d_C44,d_C45,d_C46,d_C55,d_C56,d_C66);
        }
        
        eikonal_solver();
    } 
}

void Triclinic::eikonal_solver()
{
    dim3 grid(1,1,1);
    dim3 block(MESHDIM,MESHDIM,MESHDIM);

    time_set<<<nBlocks,NTHREADS>>>(d_T, volsize);
    time_init<<<grid,block>>>(d_T,d_S,sx,sy,sz,dx,dy,dz,sIdx,sIdy,sIdz,nxx,nzz,nb);

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

void Triclinic::wavefield_refresh()
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
    
    cudaMemset(d_seismogram_Ps, 0.0f, nt*geometry->nrec*sizeof(float));
    cudaMemset(d_seismogram_Vx, 0.0f, nt*geometry->nrec*sizeof(float));
    cudaMemset(d_seismogram_Vy, 0.0f, nt*geometry->nrec*sizeof(float));
    cudaMemset(d_seismogram_Vz, 0.0f, nt*geometry->nrec*sizeof(float));
}

void Triclinic::export_travelTimes()
{
    if (eikonalClip)
    {
        cudaMemcpy(snapshot_in, d_T, volsize*sizeof(float), cudaMemcpyDeviceToHost);
        reduce_boundary(snapshot_in, snapshot_out);
        export_binary_float(snapshot_folder + "triclinic_eikonal_" + std::to_string(nz) + "x" + std::to_string(nx) + "x" + std::to_string(ny) + "_shot_" + std::to_string(srcId+1) + ".bin", snapshot_out, nPoints);    
    }
}

void Triclinic::source_injection()
{
    dim3 grid(1,1,1);
    dim3 block(DGS,DGS,DGS);

    if (timeId < nt) apply_pressure_source<<<grid,block>>>(d_Txx, d_Tyy, d_Tzz, d_skw, d_wavelet, sIdx, sIdy, sIdz, timeId, nxx, nzz, dx, dy, dz);     
}

void Triclinic::compute_snapshots()
{
    if (snapshot)
    {
        if (snapCount < snapId.size())
        {
            if ((timeId-tlag) == snapId[snapCount])
            {
                cudaMemcpy(snapshot_in, d_P, volsize*sizeof(float), cudaMemcpyDeviceToHost);
                reduce_boundary(snapshot_in, snapshot_out);
                export_binary_float(snapshot_folder + modeling_type + "_snapshot_step" + std::to_string(timeId-tlag) + "_" + std::to_string(nz) + "x" + std::to_string(nx) + "x" + std::to_string(ny) + "_shot_" + std::to_string(srcId+1) + ".bin", snapshot_out, nPoints);    
                
                ++snapCount;
            }
        }
    }
}

void Triclinic::compute_seismogram()
{
    dim3 grid(1,1,1);
    dim3 block(DGS,DGS,DGS);

    if (timeId > tlag)
    {
        for (recId = 0; recId < geometry->nrec; recId++)
        {
            rx = geometry->xrec[recId];
            ry = geometry->yrec[recId];
            rz = geometry->zrec[recId];

            rIdx = (int)((rx + 0.5f*dx) / dx) + nb;
            rIdy = (int)((ry + 0.5f*dy) / dy) + nb;
            rIdz = (int)((rz + 0.5f*dz) / dz) + nb;

            cudaMemcpy(d_rkwPs, h_rkwPs + recId*DGS*DGS*DGS, DGS*DGS*DGS*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rkwVx, h_rkwVx + recId*DGS*DGS*DGS, DGS*DGS*DGS*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rkwVy, h_rkwVy + recId*DGS*DGS*DGS, DGS*DGS*DGS*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rkwVz, h_rkwVz + recId*DGS*DGS*DGS, DGS*DGS*DGS*sizeof(float), cudaMemcpyHostToDevice);

            compute_seismogram_GPU<<<grid,block>>>(d_P, d_seismogram_Ps, d_rkwPs, rIdx, rIdy, rIdz, timeId, tlag, recId, nt, nxx, nzz);     
            compute_seismogram_GPU<<<grid,block>>>(d_Vx, d_seismogram_Vx, d_rkwVx, rIdx, rIdy, rIdz, timeId, tlag, recId, nt, nxx, nzz);     
            compute_seismogram_GPU<<<grid,block>>>(d_Vy, d_seismogram_Vy, d_rkwVy, rIdx, rIdy, rIdz, timeId, tlag, recId, nt, nxx, nzz);     
            compute_seismogram_GPU<<<grid,block>>>(d_Vz, d_seismogram_Vz, d_rkwVz, rIdx, rIdy, rIdz, timeId, tlag, recId, nt, nxx, nzz);     
        }
    }
}

void Triclinic::show_time_progress()
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

void Triclinic::export_seismograms()
{   
    cudaMemcpy(h_seismogram_Ps, d_seismogram_Ps, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_seismogram_Vx, d_seismogram_Vx, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_seismogram_Vy, d_seismogram_Vy, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_seismogram_Vz, d_seismogram_Vz, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);    

    std::string seismPs = seismogram_folder + modeling_type + "_Ps_nStations" + std::to_string(geometry->nrec) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(srcId+1) + ".bin";
    std::string seismVx = seismogram_folder + modeling_type + "_Vx_nStations" + std::to_string(geometry->nrec) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(srcId+1) + ".bin";
    std::string seismVy = seismogram_folder + modeling_type + "_Vy_nStations" + std::to_string(geometry->nrec) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(srcId+1) + ".bin";
    std::string seismVz = seismogram_folder + modeling_type + "_Vz_nStations" + std::to_string(geometry->nrec) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(srcId+1) + ".bin";

    export_binary_float(seismPs, h_seismogram_Ps, nt*geometry->nrec);    
    export_binary_float(seismVx, h_seismogram_Vx, nt*geometry->nrec);    
    export_binary_float(seismVy, h_seismogram_Vy, nt*geometry->nrec);    
    export_binary_float(seismVz, h_seismogram_Vz, nt*geometry->nrec);    
}

int Triclinic::iDivUp(int a, int b) 
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

__global__ void float_quasi_slowness(float * T, float * S, float dx, float dy, float dz, int sIdx, int sIdy, int sIdz, int nxx, int nyy, int nzz, int nb,
                                     float * C11, float * C12, float * C13, float * C14, float * C15, float * C16, float * C22, float * C23, float * C24, float * C25, 
                                     float * C26, float * C33, float * C34, float * C35, float * C36, float * C44, float * C45, float * C46, float * C55, float * C56, 
                                     float * C66)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;    
    int i = (int) (index - j*nzz - k*nxx*nzz);  

    const int n = 3;
    const int v = 6;

    float p[n];
    float C[v*v];
    float G[n*n];
    float Gv[n];

    int voigt_map[n][n] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};

    if ((i >= nb) && (i < nzz-nb) && (j >= nb) && (j < nxx-nb) && (k >= nb) && (k < nyy-nb))
    {
        if (!((i == sIdz) && (j == sIdx) && (k == sIdy)))    
        {
            float dTz = 0.5f*(T[(i+1) + j*nzz + k*nxx*nzz] - T[(i-1) + j*nzz + k*nxx*nzz]) / dz;
            float dTx = 0.5f*(T[i + (j+1)*nzz + k*nxx*nzz] - T[i + (j-1)*nzz + k*nxx*nzz]) / dx;
            float dTy = 0.5f*(T[i + j*nzz + (k+1)*nxx*nzz] - T[i + j*nzz + (k-1)*nxx*nzz]) / dy;

            float norm = sqrtf(dTx*dTx + dTy*dTy + dTz*dTz);

            p[0] = dTx / norm;
            p[1] = dTy / norm;
            p[2] = dTz / norm;
            
            C[0+0*v] = C11[index]; C[0+1*v] = C12[index]; C[0+2*v] = C13[index]; C[0+3*v] = C14[index]; C[0+4*v] = C15[index]; C[0+5*v] = C16[index];
            C[1+0*v] = C12[index]; C[1+1*v] = C22[index]; C[1+2*v] = C23[index]; C[1+3*v] = C24[index]; C[1+4*v] = C25[index]; C[1+5*v] = C26[index];
            C[2+0*v] = C13[index]; C[2+1*v] = C23[index]; C[2+2*v] = C33[index]; C[2+3*v] = C34[index]; C[2+4*v] = C35[index]; C[2+5*v] = C36[index];
            C[3+0*v] = C14[index]; C[3+1*v] = C24[index]; C[3+2*v] = C34[index]; C[3+3*v] = C44[index]; C[3+4*v] = C45[index]; C[3+5*v] = C46[index];
            C[4+0*v] = C15[index]; C[4+1*v] = C25[index]; C[4+2*v] = C35[index]; C[4+3*v] = C45[index]; C[4+4*v] = C55[index]; C[4+5*v] = C56[index];
            C[5+0*v] = C16[index]; C[5+1*v] = C26[index]; C[5+2*v] = C36[index]; C[5+3*v] = C46[index]; C[5+4*v] = C56[index]; C[5+5*v] = C66[index];

            float Ro = C33[index]*S[index]*S[index];    

            for (int indp = 0; indp < v*v; indp++)
                C[indp] = C[indp] / Ro / Ro;

            for (int indp = 0; indp < n*n; indp++) 
                G[indp] = 0.0f; 

            for (int ip = 0; ip < n; ip++) 
            {
                for (int jp = 0; jp < n; jp++) 
                {
                    for (int kp = 0; kp < n; kp++) 
                    {
                        for (int lp = 0; lp < n; lp++) 
                        {
                            int I = voigt_map[ip][kp];
                            int J = voigt_map[jp][lp];

                            G[ip + jp*n] += C[I + J*v]*p[kp]*p[lp];
                        }
                    }
                }
            }

            float a = -(G[0] + G[4] + G[8]);
    
            float b = G[0]*G[4] + G[4]*G[8] + 
                      G[0]*G[8] - G[3]*G[1] - 
                      G[6]*G[6] - G[7]*G[5];
            
            float c = -(G[0]*(G[4]*G[8] - G[7]*G[5]) -
                        G[3]*(G[1]*G[8] - G[7]*G[6]) +
                        G[6]*(G[1]*G[5] - G[4]*G[6]));

            float p = b - (a*a)/3.0f;
            float q = (2.0f*a*a*a)/27.0f - (a*b)/3.0f + c;

            float detG = 0.25f*(q*q) + (p*p*p)/27.0f;

            if (detG > 0) 
            {
                float u = cbrtf(-0.5f*q + sqrtf(detG));
                float v = cbrtf(-0.5f*q - sqrtf(detG));
                
                Gv[0] = u + v - a/3.0f;
            } 
            else if (detG == 0) 
            {       
                float u = cbrtf(-0.5f*q);

                Gv[0] = 2.0f*u - a/3.0f;
                Gv[1] =-1.0f*u - a/3.0f;         
            } 
            else  
            {
                float r = sqrtf(-p*p*p/27.0f);
                float phi = acosf(-0.5f*q/r);
                
                r = 2.0f*cbrtf(r);

                Gv[0] = r*cosf(phi/3.0f) - a/3.0f;
                Gv[1] = r*cosf((phi + 2.0f*M_PI)/3.0f) - a/3.0f;  
                Gv[2] = r*cosf((phi + 4.0f*M_PI)/3.0f) - a/3.0f;      
            }
            
            float aux;

            if (Gv[0] < Gv[1]) {aux = Gv[0]; Gv[0] = Gv[1]; Gv[1] = aux;} 
            if (Gv[1] < Gv[2]) {aux = Gv[1]; Gv[1] = Gv[2]; Gv[2] = aux;}
            if (Gv[0] < Gv[1]) {aux = Gv[0]; Gv[0] = Gv[1]; Gv[1] = aux;}    

            S[index] = 1.0f / sqrtf(Gv[0] * Ro);
        }
    }
}

__global__ void uintc_quasi_slowness(float * T, float * S, float dx, float dy, float dz, int sIdx, int sIdy, int sIdz, int nxx, int nyy, int nzz, int nb,
                                     uintc * C11, uintc * C12, uintc * C13, uintc * C14, uintc * C15, uintc * C16, uintc * C22, uintc * C23, uintc * C24, uintc * C25, 
                                     uintc * C26, uintc * C33, uintc * C34, uintc * C35, uintc * C36, uintc * C44, uintc * C45, uintc * C46, uintc * C55, uintc * C56, 
                                     uintc * C66, float minC11, float maxC11, float minC12, float maxC12, float minC13, float maxC13, float minC14, float maxC14, 
                                     float minC15, float maxC15, float minC16, float maxC16, float minC22, float maxC22, float minC23, float maxC23, float minC24, 
                                     float maxC24, float minC25, float maxC25, float minC26, float maxC26, float minC33, float maxC33, float minC34, float maxC34, 
                                     float minC35, float maxC35, float minC36, float maxC36, float minC44, float maxC44, float minC45, float maxC45, float minC46, 
                                     float maxC46, float minC55, float maxC55, float minC56, float maxC56, float minC66, float maxC66)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;    
    int i = (int) (index - j*nzz - k*nxx*nzz);  

    const int n = 3;
    const int v = 6;

    float p[n];
    float C[v*v];
    float G[n*n];
    float Gv[n];

    int voigt_map[n][n] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};

    if ((i >= nb) && (i < nzz-nb) && (j >= nb) && (j < nxx-nb) && (k >= nb) && (k < nyy-nb))
    {
        if (!((i == sIdz) && (j == sIdx) && (k == sIdy)))    
        {
            float dTz = 0.5f*(T[(i+1) + j*nzz + k*nxx*nzz] - T[(i-1) + j*nzz + k*nxx*nzz]) / dz;
            float dTx = 0.5f*(T[i + (j+1)*nzz + k*nxx*nzz] - T[i + (j-1)*nzz + k*nxx*nzz]) / dx;
            float dTy = 0.5f*(T[i + j*nzz + (k+1)*nxx*nzz] - T[i + j*nzz + (k-1)*nxx*nzz]) / dy;

            float norm = sqrtf(dTx*dTx + dTy*dTy + dTz*dTz);

            p[0] = dTx / norm;
            p[1] = dTy / norm;
            p[2] = dTz / norm;
            
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

            C[0+0*v] = c11; C[0+1*v] = c12; C[0+2*v] = c13; C[0+3*v] = c14; C[0+4*v] = c15; C[0+5*v] = c16;
            C[1+0*v] = c12; C[1+1*v] = c22; C[1+2*v] = c23; C[1+3*v] = c24; C[1+4*v] = c25; C[1+5*v] = c26;
            C[2+0*v] = c13; C[2+1*v] = c23; C[2+2*v] = c33; C[2+3*v] = c34; C[2+4*v] = c35; C[2+5*v] = c36;
            C[3+0*v] = c14; C[3+1*v] = c24; C[3+2*v] = c34; C[3+3*v] = c44; C[3+4*v] = c45; C[3+5*v] = c46;
            C[4+0*v] = c15; C[4+1*v] = c25; C[4+2*v] = c35; C[4+3*v] = c45; C[4+4*v] = c55; C[4+5*v] = c56;
            C[5+0*v] = c16; C[5+1*v] = c26; C[5+2*v] = c36; C[5+3*v] = c46; C[5+4*v] = c56; C[5+5*v] = c66;

            float Ro = c33*S[index]*S[index];    

            for (int indp = 0; indp < v*v; indp++)
                C[indp] = C[indp] / Ro / Ro;

            for (int indp = 0; indp < n*n; indp++) 
                G[indp] = 0.0f; 

            for (int ip = 0; ip < n; ip++) 
            {
                for (int jp = 0; jp < n; jp++) 
                {
                    for (int kp = 0; kp < n; kp++) 
                    {
                        for (int lp = 0; lp < n; lp++) 
                        {
                            int I = voigt_map[ip][kp];
                            int J = voigt_map[jp][lp];

                            G[ip + jp*n] += C[I + J*v]*p[kp]*p[lp];
                        }
                    }
                }
            }

            float a = -(G[0] + G[4] + G[8]);
    
            float b = G[0]*G[4] + G[4]*G[8] + 
                      G[0]*G[8] - G[3]*G[1] - 
                      G[6]*G[6] - G[7]*G[5];
            
            float c = -(G[0]*(G[4]*G[8] - G[7]*G[5]) -
                        G[3]*(G[1]*G[8] - G[7]*G[6]) +
                        G[6]*(G[1]*G[5] - G[4]*G[6]));

            float p = b - (a*a)/3.0f;
            float q = (2.0f*a*a*a)/27.0f - (a*b)/3.0f + c;

            float detG = 0.25f*(q*q) + (p*p*p)/27.0f;

            if (detG > 0) 
            {
                float u = cbrtf(-0.5f*q + sqrtf(detG));
                float v = cbrtf(-0.5f*q - sqrtf(detG));
                
                Gv[0] = u + v - a/3.0f;
            } 
            else if (detG == 0) 
            {       
                float u = cbrtf(-0.5f*q);

                Gv[0] = 2.0f*u - a/3.0f;
                Gv[1] =-1.0f*u - a/3.0f;         
            } 
            else  
            {
                float r = sqrtf(-p*p*p/27.0f);
                float phi = acosf(-0.5f*q/r);
                
                r = 2.0f*cbrtf(r);

                Gv[0] = r*cosf(phi/3.0f) - a/3.0f;
                Gv[1] = r*cosf((phi + 2.0f*M_PI)/3.0f) - a/3.0f;  
                Gv[2] = r*cosf((phi + 4.0f*M_PI)/3.0f) - a/3.0f;      
            }
            
            float aux;

            if (Gv[0] < Gv[1]) {aux = Gv[0]; Gv[0] = Gv[1]; Gv[1] = aux;} 
            if (Gv[1] < Gv[2]) {aux = Gv[1]; Gv[1] = Gv[2]; Gv[2] = aux;}
            if (Gv[0] < Gv[1]) {aux = Gv[0]; Gv[0] = Gv[1]; Gv[1] = aux;}    

            S[index] = 1.0f / sqrtf(Gv[0] * Ro);
        }
    }
}

__global__ void apply_pressure_source(float * Txx, float * Tyy, float * Tzz, float * skw, float * wavelet, int sIdx, int sIdy, int sIdz, int tId, int nxx, int nzz, float dx, float dy, float dz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    int zi = sIdz + (i - 3);
    int xi = sIdx + (j - 3);
    int yi = sIdy + (k - 3);

    int index = zi + xi*nzz + yi*nxx*nzz;
        
    Txx[index] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);
    Tyy[index] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);
    Tzz[index] += skw[i + j*DGS + k*DGS*DGS]*wavelet[tId] / (dx*dy*dz);           
}

__global__ void compute_seismogram_GPU(float * WF, float * seismogram, float * rkw, int rIdx, int rIdy, int rIdz, int tId, int tlag, int recId, int nt, int nxx, int nzz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    int yi = rIdy + (k - 3);
    int xi = rIdx + (j - 3);
    int zi = rIdz + (i - 3);

    int index = zi + xi*nzz + yi*nxx*nzz;
    
    // idk why it's not working
    //seismogram[(tId - tlag) + recId*nt] += rkw[i + j*DGS + k*DGS*DGS]*WF[index];

    if ((i == 4) && (j == 4) && (k == 4))    
        seismogram[(tId - tlag) + recId*nt] = WF[index];
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
