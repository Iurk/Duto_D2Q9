#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

#include "paths.h"
#include "LBM.h"
#include "dados.h"

using namespace myGlobals;

// Input data
__constant__ unsigned int q, Nx_d, Ny_d;
__constant__ double rho0_d, u_max_d, nu_d, tau_d, mi_ar_d;

//Lattice Data
__constant__ double cs_d, w0_d, ws_d, wd_d;
__device__ int *ex_d;
__device__ int *ey_d;

// Mesh data
__device__ bool *solid_d;

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

__global__ void gpu_init_equilibrium(double*, double*, double*, double*);
__global__ void gpu_stream_collide_save(double*, double*, double*, double*, double*, double*, bool);
__global__ void gpu_compute_flow_properties(unsigned int, double*, double*, double*, double*);
__global__ void gpu_print_mesh(int);
__global__ void gpu_initialization(double*, double);

// Poiseulle Flow
__device__ void poiseulle_eval(unsigned int t, unsigned int x, unsigned int y, double *u){

	double gradP = -8*u_max_d*mi_ar_d/(pow(Ny_d, 2) - 2*Ny_d);

	double ux = (-1/(2*mi_ar_d))*(gradP)*((Ny_d - 1)*y - pow(y, 2));

	*u = ux;
}

// Boundary Conditions
__device__ void gpu_bounce_back(unsigned int x, unsigned int y, double *f){
	
	if(y == 0){
		f[gpu_fieldn_index(x, y, 2)] = f[gpu_fieldn_index(x, y, 4)];
		f[gpu_fieldn_index(x, y, 5)] = f[gpu_fieldn_index(x, y, 7)];
		f[gpu_fieldn_index(x, y, 6)] = f[gpu_fieldn_index(x, y, 8)];
	}

	if(y == Ny_d-1){
		f[gpu_fieldn_index(x, y, 4)] = f[gpu_fieldn_index(x, y, 2)];
		f[gpu_fieldn_index(x, y, 7)] = f[gpu_fieldn_index(x, y, 5)];
		f[gpu_fieldn_index(x, y, 8)] = f[gpu_fieldn_index(x, y, 6)];
	}
}
__host__ void init_equilibrium(double *f1, double *r, double *u, double *v){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_init_equilibrium<<< grid, block >>>(f1, r, u, v);
	getLastCudaError("gpu_init_equilibrium kernel error");
}

__global__ void gpu_init_equilibrium(double *f1, double *r, double *u, double *v){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];

	double A = 1.0/(cs_d*cs_d);
	double B = 1.0/(2.0*cs_d*cs_d);

	double w0r = w0_d*rho;
	double wsr = ws_d*rho;
	double wdr = wd_d*rho;
	double omusq = 1.0 - B*(ux*ux + uy*uy);

	double Wrho[] = {w0r, wsr, wsr, wsr, wsr, wdr, wdr, wdr, wdr};

	for(int n = 0; n < q; ++n){
		double eidotu = ux*ex_d[n] + uy*ey_d[n];
		f1[gpu_fieldn_index(x, y, n)] = Wrho[n]*(omusq + A*eidotu*(1.0 + B*eidotu));
	}
}

__host__ void stream_collide_save(double *f1, double *f2, double *f1neq, double *r, double *u, double *v, bool save){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	//dim3 grid(1,1,1);
	//dim3 block(1,1,1);

	gpu_stream_collide_save<<< grid, block >>>(f1, f2, f1neq, r, u, v, save);
	getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_stream_collide_save(double *f1, double *f2, double *f1neq, double *r, double *u, double *v, bool save){

	const double omega = 1.0/tau_d;

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
/*
	unsigned int xf = (x + 1)%Nx_d;		// Forward
	unsigned int yf = (y + 1)%Ny_d;		// Forward
	unsigned int xb = (Nx_d + x - 1)%Nx_d;	// Backward
	unsigned int yb = (Ny_d + y - 1)%Ny_d; // Backward

	// Streaming step
	double ft0 = f1[gpu_fieldn_index(x, y, 0)];
	double ft1 = f1[gpu_fieldn_index(xb, y, 1)];
	double ft2 = f1[gpu_fieldn_index(x, yb, 2)];
	double ft3 = f1[gpu_fieldn_index(xf, y, 3)];
	double ft4 = f1[gpu_fieldn_index(x, yf, 4)];
	double ft5 = f1[gpu_fieldn_index(xb, yb, 5)];
	double ft6 = f1[gpu_fieldn_index(xf, yb, 6)];
	double ft7 = f1[gpu_fieldn_index(xf, yf, 7)];
	double ft8 = f1[gpu_fieldn_index(xb, yf, 8)];

	double f[] = {ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8};
*/
	double rho = 0, ux_i = 0, uy_i = 0;

	for(int n = 0; n < q; ++n){
		rho += f1[gpu_fieldn_index(x, y, n)];
		ux_i += f1[gpu_fieldn_index(x, y, n)]*ex_d[n];
		uy_i += f1[gpu_fieldn_index(x, y, n)]*ey_d[n];
	}

	double rhoinv = 1.0/rho;

	double ux = rhoinv*ux_i;
	double uy = rhoinv*uy_i;

	if(save){
		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux;
		v[gpu_scalar_index(x, y)] = uy;
	}

	double A = 1.0/(cs_d*cs_d);
	double B = 1.0/(2.0*cs_d*cs_d);

	double w0r = w0_d*rho;
	double wsr = ws_d*rho;
	double wdr = wd_d*rho;

	double W[] = {w0_d, ws_d, ws_d, ws_d, ws_d, wd_d, wd_d, wd_d, wd_d};
	double Wrho[] = {w0r, wsr, wsr, wsr, wsr, wdr, wdr, wdr, wdr};

	double omusq = 1.0 - B*(ux*ux + uy*uy);

	// Approximation of fneq
	for(int n = 0; n < q; ++n){
		double eidotu = ux*ex_d[n] + uy*ey_d[n];
		double feq = Wrho[n]*(omusq + A*eidotu*(1.0 + B*eidotu));
		f1neq[gpu_fieldn_index(x, y, n)] = f1[gpu_fieldn_index(x, y, n)] - feq;
	}

	// Calculating the Viscous stress tensor
	double tauxx = 0, tauxy = 0, tauyy = 0;
	for(int n = 1; n < q; ++n){
		tauxx += f1neq[gpu_fieldn_index(x, y, n)]*ex_d[n]*ex_d[n];
		tauxy += f1neq[gpu_fieldn_index(x, y, n)]*ex_d[n]*ey_d[n];
		tauyy += f1neq[gpu_fieldn_index(x, y, n)]*ey_d[n]*ey_d[n];
	}

	for(int n = 0; n < q; ++n){
		f1neq[gpu_fieldn_index(x, y, n)] = B*W[n]*(tauxx*(A*ex_d[n]*ex_d[n] - 1.0) + 2.0*tauxy*A*ex_d[n]*ey_d[n] + tauyy*(A*ey_d[n]*ey_d[n] - 1.0));
		double eidotu = ux*ex_d[n] + uy*ey_d[n];
		double feq = Wrho[n]*(omusq + A*eidotu*(1.0 + B*eidotu));
		f1[gpu_fieldn_index(x, y, n)] = (1.0 - omega)*f1neq[gpu_fieldn_index(x, y, n)] + feq;
	}

	double gradP = -8*u_max_d*mi_ar_d/(pow(Ny_d, 2) - 2*Ny_d);
	double gradRho = (Nx_d/(pow(cs_d, 2)))*gradP;

	double rho_in = rho0_d;
	double rho_out = rho_in + gradRho;

	if(x == 0){
		double ux_in = u[gpu_scalar_index(Nx_d-1, y)];
		double uy_in = v[gpu_scalar_index(Nx_d-1, y)];
		for(int n = 1; n < q; ++n){
			double eidotu = ux_in*ex_d[n] + uy_in*ey_d[n];
			double feq_in = W[n]*rho_in*(omusq + A*eidotu*(1.0 + B*eidotu));
			f1[gpu_fieldn_index(0, y, n)] = feq_in + f1neq[gpu_fieldn_index(Nx_d-1, y, n)];		// Periodic with pressure Inlet
		}
	}

	if(x == Nx_d-1){
		double ux_out = u[gpu_scalar_index(0, y)];
		double uy_out = v[gpu_scalar_index(0, y)];
		for(int n = 1; n < q; ++n){
			double eidotu = ux_out*ex_d[n] + uy_out*ey_d[n];
			double feq_out = W[n]*rho_out*(omusq + A*eidotu*(1.0 + B*eidotu));
			f1[gpu_fieldn_index(Nx_d-1, y, n)] = feq_out + f1neq[gpu_fieldn_index(0, y, n)];	// Periodic with pressure Outlet
		}
	}

	// Streaming Step
	int x_att, y_att;
	for(int n = 0; n < q; ++n){
		x_att = (x + ex_d[n] + Nx_d)%Nx_d;
		y_att = (y + ey_d[n] + Ny_d)%Ny_d;

		f2[gpu_fieldn_index(x_att, y_att, n)] = f1[gpu_fieldn_index(x, y, n)];
	}

	bool node_solid = solid_d[gpu_scalar_index(x, y)];

	// Applying Boundary Conditions
	if(node_solid){
		gpu_bounce_back(x, y, f2);
	}
}

__host__ std::vector<double> compute_flow_properties(unsigned int t, double *r, double *u, double *v, std::vector<double> prop, double *prop_gpu, double *prop_host){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_compute_flow_properties<<< grid, block, 3*block.x*sizeof(double) >>>(t, r, u, v, prop_gpu);
	getLastCudaError("gpu_compute_flow_properties kernel error");

	size_t prop_size_bytes = 3*grid.x*grid.y*sizeof(double);
	checkCudaErrors(cudaMemcpy(prop_host, prop_gpu, prop_size_bytes, cudaMemcpyDeviceToHost));

	double E = 0.0;

	double sumuxe2 = 0.0;
	double sumuxa2 = 0.0;

	for(unsigned int i = 0; i < grid.x*grid.y; ++i){

		E += prop_host[3*i];

		sumuxe2  += prop_host[3*i+1];
		sumuxa2  += prop_host[3*i+2];
	}

	prop.push_back(E);
	prop.push_back(sqrt(sumuxe2/sumuxa2));

	return prop;
}

__global__ void gpu_compute_flow_properties(unsigned int t, double *r, double *u, double *v, double *prop_gpu){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	extern __shared__ double data[];

	double *E = data;
    double *uxe2  = data + 1*blockDim.x;
    double *uxa2  = data + 2*blockDim.x;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];

	E[threadIdx.x] = rho*(ux*ux + uy*uy);

	// compute analytical results
    double uxa;
    poiseulle_eval(t, x, y, &uxa);

     // compute terms for L2 error
    uxe2[threadIdx.x]  = (ux - uxa)*(ux - uxa);
    uxa2[threadIdx.x]  = uxa*uxa;

	__syncthreads();

	if (threadIdx.x == 0){
		
		size_t idx = 3*(gridDim.x*blockIdx.y + blockIdx.x);

		for(int n = 0; n < 3; ++n){
			prop_gpu[idx+n] = 0.0;
		}

		for(int i = 0; i < blockDim.x; ++i){
			prop_gpu[idx  ] += E[i];
            prop_gpu[idx+1] += uxe2[i];
            prop_gpu[idx+2] += uxa2[i];
		}
	}
}

__host__ std::vector<double> report_flow_properties(unsigned int t, double *rho, double *ux, double *uy,
									 double *prop_gpu, double *prop_host, bool msg, bool computeFlowProperties){

	std::vector<double> prop;
	prop = compute_flow_properties(t, rho, ux, uy, prop, prop_gpu, prop_host);

	if(msg){
		if(computeFlowProperties){
			printf("%u \t %g \t %g\n", t, prop[0], prop[1]);
		}

		if(!quiet){
			printf("Completed timestep %d\n", t);
		}
	}
	
	return prop;
}

__host__ void save_scalar(const std::string name, double *scalar_gpu, double *scalar_host, unsigned int n){

	std::ostringstream path, filename;

	std::string ext = ".dat";

	int ndigits = floor(log10((double)NSTEPS) + 1.0);

	const char* path_results_c = strdup(folder.c_str());

	DIR *dir_results = opendir(path_results_c);
	if(ENOENT == errno){
		mkdir(path_results_c, ACCESSPERMS);
	}

	closedir(dir_results);

	path << folder << name << "/";
	const char* path_c = strdup(path.str().c_str());

	DIR *dir = opendir(path_c);
	if(ENOENT == errno){
		mkdir(path_c, ACCESSPERMS);
	}

	closedir(dir);

	filename << path.str() << name << std::setfill('0') << std::setw(ndigits) << n << ext;
	const char* filename_c = strdup(filename.str().c_str());

	checkCudaErrors(cudaMemcpy(scalar_host, scalar_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

	FILE* fout = fopen(filename_c, "wb+");

	fwrite(scalar_host, 1, mem_size_scalar, fout);

	if(ferror(fout)){
		fprintf(stderr, "Error saving to %s\n", filename_c);
		perror("");
	}
	else{
		if(!quiet){
			printf("Saved to %s\n", filename_c);
		}
	}
	fclose(fout);
}

void wrapper_input(unsigned int *nx, unsigned int *ny, double *rho, double *u, double *nu, const double *tau, const double *mi_ar){
	checkCudaErrors(cudaMemcpyToSymbol(Nx_d, nx, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(Ny_d, ny, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(rho0_d, rho, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(u_max_d, u, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(nu_d, nu, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(tau_d, tau, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(mi_ar_d, mi_ar, sizeof(double)));
}

void wrapper_lattice(unsigned int *ndir, double *c, double *w_0, double *w_s, double *w_d){
	checkCudaErrors(cudaMemcpyToSymbol(q, ndir, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(cs_d, c, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(w0_d, w_0, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(ws_d, w_s, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(wd_d, w_d, sizeof(double)));
}

__host__ int* generate_e(int *e, std::string mode){

	int *temp_e;

	size_t mem_e = ndir*sizeof(int);

	checkCudaErrors(cudaMalloc(&temp_e, mem_e));
	checkCudaErrors(cudaMemcpy(temp_e, e, mem_e, cudaMemcpyHostToDevice));

	if(mode == "x"){
		checkCudaErrors(cudaMemcpyToSymbol(ex_d, &temp_e, sizeof(temp_e)));
	}
	else if(mode == "y"){
		checkCudaErrors(cudaMemcpyToSymbol(ey_d, &temp_e, sizeof(temp_e)));
	}

	return temp_e;
}

__host__ bool* generate_mesh(bool *mesh, std::string mode){

	int mode_num;
	bool *temp_mesh;

	checkCudaErrors(cudaMalloc(&temp_mesh, mem_mesh));
	checkCudaErrors(cudaMemcpy(temp_mesh, mesh, mem_mesh, cudaMemcpyHostToDevice));
	

	if(mode == "solid"){
		checkCudaErrors(cudaMemcpyToSymbol(solid_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 1;
	}

	if(meshprint){
		gpu_print_mesh<<< 1, 1 >>>(mode_num);
		printf("\n");
	}

	return temp_mesh;
}

__global__ void gpu_print_mesh(int mode){
	if(mode == 1){
		for(int y = 0; y < Ny_d; ++y){
			for(int x = 0; x < Nx_d; ++x){
				printf("%d ", solid_d[Nx_d*y + x]);
			}
		printf("\n");
		}
	}
}

__host__ void initialization(double *array, double value){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_initialization<<< grid, block >>>(array, value);
	getLastCudaError("gpu_print_array kernel error");
}

__global__ void gpu_initialization(double *array, double value){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	array[gpu_scalar_index(x, y)] = value;
}

__host__ bool* create_pinned_mesh(bool *array){

	bool *pinned;
	const unsigned int bytes = Nx*Ny*sizeof(bool);

	checkCudaErrors(cudaMallocHost((void**)&pinned, bytes));
	memcpy(pinned, array, bytes);
	return pinned;
}

__host__ double* create_pinned_double(){

	double *pinned;

	checkCudaErrors(cudaMallocHost((void**)&pinned, mem_size_scalar));
	return pinned;
}
