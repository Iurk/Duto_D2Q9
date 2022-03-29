#include "LBM.h"
#include "dados.h"
#include "boundary.h"

using namespace myGlobals;

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

__global__ void gpu_bounce_back(double*);
__global__ void gpu_inlet(double, double, double*, double*, double*, double*, unsigned int);
__global__ void gpu_outlet(double, double, double*, double*, double*, double*, unsigned int);

// Boundary Conditions
__device__ void device_bounce_back(unsigned int x, unsigned int y, double *f){
	
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

__device__ void device_inlet_VP(unsigned int x, unsigned int y, double u_in, double *f, double *r, double *u, double *v){

	double ux = u_in;
	double uy = 0.0;

	unsigned int idx_0 = gpu_fieldn_index(x, y, 0);
	unsigned int idx_2 = gpu_fieldn_index(x, y, 2);
	unsigned int idx_3 = gpu_fieldn_index(x, y, 3);
	unsigned int idx_4 = gpu_fieldn_index(x, y, 4);
	unsigned int idx_6 = gpu_fieldn_index(x, y, 6);
	unsigned int idx_7 = gpu_fieldn_index(x, y, 7);

	double rho = (f[idx_0] + f[idx_2] + f[idx_4] + 2*(f[idx_3] + f[idx_6] + f[idx_7]))/(1.0 - ux);

	f[gpu_fieldn_index(x, y, 1)] = (3*f[idx_3] + ux*(f[idx_3] + 2*(f[idx_0] + f[idx_2] + f[idx_4]) + 4*(f[idx_6] + f[idx_7])))/(3*(1.0 - ux));
	f[gpu_fieldn_index(x, y, 5)] = (6*f[idx_7] + 3*(f[idx_4] - f[idx_2]) + ux*(f[idx_0] + 2*(f[idx_3] + f[idx_6] - f[idx_4]) + 4*(f[idx_2] - f[idx_7])))/(6*(1.0 - ux));
	f[gpu_fieldn_index(x, y, 8)] = (6*f[idx_6] + 3*(f[idx_2] - f[idx_4]) + ux*(f[idx_0] + 2*(f[idx_3] + f[idx_7] - f[idx_2]) + 4*(f[idx_4] - f[idx_6])))/(6*(1.0 - ux));

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;
}

__device__ void device_inlet_PP(unsigned int x, unsigned int y, double rho_in, double *f, double *r, double *u, double *v){

	double rho = rho_in;
	double uy = 0.0;

	unsigned int idx_0 = gpu_fieldn_index(x, y, 0);
	unsigned int idx_2 = gpu_fieldn_index(x, y, 2);
	unsigned int idx_3 = gpu_fieldn_index(x, y, 3);
	unsigned int idx_4 = gpu_fieldn_index(x, y, 4);
	unsigned int idx_6 = gpu_fieldn_index(x, y, 6);
	unsigned int idx_7 = gpu_fieldn_index(x, y, 7);

	double ux = (rho - f[idx_0] - f[idx_2] - f[idx_4] - 2*(f[idx_3] + f[idx_6] + f[idx_7]))/(rho);

	f[gpu_fieldn_index(x, y, 1)] = (2*rho - f[idx_3] - 2*(f[idx_0] + f[idx_2] + f[idx_4]) - 4*(f[idx_6] + f[idx_7]))/(3);
	f[gpu_fieldn_index(x, y, 5)] = (rho - f[idx_0] + 2*(f[idx_4] - f[idx_3] - f[idx_6]) + 4*(f[idx_7] - f[idx_2]))/(6);
	f[gpu_fieldn_index(x, y, 8)] = (rho - f[idx_0] + 2*(f[idx_2] - f[idx_3] - f[idx_7]) + 4*(f[idx_6] - f[idx_4]))/(6);

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;
}

__device__ void device_outlet_FD(unsigned int x, unsigned int y, double *f){

	for(int n = 0; n < q; ++n){
		f[gpu_fieldn_index(x, y, n)] = f[gpu_fieldn_index(x-1, y, n)];
	}
}

__device__ void device_outlet_FDP(unsigned int x, unsigned int y, double rho_out, double *f){
	
	double sumRho = 0.0;
	for(int n = 0; n < q; ++n){
		sumRho += f[gpu_fieldn_index(x-1, y, n)];
	}

	for(int n = 0; n < q; ++n){
		f[gpu_fieldn_index(x, y, n)] = (rho_out/sumRho)*f[gpu_fieldn_index(x-1, y, n)];
	}
}

__device__ void device_outlet_VP(unsigned int x, unsigned int y, double u_out, double *f, double *r, double *u, double *v){
	
	double ux = u_out;
	double uy = 0.0;

	unsigned int idx_0 = gpu_fieldn_index(x, y, 0);
	unsigned int idx_1 = gpu_fieldn_index(x, y, 1);
	unsigned int idx_2 = gpu_fieldn_index(x, y, 2);
	unsigned int idx_4 = gpu_fieldn_index(x, y, 4);
	unsigned int idx_5 = gpu_fieldn_index(x, y, 5);
	unsigned int idx_8 = gpu_fieldn_index(x, y, 8);

	double rho = (f[idx_0] + f[idx_2] + f[idx_4] + 2*(f[idx_1] + f[idx_5] + f[idx_8]))/(1.0 + ux);

	f[gpu_fieldn_index(x, y, 3)] = (3*f[idx_1] + ux*(-f[idx_1] - 2*(f[idx_0] + f[idx_2] + f[idx_4]) - 4*(f[idx_5] + f[idx_8])))/(3*(1.0 + ux));
	f[gpu_fieldn_index(x, y, 6)] = (6*f[idx_8] + 3*(f[idx_4] - f[idx_2]) + ux*(-f[idx_0] - 2*(f[idx_1] - f[idx_4] + f[idx_5]) - 4*(f[idx_2] - f[idx_8])))/(6*(1.0 + ux));
	f[gpu_fieldn_index(x, y, 7)] = (6*f[idx_5] + 3*(f[idx_2] - f[idx_4]) + ux*(-f[idx_0] - 2*(f[idx_1] - f[idx_2] + f[idx_8]) - 4*(f[idx_4] - f[idx_5])))/(6*(1.0 + ux));

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;
}

__device__ void device_outlet_PP(unsigned int x, unsigned int y, double rho_out, double *f, double *r, double *u, double *v){
	
	double rho = rho_out;
	double uy = 0.0;

	unsigned int idx_0 = gpu_fieldn_index(x, y, 0);
	unsigned int idx_1 = gpu_fieldn_index(x, y, 1);
	unsigned int idx_2 = gpu_fieldn_index(x, y, 2);
	unsigned int idx_4 = gpu_fieldn_index(x, y, 4);
	unsigned int idx_5 = gpu_fieldn_index(x, y, 5);
	unsigned int idx_8 = gpu_fieldn_index(x, y, 8);

	double ux = (f[idx_0] + f[idx_2] + f[idx_4] + 2*(f[idx_1] + f[idx_5] + f[idx_8]) - rho)/(rho);

	f[gpu_fieldn_index(x, y, 3)] = (2*rho - f[idx_1] - 2*(f[idx_0] + f[idx_2] + f[idx_4]) - 4*(f[idx_5] + f[idx_8]))/(3);
	f[gpu_fieldn_index(x, y, 6)] = (rho - f[idx_0] - 2*(f[idx_1] - f[idx_4] + f[idx_5]) - 4*(f[idx_2] - f[idx_8]))/(6);
	f[gpu_fieldn_index(x, y, 7)] = (rho - f[idx_0] - 2*(f[idx_1] - f[idx_2] + f[idx_8]) - 4*(f[idx_4] - f[idx_5]))/(6);

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;
}

__host__ void bounce_back(double *f){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_bounce_back<<< grid, block >>>(f);
	getLastCudaError("gpu_bounce_back kernel error");
}

__global__ void gpu_bounce_back(double *f){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_walls = walls_d[gpu_scalar_index(x, y)];
	if(node_walls){
		device_bounce_back(x, y, f);
	}
}

__host__ void inlet_BC(double rho_in, double ux_in, double *f, double *r, double *u, double *v, std::string mode){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	unsigned int mode_num;
	if(mode == "VP"){
		mode_num = 1;
	}
	else if(mode == "PP"){
		mode_num = 2;
	}

	gpu_inlet<<< grid, block >>>(rho_in, ux_in, f, r, u, v, mode_num);
	getLastCudaError("gpu_inlet kernel error");
}

__global__ void gpu_inlet(double rho_in, double ux_in, double *f, double *r, double *u, double *v, unsigned int mode_num){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_inlet = inlet_d[gpu_scalar_index(x, y)];
	if(node_inlet){
		if(mode_num == 1){
			device_inlet_VP(x, y, ux_in, f, r, u, v);
		}
		else if(mode_num == 2){
			device_inlet_PP(x, y, rho_in, f, r, u, v);
		}
	}
}

__host__ void outlet_BC(double rho_out, double ux_out, double *f, double *r, double *u, double *v, std::string mode){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	unsigned int mode_num;
	if(mode == "FD"){
		mode_num = 1;
	}
	else if(mode == "FDP"){
		mode_num = 2;
	}
	else if(mode == "VP"){
		mode_num = 3;
	}
	else if(mode == "PP"){
		mode_num = 4;
	}

	gpu_outlet<<< grid, block >>>(rho_out, ux_out, f, r, u, v, mode_num);
	getLastCudaError("gpu_outlet kernel error");
}

__global__ void gpu_outlet(double rho_out, double ux_out, double *f, double *r, double *u, double *v, unsigned int mode_num){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_outlet = outlet_d[gpu_scalar_index(x, y)];
	if(node_outlet){
		if(mode_num == 1){
			device_outlet_FD(x, y, f);
		}
		else if(mode_num == 2){
			device_outlet_FDP(x, y, rho_out, f);
		}
		else if(mode_num == 3){
			device_outlet_VP(x, y, ux_out, f, r, u, v);
		}
		else if(mode_num == 4){
			device_outlet_PP(x, y, rho_out, f, r, u, v);
		}
	}
}