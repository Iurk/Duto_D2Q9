#include <math.h>

#include "dados.h"
#include "LBM.h"
#include "boundary.h"

#define real __cuminpack_real__

using namespace myGlobals;
__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

__global__ void gpu_bounce_back(double*);
__global__ void gpu_inlet(double*, double*, double*, double*);
__global__ void gpu_outlet(double*);

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

__host__ void bounce_back(double *f){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_bounce_back<<< grid, block >>>(f);
	getLastCudaError("gpu_bounce_back kernel error");
}

__global__ void gpu_bounce_back(double *f){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_wall = solid_d[gpu_scalar_index(x, y)];
	if(node_wall){
		device_bounce_back(x, y, f);
	}
}

__device__ void device_inlet(unsigned int x, unsigned int y, double *f, double *r, double *u, double *v){

	double ux = u_max_d;
	double uy = 0.0;

	unsigned int idx_0 = gpu_fieldn_index(x, y, 0);
	unsigned int idx_2 = gpu_fieldn_index(x, y, 2);
	unsigned int idx_3 = gpu_fieldn_index(x, y, 3);
	unsigned int idx_4 = gpu_fieldn_index(x, y, 4);
	unsigned int idx_6 = gpu_fieldn_index(x, y, 6);
	unsigned int idx_7 = gpu_fieldn_index(x, y, 7);

	double rho = (f[idx_0] + f[idx_2] + f[idx_4] + 2*(f[idx_3] + f[idx_6] + f[idx_7]))/(1.0 - ux);

	f[gpu_fieldn_index(x, y, 1)] = f[idx_3] + (2.0/3.0)*rho*ux;
	f[gpu_fieldn_index(x, y, 5)] = f[idx_7] - 0.5*(f[idx_2] - f[idx_4]) + (1.0/6.0)*rho*ux;
	f[gpu_fieldn_index(x, y, 8)] = f[idx_6] + 0.5*(f[idx_2] - f[idx_4]) + (1.0/6.0)*rho*ux;

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;

}

__host__ void inlet_BC(double *f, double *r, double *u, double *v){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_inlet<<< grid, block >>>(f, r, u, v);
	getLastCudaError("gpu_inlet kernel error");

}

__global__ void gpu_inlet(double *f, double *r, double *u, double *v){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x == 0){
		device_inlet(x, y, f, r, u, v);
	}

}

__host__ void outlet_BC(double *f){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_outlet<<< grid, block >>>(f);
	getLastCudaError("gpu_outlet kernel error");
}

__global__ void gpu_outlet(double *f){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x == Nx_d-1){
		for(int n = 0; n < q; ++n){
			f[gpu_fieldn_index(x, y, n)] = f[gpu_fieldn_index(x-1, y, n)];
		}
	}
}