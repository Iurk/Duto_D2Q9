#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>

#include <cuda.h>
#include <errno.h>

#include "LBM.h"
#include "dados.h"
//#include "boundary.h"

using namespace myGlobals;

// Input data
__constant__ unsigned int Nx_d, Ny_d;
__constant__ double D_d, delx_d, dely_d, delt_d;
__constant__ double rho0_d, u_max_d, nu_d, mi_ar_d, umax_d;

// LBM Data
__constant__ double tau_d, gx_d, gy_d;

//Lattice Data
__constant__ unsigned int q;
__constant__ double cs_d, w0_d, wp_d, ws_d;
__device__ int *ex_d, *ey_d;

// Mesh data
__device__ bool *walls_d, *inlet_d, *outlet_d;

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

__global__ void gpu_init_equilibrium(double*, double*, double*, double*);
__global__ void gpu_stream_collide_save(double*, double*, double*, double*, double*, double*, double*, double*, bool);
__global__ void gpu_compute_convergence(double*, double*, double*);
__global__ void gpu_compute_flow_properties(unsigned int, double*, double*, double*, double*);
__global__ void gpu_print_mesh(int);
__global__ void gpu_initialization(double*, double);
__global__ void gpu_compute_diff_u(double*, double*);

// Equilibrium
__device__ void gpu_equilibrium(unsigned int x, unsigned int y, double rho, double ux, double uy, double *feq){

	double cs2 = cs_d*cs_d;
	double cs4 = cs2*cs2;
	double cs6 = cs4*cs2;

	double A = 1.0/(cs2);
	double B = 1.0/(2.0*cs4);
	double C = 1.0/(2.0*cs6);

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d};

	for(int n = 0; n < q; ++n){

		double ux2 = ux*ux;
		double uy2 = uy*uy;
		double ex2 = ex_d[n]*ex_d[n];
		double ey2 = ey_d[n]*ey_d[n];

		double order_1 = A*(ux*ex_d[n] + uy*ey_d[n]);
		double order_2 = B*(ux2*(ex2 - cs2) + 2*ux*uy*ex_d[n]*ey_d[n] + uy2*(ey2 - cs2));

		feq[gpu_fieldn_index(x, y, n)] = W[n]*rho*(1 + order_1 + order_2);
	}
}

__device__ void gpu_non_equilibrium(unsigned int x, unsigned int y, double tauxx, double tauxy, double tauyy, double *fneq){

	double cs2 = cs_d*cs_d;
	double cs4 = cs2*cs2;

	double B = 1.0/(2.0*cs4);

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d};

	for(int n = 0; n < q; ++n){
		double ex2 = ex_d[n]*ex_d[n];
		double ey2 = ey_d[n]*ey_d[n];

		double order_2 = B*(tauxx*(ex2 - cs2) + 2*tauxy*ex_d[n]*ey_d[n] + tauyy*(ey2 - cs2));

		fneq[gpu_fieldn_index(x, y, n)] = W[n]*(order_2);
	}
}

__device__ void gpu_source(unsigned int x, unsigned int y, double rho, double ux, double uy, double *S){

	double cs2 = cs_d*cs_d;

	double A = 1.0/cs2;
	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d};

	for(int n = 0; n < q; ++n){
		double gdotei = gx_d*ex_d[n] + gy_d*ey_d[n];
		double udotei = ux*ex_d[n] + uy*ey_d[n];

		double order_1 = gx_d*(ex_d[n] - ux) + gy_d*(ey_d[n] - uy);
		double order_2 = A*gdotei*udotei;

		S[gpu_fieldn_index(x, y, n)] = A*W[n]*rho*(order_1 + order_2);
	}
}

// Poiseulle Flow
__device__ double poiseulle_eval(unsigned int x, unsigned int y){

	double rA = y*dely_d;
	double R = D_d/2.0;
	double rl = rA - R;

	double ux_si = umax_d*(1 - (rl/R)*(rl/R));
	double ux_lattice = ux_si*(delt_d/delx_d);

	return ux_lattice;
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

	gpu_equilibrium(x, y, rho, ux, uy, f1);
}

__host__ void stream_collide_save(double *f1, double *f2, double *feq, double *fneq, double *S, double *r, double *u, double *v, bool save){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	//dim3 grid(1,1,1);
	//dim3 block(1,1,1);

	gpu_stream_collide_save<<< grid, block >>>(f1, f2, feq, fneq, S, r, u, v, save);
	getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_stream_collide_save(double *f1, double *f2, double *feq, double *fneq, double *S, double *r, double *u, double *v, bool save){

	const double omega = 1.0/tau_d;

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int x_att, y_att;

	double rho = 0, ux_i = 0, uy_i = 0;
	for(int n = 0; n < q; ++n){
		x_att = (x - ex_d[n] + Nx_d)%Nx_d;
		y_att = (y - ey_d[n] + Ny_d)%Ny_d;

		rho += f1[gpu_fieldn_index(x_att, y_att, n)];
		ux_i += f1[gpu_fieldn_index(x_att, y_att, n)]*ex_d[n];
		uy_i += f1[gpu_fieldn_index(x_att, y_att, n)]*ey_d[n];
	}

	double ux = (ux_i + 0.5*rho*gx_d)/rho;
	double uy = (uy_i + 0.5*rho*gy_d)/rho;

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;

	gpu_source(x, y, rho, ux, uy, S);
	gpu_equilibrium(x, y, rho, ux, uy, feq);
	
	// Approximation of fneq
	for(int n = 0; n < q; ++n){
		x_att = (x - ex_d[n] + Nx_d)%Nx_d;
		y_att = (y - ey_d[n] + Ny_d)%Ny_d;
		fneq[gpu_fieldn_index(x, y, n)] = f1[gpu_fieldn_index(x_att, y_att, n)] - feq[gpu_fieldn_index(x, y, n)];
	}

	// Calculating the Viscous stress tensor
	double tauxx = 0, tauxy = 0, tauyy = 0;
	for(int n = 0; n < q; ++n){
		tauxx += fneq[gpu_fieldn_index(x, y, n)]*ex_d[n]*ex_d[n];
		tauxy += fneq[gpu_fieldn_index(x, y, n)]*ex_d[n]*ey_d[n];
		tauyy += fneq[gpu_fieldn_index(x, y, n)]*ey_d[n]*ey_d[n];
	}

	gpu_non_equilibrium(x, y, tauxx, tauxy, tauyy, fneq);

	// Collision and Stream Step
	for(int n = 0; n < q; ++n){
		
		f2[gpu_fieldn_index(x_att, y_att, n)] = omega*feq[gpu_fieldn_index(x, y, n)] + (1 - omega)*f1[gpu_fieldn_index(x, y, n)] + (1 - 0.5*omega)*S[gpu_fieldn_index(x, y, n)];
		//f2[gpu_fieldn_index(x_att, y_att, n)] = feq[gpu_fieldn_index(x, y, n)] + (1 - omega)*fneq[gpu_fieldn_index(x, y, n)] + (1 - 0.5*omega)*S[gpu_fieldn_index(x, y, n)];
	}
	

	bool node_solid = walls_d[gpu_scalar_index(x, y)];
	// Applying Boundary Conditions
	if(node_solid){
		gpu_bounce_back(x, y, f2);
	}
}

__host__ double report_convergence(unsigned int t, double *u, double *u_old, double *conv_host, double *conv_gpu, bool msg){

	double conv;
	conv = compute_convergence(u, u_old, conv_host, conv_gpu);

	if(msg){
		std::cout << std::setw(10) << t << std::setw(20) << conv << std::endl;
	}

	return conv;
}

__host__ double compute_convergence(double *u, double *u_old, double *conv_host, double *conv_gpu){

	dim3 grid(1, Ny/nThreads, 1);
	dim3 block(1, nThreads, 1);

	gpu_compute_convergence<<< grid, block, 2*block.y*sizeof(double) >>>(u, u_old, conv_gpu);
	getLastCudaError("gpu_compute_convergence kernel error");

	size_t conv_size_bytes = 2*grid.x*grid.y*sizeof(double);
	checkCudaErrors(cudaMemcpy(conv_host, conv_gpu, conv_size_bytes, cudaMemcpyDeviceToHost));

	double convergence;
	double sumuxe2 = 0.0;
	double sumuxa2 = 0.0;

	for(unsigned int i = 0; i < grid.x*grid.y; ++i){

		sumuxe2 += conv_host[2*i];
		sumuxa2 += conv_host[2*i+1];
	}

	convergence = sqrt(sumuxe2/sumuxa2);
	return convergence;
}

__global__ void gpu_compute_convergence(double *u, double *u_old, double *conv){

	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int x = (Nx_d-1)/4;

	extern __shared__ double data[];

	double *uxe2 = data;
	double *uxa2 = data + 1*blockDim.y;

	double ux = u[gpu_scalar_index(x, y)];
	double ux_old = u_old[gpu_scalar_index(x, y)];

	uxe2[threadIdx.y] = (ux - ux_old)*(ux - ux_old);
	uxa2[threadIdx.y] = ux_old*ux_old;

	__syncthreads();

	if(threadIdx.y == 0){

		size_t idx = 2*(gridDim.x*blockIdx.y + blockIdx.x);

		for(int n = 0; n < 2; ++n){
			conv[idx+n] = 0.0;
		}

		for(int i = 0; i < blockDim.x; ++i){
			conv[idx  ] += uxe2[i];
			conv[idx+1] += uxa2[i];
		}
	}
}

__host__ std::vector<double> report_flow_properties(unsigned int t, double conv, double *rho, double *ux, double *uy, double *prop_gpu, double *prop_host, bool msg){

	std::vector<double> prop;

	if(msg){
		prop = compute_flow_properties(t, rho, ux, uy, prop, prop_gpu, prop_host);

		std::cout << std::setw(10) << "Timestep" << std::setw(10) << "E" << std::setw(18)  << "Convergence" << std::endl;
		std::cout << std::setw(10) << t << std::setw(13) << prop[0] << std::setw(15) << conv << std::endl;
		std::cout << "Norms" << std::endl;
		std::cout << std::setw(10) << "L2" << std::setw(15) << "P1" << std::setw(13) << "Pinf" << std::endl;
		std::cout << std::setw(13) << prop[1] << std::setw(15) << prop[2] << std::setw(13) << prop[3] << std::endl;
	}

	return prop;
}

__host__ std::vector<double> compute_flow_properties(unsigned int t, double *r, double *u, double *v, std::vector<double> prop, double *prop_gpu, double *prop_host){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_compute_flow_properties<<< grid, block, 5*block.x*sizeof(double) >>>(t, r, u, v, prop_gpu);
	getLastCudaError("gpu_compute_flow_properties kernel error");

	size_t prop_size_bytes = 5*grid.x*grid.y*sizeof(double);
	checkCudaErrors(cudaMemcpy(prop_host, prop_gpu, prop_size_bytes, cudaMemcpyDeviceToHost));

	double E = 0.0, sumuxe1 = 0.0, sumuxa1 = 0.0, sumuxe2 = 0.0, sumuxa2 = 0.0;
	for(unsigned int i = 0; i < grid.x*grid.y; ++i){

		E += prop_host[5*i];
		sumuxe1 += prop_host[5*i+1];
		sumuxa1 += prop_host[5*i+2];
		sumuxe2  += prop_host[5*i+3];
		sumuxa2  += prop_host[5*i+4];
	}

	double P1 = sumuxe1/sumuxa1;
	double L2 = sqrt(sumuxe2/sumuxa2);

	double *diff_uh, *diff_ud;
	diff_uh = create_pinned_double();
	checkCudaErrors(cudaMalloc((void**)&diff_ud, mem_size_scalar));

	gpu_compute_diff_u<<< grid, block >>>(u, diff_ud);
	getLastCudaError("gpu_compute_diff_u kernel error");

	checkCudaErrors(cudaMemcpy(diff_uh, diff_ud, mem_size_scalar, cudaMemcpyDeviceToHost));

	double Pinf = 0.0;
	for(int y = 0; y < Ny; ++y){
		for(int x = 0; x < Nx; ++x){
			if(abs(diff_uh[Nx*y+x]) > Pinf){
				Pinf = abs(diff_uh[Nx*y+x]);
			}
		}
	}

	prop.push_back(E);
	prop.push_back(L2);
	prop.push_back(P1);
	prop.push_back(Pinf);

	return prop;
}

__global__ void gpu_compute_diff_u(double *u, double *diff){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	double ux = u[gpu_scalar_index(x, y)];
	double uxa = poiseulle_eval(x, y);

	if(y == 0 || y == Ny_d-1){
		diff[gpu_scalar_index(x, y)] = 0.0;
	}
	else{
		diff[gpu_scalar_index(x, y)] = (ux - uxa)/uxa;
	}
}

__global__ void gpu_compute_flow_properties(unsigned int t, double *r, double *u, double *v, double *prop_gpu){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	extern __shared__ double data[];

	double *E = data;
	double *uxe1 = data + 1*blockDim.x;
	double *uxa1  = data + 2*blockDim.x;
	double *uxe2  = data + 3*blockDim.x;
	double *uxa2  = data + 4*blockDim.x;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];

	E[threadIdx.x] = rho*(ux*ux + uy*uy);

	// Compute analytical results
    double uxa = poiseulle_eval(x, y);
    
    // Compute terms for P1 error norm
    uxe1[threadIdx.x]  = abs((ux - uxa));
    uxa1[threadIdx.x]  = uxa;

    // Compute terms for L2 error norm
    uxe2[threadIdx.x]  = (ux - uxa)*(ux - uxa);
    uxa2[threadIdx.x]  = uxa*uxa;

	__syncthreads();

	if (threadIdx.x == 0){
		
		size_t idx = 5*(gridDim.x*blockIdx.y + blockIdx.x);

		for(int n = 0; n < 5; ++n){
			prop_gpu[idx+n] = 0.0;
		}

		for(int i = 0; i < blockDim.x; ++i){
			prop_gpu[idx  ] += E[i];
            prop_gpu[idx+1] += uxe1[i];
            prop_gpu[idx+2] += uxa1[i];
            prop_gpu[idx+3] += uxe2[i];
            prop_gpu[idx+4] += uxa2[i];
		}
	}
}

__host__ void wrapper_input(unsigned int *nx, unsigned int *ny, double *rho, double *u, double *nu, const double *mi_ar){
	checkCudaErrors(cudaMemcpyToSymbol(Nx_d, nx, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(Ny_d, ny, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(rho0_d, rho, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(u_max_d, u, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(nu_d, nu, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(mi_ar_d, mi_ar, sizeof(double)));
}

__host__ void wrapper_analytical(double *d, double *delx, double *dely, double *delt, double *umax){
	checkCudaErrors(cudaMemcpyToSymbol(D_d, d, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(delx_d, delx, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(dely_d, dely, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(delt_d, delt, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(umax_d, umax, sizeof(double)));
}

__host__ void wrapper_LBM(double *gx, double *gy, const double *tau){
	checkCudaErrors(cudaMemcpyToSymbol(gx_d, gx, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(gy_d, gy, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(tau_d, tau, sizeof(double)));
}

__host__ void wrapper_lattice(unsigned int *ndir, double *cs, double *w_0, double *w_p, double *w_s){
	checkCudaErrors(cudaMemcpyToSymbol(q, ndir, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(cs_d, cs, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(w0_d, w_0, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(wp_d, w_p, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(ws_d, w_s, sizeof(double)));
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
	

	if(mode == "walls"){
		checkCudaErrors(cudaMemcpyToSymbol(walls_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 1;
	}

	else if(mode == "inlet"){
		checkCudaErrors(cudaMemcpyToSymbol(inlet_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 2;
	}

	else if(mode == "outlet"){
		checkCudaErrors(cudaMemcpyToSymbol(outlet_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 3;
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
				printf("%d ", walls_d[Nx_d*y + x]);
			}
		printf("\n");
		}
	}

	else if(mode == 2){
		for(int y = 0; y < Ny_d; ++y){
			for(int x = 0; x < Nx_d; ++x){
				printf("%d ", inlet_d[Nx_d*y + x]);
			}
		printf("\n");
		}
	}

	else if(mode == 3){
		for(int y = 0; y < Ny_d; ++y){
			for(int x = 0; x < Nx_d; ++x){
				printf("%d ", outlet_d[Nx_d*y + x]);
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
