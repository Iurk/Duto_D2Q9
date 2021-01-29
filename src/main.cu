#include <stdio.h>
#include <stdlib.h>

#include "seconds.h"
#include "LBM.h"
#include "dados.h"

using namespace myGlobals;

int main(int argc, char const *argv[]){

	// Writing Simulation Parameters
	printf("Simulating the Poiseulle Flow\n");
	printf("  Domain size: %ux%u\n", Nx, Ny);
	printf("           nu: %g\n", nu);
	printf("          tau: %g\n", tau);
	printf("        u_max: %g\n", u_max);
	printf("         rho0: %g\n", rho0);
	printf("           Re: %g\n", Re);
	printf("  Times Stpes: %u\n", NSTEPS);
	printf("   Save every: %u\n", NSAVE);
	printf("Message every: %u\n", NMSG);

	double bytesPerMiB = 1024.0*1024.0;
	double bytesPerGiB = 1024.0*1024.0*1024.0;

	// Getting Device Info and Writing then
	checkCudaErrors(cudaSetDevice(0));
	int deviceId = 0;
	checkCudaErrors(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));

	size_t gpu_free_mem, gpu_total_mem;
	checkCudaErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));

	printf("CUDA information\n");
	printf("      Using device: %d\n", deviceId);
	printf("              Name: %s\n", deviceProp.name);
	printf("   Multiprocessors: %d\n", deviceProp.multiProcessorCount);
	printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("     Global Memory: %.1f MiB\n", deviceProp.totalGlobalMem/bytesPerMiB);
	printf("       Free Memory: %.1f MiB\n", gpu_free_mem/bytesPerMiB);
	printf("\n");

	// Declaration and Allocation in device Memory
	double *f1_gpu, *f2_gpu, *f1neq_gpu;
	double *rho_gpu, *ux_gpu, *uy_gpu;
	double *prop_gpu;

	checkCudaErrors(cudaMalloc((void**)&f1_gpu, mem_size_ndir));
	checkCudaErrors(cudaMalloc((void**)&f2_gpu, mem_size_ndir));
	checkCudaErrors(cudaMalloc((void**)&f1neq_gpu, mem_size_ndir));
	checkCudaErrors(cudaMalloc((void**)&rho_gpu, mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&ux_gpu, mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&uy_gpu, mem_size_scalar));

	const size_t mem_size_props = 3*Nx/nThreads*Ny*sizeof(double);
	checkCudaErrors(cudaMalloc((void**)&prop_gpu, mem_size_props));

	double *scalar_host;
	scalar_host = create_pinned_double();
	if(scalar_host == NULL){
		fprintf(stderr, "Error: unable to allocate required memory (%.1f MiB).\n", mem_size_scalar/bytesPerMiB);
		exit(-1);
	}

	size_t total_mem_bytes = 3*mem_size_ndir + 3*mem_size_scalar + mem_size_props;
	
	// Creating Events for time measure
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// Allocation of Input data in Device constant memory
	wrapper_input(&Nx, &Ny, &rho0, &u_max, &nu, &tau, &mi_ar);

	// Allocation of Lattice data in Device constant and global memory
	wrapper_lattice(&ndir, &cs, &w0, &ws, &wd);

	int *ex_gpu, *ey_gpu;

	ex_gpu = generate_e(ex, "x");
	ey_gpu = generate_e(ey, "y");

	bool *solid_p;
	bool *solid_gpu;

	solid_p = create_pinned_mesh(solid);

	// Generating Mesh
	solid_gpu = generate_mesh(solid_p, "solid");

	// Initialization
	initialization(rho_gpu, rho0);
	initialization(ux_gpu, u_max);
	initialization(uy_gpu, 0.0);

	init_equilibrium(f1_gpu, rho_gpu, ux_gpu, uy_gpu);
	checkCudaErrors(cudaMemset(f1neq_gpu, 0, mem_size_ndir));

	save_scalar("rho",rho_gpu, scalar_host, 0);
	save_scalar("ux", ux_gpu, scalar_host, 0);
	save_scalar("uy", uy_gpu, scalar_host, 0);
	
	// Simulation Start
	double begin = seconds();
	checkCudaErrors(cudaEventRecord(start, 0));

	std::vector<double> error_prop;

	// Main Loop
	for(unsigned int n = 0; n < NSTEPS; ++n){
		bool save = (n+1)%NSAVE == 0;
		bool msg = (n+1)%NMSG == 0;
		bool need_scalars = save || (msg && computeFlowProperties);

		double *ux_test;

		ux_test = (double*)malloc(mem_size_scalar);
		checkCudaErrors(cudaMemcpy(ux_test, ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

		for(int y = 0; y < Ny; ++y){
			std::cout << y << "-> ";
			for(int x = 0; x < Nx; ++x){
				std::cout << ux_test[Nx*y+x] << " ";
			}
			std::cout << std::endl;
		}

		stream_collide_save(f1_gpu, f2_gpu, f1neq_gpu, rho_gpu, ux_gpu, uy_gpu, need_scalars);

		if(save){
			save_scalar("rho",rho_gpu, scalar_host, n+1);
			save_scalar("ux", ux_gpu, scalar_host, n+1);
			save_scalar("uy", uy_gpu, scalar_host, n+1);
		}

		double *temp = f1_gpu;
		f1_gpu = f2_gpu;
		f2_gpu = temp;

		error_prop = report_flow_properties(n+1, rho_gpu, ux_gpu, uy_gpu, prop_gpu, scalar_host, msg, computeFlowProperties);

		if(error_prop[1] < erro_max){
			printf("%u, %g, %g\n", n+1, error_prop[0], error_prop[1]);
			break;
		}
	}
	
	// Measuring time
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float miliseconds = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&miliseconds, start, stop));

	double end = seconds();
	double runtime = end - begin;
	double gpu_runtime = 0.001*miliseconds;

	size_t doubles_read = ndir;
	size_t doubles_wirtten = ndir;
	size_t doubles_saved = 3;

	size_t nodes_updated = NSTEPS*size_t(Nx*Ny);
	size_t nodes_saved = (NSTEPS/NSAVE)*size_t(Nx*Ny);
	double speed = nodes_updated/(1e6*runtime);

	double bandwidth = (nodes_updated*(doubles_read + doubles_wirtten) + nodes_saved*(doubles_saved))*sizeof(double)/(runtime*bytesPerGiB);

	// Writing the performance
	printf("Performance Information\n");
	printf(" Memory Allocated (GPU): %.1f (MiB)\n", total_mem_bytes/bytesPerMiB);
	printf("Memory Allocated (host): %.1f (MiB)\n", mem_size_scalar/bytesPerMiB);
	printf("              Timesteps: %u\n", NSTEPS);
	printf("             Clock Time: %.3f (s)\n", runtime);
	printf("            GPU runtime: %.3f (s)\n", gpu_runtime);
	printf("                  Speed: %.2f (Mlups)\n", speed);
	printf("               Bandwith: %.1f (GiB/s)\n", bandwidth);

	// Cleaning up

	// Destroying Events
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	// Freeing Device and CPU Memory
	// LBM variables
	checkCudaErrors(cudaFree(f1_gpu));
	checkCudaErrors(cudaFree(f2_gpu));
	checkCudaErrors(cudaFree(f1neq_gpu));
	checkCudaErrors(cudaFree(rho_gpu));
	checkCudaErrors(cudaFree(ux_gpu));
	checkCudaErrors(cudaFree(uy_gpu));
	checkCudaErrors(cudaFree(prop_gpu));
	checkCudaErrors(cudaFree(ex_gpu));
	checkCudaErrors(cudaFree(ey_gpu));

	// Mesh arrays
	checkCudaErrors(cudaFree(solid_gpu));
	checkCudaErrors(cudaFreeHost(solid_p));

	// Host arrays
	checkCudaErrors(cudaFreeHost(scalar_host));

	cudaDeviceReset();

	return 0;
}
