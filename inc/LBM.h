#ifndef __LBM_H
#define __LBM_H
#include <iostream>
#include <vector>

void initialization(double*, double);
void init_equilibrium(double*, double*, double*, double*);
void stream_collide_save(double*, double*, double*, double*, double*, double*, double*, double*, bool);
double compute_convergence(double*, double*, double*, double*);
std::vector<double> compute_flow_properties(unsigned int, double*, double*, double*, std::vector<double>, double*, double*);
void report_flow_properties(unsigned int, double, double*, double*, double*, double*, double*, bool, bool);
void save_scalar(const std::string , double*, double*, unsigned int);
void wrapper_input(unsigned int*, unsigned int*, double*, double*, double*, const double*, const double*);
void wrapper_lattice(unsigned int*, double*, double*, double*, double*);
int* generate_e(int*, std::string);
bool* generate_mesh(bool*, std::string);
bool* create_pinned_mesh(bool *);
double* create_pinned_double();

#define checkCudaErrors(err) __checkCudaErrors(err, #err, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line){
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA error at %s(%d)\"%s\": [%d] %s.\n", file, line, func, (int)err, cudaGetErrorString(err));
		exit(-1);
	}
}

inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n", file, line, (int)err, cudaGetErrorString(err));
		exit(-1);
	}
}

#endif
