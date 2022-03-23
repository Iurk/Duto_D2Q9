#ifndef __DADOS_H
#define __DADOS_H

#include <vector>

namespace myGlobals{

	//Domain
	extern unsigned int Nx, Ny;

	//Simulation
	extern unsigned int NSTEPS, NSAVE, NMSG;
	extern bool meshprint;
	extern double erro_max;

	//GPU
	extern unsigned int nThreads;

	//Input
	extern double u_max_lattice, rho0, Re;
	extern double nu;
	extern const double tau;

	//Boundary
	extern bool periodic;
	extern double gx, gy, rhoin, rhoout;
	extern std::string inlet_bc, outlet_bc;

	//Air
	extern const double mi_ar;

	//Lattice Info
	extern unsigned int ndir;
	extern int *ex;
	extern int *ey;
	extern double cs, w0, wp, ws;

	//Memory Sizes
	extern const size_t mem_mesh;
	extern const size_t mem_size_ndir;
	extern const size_t mem_size_scalar;

	// Deltas
	//extern double delx, dely, delt;

	extern bool *walls, *inlet, *outlet;
}

#endif
