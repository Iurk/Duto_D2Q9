#include <iostream>
#include "yaml-cpp/yaml.h"

#include "utilidades.h"
#include "dados.h"
#include "paths.h"

// Opening yaml file
YAML::Node config = YAML::LoadFile("./bin/dados.yml");
YAML::Node config_lattice = YAML::LoadFile("./bin/lattices.yml");

// Getting sections
const YAML::Node& domain = config["domain"];
const YAML::Node& simulation = config["simulation"];
const YAML::Node& gpu = config["gpu"];
const YAML::Node& input = config["input"];
const YAML::Node& boundary = config["boundary"];
const YAML::Node& air = config["air"];

std::string Lattice = simulation["lattice"].as<std::string>();
const YAML::Node& lattice = config_lattice[Lattice];

namespace myGlobals{

	//Domain
	double H = domain["H"].as<double>();
	double D = domain["D"].as<double>();
	unsigned int Nx = domain["Nx"].as<unsigned int>();
	unsigned int Ny = domain["Ny"].as<unsigned int>();

	//Simulation
	unsigned int NSTEPS = simulation["NSTEPS"].as<unsigned int>();
	unsigned int NSAVE = simulation["NSAVE"].as<unsigned int>();
	unsigned int NMSG = simulation["NMSG"].as<unsigned int>();
	bool meshprint = simulation["meshprint"].as<bool>();
	double erro_max = simulation["erro_max"].as<double>();

	//GPU
	unsigned int nThreads = gpu["nThreads"].as<unsigned int>();

	//Input
	double u_max_si = input["u_max_si"].as<double>();
	double rho0 = input["rho0"].as<double>();
	double Re = input["Re"].as<double>();

	//Boundary
	bool periodic = boundary["periodic"].as<bool>();
	double gx = boundary["gx"].as<double>();
	double gy = boundary["gy"].as<double>();
	std::string inlet_bc = boundary["inlet"].as<std::string>();
	std::string outlet_bc = boundary["outlet"].as<std::string>();
	double rhoin = boundary["rhoin"].as<double>();
	double rhoout = boundary["rhoout"].as<double>();
	
	//Air
	const double mi_ar = air["mi"].as<double>();

	//Lattice Info
	unsigned int ndir = lattice["q"].as<unsigned int>();
	std::vector<int> ex_vec = lattice["ex"].as<std::vector<int>>();
	std::vector<int> ey_vec = lattice["ey"].as<std::vector<int>>();
	std::string cs_str = lattice["cs"].as<std::string>();
	std::string w0_str = lattice["w0"].as<std::string>();
	std::string wp_str = lattice["wp"].as<std::string>();
	std::string ws_str = lattice["ws"].as<std::string>();

	int *ex = ex_vec.data();
	int *ey = ey_vec.data();
	double cs = equation_parser(cs_str);
	double w0 = equation_parser(w0_str);
	double wp = equation_parser(wp_str);
	double ws = equation_parser(ws_str);

	//Memory Sizes
	const size_t mem_mesh = sizeof(bool)*Nx*Ny;
	const size_t mem_size_ndir = sizeof(double)*Nx*Ny*ndir;
	const size_t mem_size_scalar = sizeof(double)*Nx*Ny;

	// Deltas
	double delx = H/(Nx-1);
	double dely = D/(Ny-1);
	double delt = delx/16;

	// Nu and Tau and Conversions
	double nu_si = (u_max_si*D)/Re;

	double u_max = u_max_si*delt/delx;
	double nu = nu_si*delt/(delx*delx);
	
	//double nu = 0.1/4;
	//double u_max = (nu*Re)/Ny;
	const double tau = nu*(cs*cs) + 0.5;

	bool *walls = read_bin(walls_mesh);
	bool *inlet = read_bin(inlet_mesh);
	bool *outlet = read_bin(outlet_mesh);

}
