#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <string>
#include <dirent.h>
#include <sys/stat.h>

#include "paths.h"
#include "LBM.h"
#include "dados.h"

using namespace myGlobals;

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
	
	fclose(fout);
}

__host__ void save_terminal(int time, double conv, std::vector<double> prop){

	std::ostringstream filename;

	std::string ext = ".dat";

	filename << bin_folder << "error_data" << ext;
	const char* filename_c = strdup(filename.str().c_str());

	std::ofstream fout;
	fout.open(filename.str());

	fout << std::setw(10) << "Timestep" << std::setw(10) << "E" << std::setw(15) << "L2" << std::setw(23) << "Convergence" << std::endl;
	fout << std::setw(10) << time << std::setw(13) << prop[0] << std::setw(15) << prop[1] << std::setw(20) << conv << std::endl;
	fout.close();
}