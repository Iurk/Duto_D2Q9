#include <iostream>
#include <fstream>
#include <string.h>

#include "tinyexpr.h"
#include "utilidades.h"

double equation_parser(const std::string equation){

	char expr[32];

	strcpy(expr, equation.c_str());

	double result = te_interp(expr, 0);

	return result;
}

bool* read_bin(const std::string mesh_path){

	std::ifstream file(mesh_path);

	file.seekg(0, file.end);
	int length = file.tellg();
	file.seekg(0, file.beg);

	char *buffer = new char[length];

	file.read(buffer, length);
	file.close();

	bool *mesh = (bool*)buffer;

	return mesh;
}