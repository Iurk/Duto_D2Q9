#ifndef __BOUNDARY_H
#define __BOUNDARY_H

void bounce_back(double*);
void inlet_BC(double, double, double*, double*, double*, double*, std::string);
void outlet_BC(double, double, double*, double*, double*, double*, std::string);

#endif