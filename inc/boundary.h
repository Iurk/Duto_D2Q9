#ifndef __BOUNDARY_H
#define __BOUNDARY

void bounce_back(double *f);
void inlet_BC(double, double, double*, double*, double*, double*, double*, double*, std::string);
void outlet_BC(double, double, double*, double*, double*, double*, double*, double*, std::string);
#endif