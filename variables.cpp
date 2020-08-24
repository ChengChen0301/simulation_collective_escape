#include "variables.h"

Point pt[Np];
Line wall[Nw];
vec door;
double a;
double len = 2*_r + 5*paraB;
int number;
list<int> grids[Ng_x][Ng_y];
double save_position[Np][2][2000];
int save_flag[2000][Np];
int Out_number[2000];
int ET[Np];

double _left = -15.0;
double _right = 15.0;
double _up = 15.0;
double _down = -15.0;
double width = 1.0;
double theta = 90;


