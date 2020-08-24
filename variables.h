#ifndef VARIABLES_H
#define VARIABLES_H

#include <list>
#include <string>

#include "Point.h"

using namespace std;

// Constants & Input
#define Np  (1000)      // number of particles
#define Nt  (3000)     // given number of records
#define Nw  (5)        // number of walls, 5 for a square room with a door
#define _r   (0.3)      // radius of the particle
#define v   (1.0)      // desired speed
#define m   (80.0)     // mass of a particle
#define paraA   (2000.0)
#define paraB   (0.08)
#define k_bump   (120000.0) // bump effect
#define K_friction   (240000.0) // friction effect
#define tau (0.5)      // acceleration time
#define delta (0.001)   // record interval
#define Ng_x (70) // number of grids per column, more than actual size
#define Ng_y (50) // number of grids per row
#define pi 3.1415926

extern Point pt[Np];
extern Line wall[Nw];
extern vec door;
extern double a;       // noise ratio
extern double len;     // side length of grids
extern int number;     // number of points in the room
extern list<int> grids[Ng_x][Ng_y]; // to store the labels of points in each grid
extern int ET[Np];     // evacuation time of each point
extern double save_position[Np][2][2000]; // to save the positions
extern int save_flag[2000][Np]; // to save the in/out state
extern int Out_number[2000]; // to save the number of out

extern double _left;   // "left" conflicts with the namespace
extern double _right;
extern double _up;
extern double _down;
extern double width;   // half the width of the door
extern double theta;   // half the angle between the wall

#endif // VARIABLES_H
