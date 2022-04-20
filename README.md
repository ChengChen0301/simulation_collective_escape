# simulation_collective_escape
This repository contains codes for simulating crowd dynamics with the social force model and statistcal analysis of simualtion data, which is the source code of the publication https://pubs.rsc.org/en/content/articlelanding/2021/sm/d1sm00033k

Process 1: produce the simulation data
The codes for this part are put in the folder "SimulationProgram". 
Makefile and run the executable with ./test $1, you will get the simulation data in "samples/round1".
The data contains£º
(1) the positions of all particles in "position.txt";
(2) the escape time of each particle in "evacuation_time.txt";
(3) the successful escape at any time in "outside_point.txt".
Note: You may change the value of 'a' in the 'main.cpp' to get different magnitudes of noises. Especially, a=0 indicates the case without noises.


Process 2: analyse the simulation data
Since this process is implemented by python, we first convert the data from '.txt' format to '.npz' format, which could largely improve the loading speed. 
This is achieved by the function convert_data() in 'compute.py'.
Then with all the data converted to '.npz' files, run the functions in 'compute.py' to analyze the simulation data, and the results are stored in the folder "computeResults" in the '.npy' format.


Process 3: perform Delaunay triangulation
The codes for this part are put in the folder "triangulation".
The program deals with '.txt' file which contains the positions of particles at certain time, thus we need to store the simulation data separately (see samples/round1/position).
The program produces two kind of files: '_elements.txt' contains three index of a triangle; '_neighbor.txt' contains the number of neighbors for each positioned particle.
Note: For particles on the boundary, the number of neighbors in the "_neighbor.txt" file is false. It is one less than the actual. The function fix_boundary_neighbor() can help fix the error.


Process 4: visualize the analysis results
With the analysis results ready in Process 2, run the functions in 'show.py' to generate the pictures in the paper. All the pictures can be reproduced and stored in the folder "figures".


2021/5/24
