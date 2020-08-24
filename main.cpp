#include <cstring>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
// #include "boost/format.hpp"
// #include "boost/program_options.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>

// #include "system.h"
#include "vec.h"
#include "Point.h"
#include "variables.h"
#include "Line.h"
#include "simulation.h"

// using boost::format;
// using namespace boost::program_options;
using namespace std;

int main(int argc, char *argv[]){

    
    // System S;
    // S.ReadInput(argc,argv);
    // int k;
    // char index[7]={0};
    // k = (int)S.gamma;
    
    char *k = argv[1];
    string dir = "new_sample";

    //-----------------------create the folders------------------------
    string folder = dir + "/round";
    folder += k;
    //sprintf(index,"%d",k);
    //folder += index;
    mkdir(folder.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    //CreateDirectory(folder.c_str(), NULL);

    string PosName = folder + "/output.txt";
    string ETName = folder + "/evacuation_time.txt";
    string OPName = folder + "/outside_point.txt";
    //-----------------------create the folders------------------------

    initialize();
    
    int t=0;
    if(t%100 == 0)
        savedata(t);

    while(number != 0){
        t++;
        update(t);
        if(t%100 == 0){ // record every 0.1 seconds
            savedata(t);
            // cout<<t<<"\n";
        }
    }

    ofstream fout;

    fout.open(ETName.c_str());
    for(int i=0; i<Np; i++)
        fout<<left<<setw(7)<<ET[i]<<"\n";
    fout<<left<<setw(7)<<t<<"\n"; // total evacuation time
    fout<<flush;fout.close();

//    fout.open(PosName.c_str());
    for(int j=0; j<=t/100; j++) {
        string index = to_string(j*100);
        string PosName = folder + "/output" + index + ".txt";
        fout.open(PosName.c_str());
        for (int i = 0; i < Np; i++) {
            fout << left << setw(13) << save_position[i][0][j] << '\t' << left << setw(13) << save_position[i][1][j] <<"\n";
                 // << '\t' << left << setw(3) << save_flag[j][i] << "\n";
        }
        fout<<flush;fout.close();
    }
//    fout<<flush;fout.close();

    fout.open(OPName.c_str());
    for(int j=0; j<=t/100; j++)
        fout << left << setw(7) << j * 100 << '\t' << left << setw(7) << Out_number[j] << "\n";   
    fout<<flush;fout.close();

    cout<<"The" <<k<<"th running ends"<<"\n";
    return 0;
}

