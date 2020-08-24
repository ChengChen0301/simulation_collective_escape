#ifndef POINT_H
#define POINT_H

#include "vec.h"

using namespace std;

class Point{
private:
	vec m_C; // coordinate of the point
	vec m_F; // force on the point
	vec m_A; // acceleration of the point
	vec m_V; // velocity of the point
	int m_G[2]; // grid information of the point
	int flag;// indicate whether the point is still in the room
public:

	Point()
	{
	  m_C.set_X(0.0, 0.0);
	  m_F.set_X(0.0, 0.0);
	  m_A.set_X(0.0, 0.0);
	  m_V.set_X(0.0, 0.0);
	  m_G[0] = -1; m_G[1] = -1;
	  flag = 1;
	};

	///////////////////////
	///set member data/////
	///////////////////////
	void set_C(vec arg_C){
		m_C = arg_C;
	};
	void set_V(vec arg_V){
		m_V = arg_V;
	};
	void set_F(vec arg_F){
		m_F = arg_F;
	};
	void set_A(vec arg_A){
		m_A = arg_A;
	};
	void set_G(int arg0, int arg1){
        m_G[0] = arg0;
        m_G[1] = arg1;
	};
    void set_flag(int arg){
        flag = arg;
    };

	///////////////////////
	///visit member data///
	///////////////////////
	vec get_C(){
		return m_C;
	};
	vec get_V(){
		return m_V;
	};
	vec get_F(){
		return m_F;
	};
    vec get_A(){
		return m_A;
	};
	int *get_G(){
        return m_G;
    };
    int get_flag(){
        return flag;
    };
};

#endif
