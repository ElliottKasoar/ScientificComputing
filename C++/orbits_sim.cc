// Here begins the file orbits_sim.cc

// Works with gnu.orbitsim
// Output file to output/6 for video to work

#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

#define D 2 // number of dimensions
struct particle {
	double x[D]; // (x,y) coordinates
	double v[D] ; //velocity
	double F[D]; // force
	double GMm1; // grav parameter of comet for Earth
	double GMm2; //grav parameter of comet for moon
	double p[D]; // momentum
	double T; // kinetic energy
	double V; // potential energy
	double r; // distance from origin
	double m; // mass
	double im; // mass of particle	
	double J; // angular momentum
	double E; // total energy
} ;

void ShowState(particle &earth, particle &moon, particle &comet)
{
	int i=0, j=0, k=0, l=0;
	
	for (i=0; i<D; i++) 
	{ 
		cout << "\t"<<moon.x[i] << "\t"<<comet.x[i];
	}
	for (j=0; j<D; j++) 
	{ 
		cout << "\t"<<moon.v[i] << "\t"<<comet.v[i];
	}
   
	cout << endl;
  
	cout << "#";
  
	for (k=0; k<D; k++) 
	{ 
		cout << "\tmoonx["<<k<<"]\t" << "\tcometx["<<k<<"]\t"; 
	}
  
	for (l=0; l<D; l++) 
	{ 
		cout << "\tmoonv["<<l<<"]\t" << "\tcometv["<<l<<"]\t";
	}

	cout << endl;
}

double MoonDist(particle &moon)
{
	double rsqrd=0.0, r=0;
	int i=0;
	
	for (i=0; i<D; i++)
	{
		rsqrd += moon.x[i] * moon.x[i];
	}
	
	r = sqrt(rsqrd);
	return r;
}

double CometDistEarth(particle &comet)
{
	double rsqrd=0.0, r=0;
	int i=0;
	
	for (i=0; i<D; i++)
	{
		rsqrd += comet.x[i] * comet.x[i];
	}
	
	r = sqrt(rsqrd);
	return r;
}

double CometDistMoon(particle &comet, particle &moon)
{
	double rsqrd=0.0, r=0;
	int i=0;
	
	for (i=0; i<D; i++)
	{
		rsqrd += (comet.x[i] - moon.x[i]) * (comet.x[i] - moon.x[i]);
	}
	
	r = sqrt(rsqrd);
	return r;
}

void MoonForce(particle &moon)
{
	int i=0;
	
	for (i=0; i<D; i++)
	{
		moon.F[i] = -moon.GMm1 * moon.x[i] / pow(MoonDist(moon),3);
	} 

}

void CometForce(particle &comet, particle &moon)
{
	int i=0;
	
	for (i=0; i<D; i++)
	{
		comet.F[i] = (-comet.GMm1 * comet.x[i] / pow(CometDistEarth(comet),3)) + (-comet.GMm2 * (comet.x[i] - moon.x[i]) / pow(CometDistMoon(comet, moon),3)) ;
	} 

}

void CometPositionStep(particle &comet, double dt)
{ 
	int i=0;
	
	for (i=0 ; i<D ; i++ ) 
	{	
		comet.x[i] += dt * comet.v[i] ; 
	}
}

void MoonPositionStep(particle &moon, double dt)
{ 
	int i=0;
	
	for (i=0 ; i<D ; i++) 
	{	
		moon.x[i] += dt * moon.v[i] ; 
	}
}

void MoonVelocityStep(particle &moon, double dt)
{
int i=0;
	
	for (i=0; i<D; i++) 
	{	
		moon.v[i] += moon.F[i] * moon.im * dt;
	}
}

void CometVelocityStep(particle &comet, double dt)
{
int i=0;
	
	for (i=0; i<D; i++) 
	{	
		comet.v[i] += comet.F[i] * comet.im * dt;
	}
}

void Dynamics(particle &earth, particle &moon, particle &comet, double dt, double &t, int n)
{
	int i=0;
	
	for (i=0; i<n; i++) 
	{
		cout << t ; ShowState(earth,moon,comet); // print out the state
		
		MoonPositionStep(moon, dt*0.5 ); // increment moon position a half-step
		CometPositionStep(comet, dt*0.5 ); // increment comet position a half-step
		
		t += 0.5 * dt; // increment time a half-step
		
		MoonForce(moon); // force on the moon at this position
		CometForce(comet, moon); // force on the comet at this position

		MoonVelocityStep(moon, dt); //increment moon velocity
		CometVelocityStep(comet, dt); //increment comet velocity

		MoonPositionStep(moon, dt*0.5 ); // increment moon position a half-step
		CometPositionStep(comet, dt*0.5 ); // increment comet position a half-step		
		
		t += 0.5*dt; // increment time a half-step
	}
}

int main()
{
	particle earth;
	particle moon;
	particle comet;
	
	// Earth 
	earth.v[0] = 0.0;
	earth.v[1]  = 0.0;
	earth.x[0] = 0.0;
	earth.x[1] = 0.0;
	earth.m = 5.972 * pow(10,24);
	earth.im = 1/earth.m;
	earth.GMm1 = 1*earth.m;
	
	// Moon
	moon.v[0] = 0.0;
	moon.v[1]  = 1 * pow(10,3);
	moon.x[0] = 3.84*pow(10,8);
	moon.x[1] = 0.0;
	moon.m = 7.35 * pow(10,22);
	moon.im = 1/moon.m;
	moon.GMm1 = 6.67* pow(10,-11) *moon.m * earth.m;
	
	// Comet
	comet.v[0] = 0.5 * pow(10,3);
	comet.v[1]  = -1.0 * pow(10,2);
	comet.x[0] = 5.0*pow(10,8);
	comet.x[1] = 5.0*pow(10,8);
	comet.m = 2.2 * pow(10,14); 
	comet.im = 1/comet.m;
	comet.GMm1 = 6.67* pow(10,-11) * comet.m * earth.m;
	comet.GMm2 = 6.67* pow(10,-11) * comet.m * moon.m;
	
	double dt=1000.0; // time step
	double t=0.0;
	int n=10000; 	// number of iterations
	
	Dynamics(earth, moon, comet, dt, t, n);
	
	return 0;
}
