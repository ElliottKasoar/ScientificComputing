// Here begins file 2D_particles.cc

// Video  of N particles in a 2D box can be created with gnu.particles

// How to run:
// 1. Make 2D_particles.cc with gnuplot=1, EPcheck=0
// 2. Run 2D_particles.cc -> partprint
// 3. Make 2D_particles.cc with gnuplot=0, EPcheck=0
// 4. Run 2D_particles.cc -> tmp/file10
// 5. (Optional check) Make 2D_particles.cc with gnuplot=0, EPcheck=1
// 6. (Optional check) Run 2D_particles.cc
// 7. load 'gnu.particles' in gnuplot

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>

using namespace std;

const int N=24;	//total number of particles (including four walls)

#define D 2  // number of dimensions

struct particle {
  double x[D]; 	// position
  double p[D]; 	// momentum
  double im; 	// inverse mass
  double v[D]; 	// velocity
  double T; 	// kinetic energy
  double r;		// radius of particle
};

#define ranf() \
  ((double)rand()/(1.0+(double)RAND_MAX)) // Uniform from interval [0,1) */

void ShowState(particle a[])
{
	int i=0, j=0, k=0;
	
	for (i=0; i<D; i++) 
	{ 
		for (j=2; j<N-2; j++)
		{
			cout << "\t"<<a[j].x[i];
		}
	}
   
	cout << endl;
  
	cout << "#";
  
	for (i=0; i<D; i++) 
	{ 
		for (j=2; j<N-2; j++)
		{
			cout << "\tP" << (j-1) << "x["<<k<<"]\t";
		}
	}
	
	cout << endl;    
 
}

double KE(particle a[])
{
	// calculate kinetic energy
	
	double totKE=0.0;
	int i, j;
	
	for (i=0; i<N; i++)
 	{
 		a[i].T = 0;
 	}	
	
	for (i=0; i<D; i++)
 	{
 		for (j=2; j < N-2; j++)
 	 	{
 	 		a[j].T += (0.5 * a[j].v[i] * a[j].v[i]) / a[j].im;
 	 	}
 	}
 	 
	for (j=0; j<N; j++)
 	{
 		totKE += a[j].T;
 	}	
 	
 	return totKE;
}

void Mom(particle a[], int part1, int part2, double P[])
{
	// calculate momentum of two particles
	
	int i;
	
	for (i=0; i<D; i++)
	{
		if (part1==0 || part1==1)
		{
			a[part1].p[i] = 0;
			a[part2].p[i] = a[part2].v[i] / a[part2].im;
		} else if (part2 == N-2 || part2 == N-1)
		{
			a[part1].p[i] = a[part1].v[i] / a[part1].im;
			a[part2].p[i] = 0;
		} else {
			a[part1].p[i] = a[part1].v[i] / a[part1].im;
			a[part2].p[i] = a[part2].v[i] / a[part2].im;
		}
		
		P[i] = a[part1].p[i] + a[part2].p[i];
	}
}


void PositionStep(particle a[], double dt)
{
	// update position of all particles
	
 	int i=0, j=0;
 	 
 	for (i=0; i<D; i++)
 	{
 	 	for (j=0; j<N; j++)
 	 	{
 	 	 	a[j].x[i] += dt * a[j].v[i];
 	 	}
 	}
}
 
void collide(particle a[], int part1, int part2, bool prevcoll[N][N], bool EPcheck)
{
	// simulate collision of two particles
	
 	int i, j;
 	double  *vzm, *dir, usqr[2], u[2], vsqr[2], v[2], dirmag=0.0, dirmagsqr=0.0, partspeed=0.0, *P, *initialP, *finalP, *changeP; 
 	vzm = new double[D];
 	dir = new double[D]; 
 	P = new double[D];
 	initialP = new double[D];
 	finalP = new double[D];
 	changeP = new double[D];
 	
 	//intitialise variables
 	 
 	for (i=0; i<D; i++)
 	{
 	 	vzm[i]=0;
 	 	dir[i]=0;
 	}
 	 
 	for (i=0; i<2; i++)
 	{
 	 	usqr[i]=0;
 	 	u[i]=0;
 	 	vsqr[i]=0;
 	 	v[i]=0;
 	}
 	 
 	//intial momentum:
 	
 	for (i=0; i<D; i++)
 	{
 		Mom(a, part1, part2, P);
 		initialP[i] = P[i];
 	}
 	 
 	 if (part1==0 || part1==1)	//vertical wall
 	 {
 	 	 
 	 	 partspeed = a[part2].v[0];
 	 	 a[part2].v[0] -= 2 * partspeed;
 	 	 
 	 } else if (part2==N-2 || part2==N-1)	//horizontal walls
 	 {
 	 	 
 	 	 partspeed = a[part1].v[1];
 	 	 a[part1].v[1] -= 2 * partspeed; 	 
 	 	 
 	 } else {	//ZMF for particles
 	 	 
 	 	 for (i=0; i<D; i++)
 	 	 {
 	 	 	 vzm[i] = (a[part1].v[i]/a[part1].im) + (a[part2].v[i]/a[part2].im);
 	 	 	 vzm[i] /= ((1/a[part1].im) + (1/a[part2].im));
 	 	 	 
 	 	 	 a[part1].v[i] -= vzm[i];
 	 	 	 a[part2].v[i] -= vzm[i];
 	 	 	 
 	 	 	 	 	 	 
 	 	 	 dir[i] = a[part1].x[i] - a[part2].x[i];
 	 	 	 dirmagsqr += dir[i] * dir[i];
 	 	 	 
 	 	 	 usqr[0] += a[part1].v[i]*a[part1].v[i];
 	 	 	 usqr[1] += a[part2].v[i]*a[part2].v[i];
 	 	 }
 	 	 
 	 	 u[0] = sqrt(usqr[0]);
 	 	 u[1] = sqrt(usqr[1]);
 	 	 
 	 	 v[0]=u[0];
 	 	 v[1]=u[1];
 	 	 
 	 	 dirmag = sqrt(dirmagsqr);
 	 	 
 	 	 for (i=0; i<D; i++)
 	 	 {
 	 	 	 dir[i] /= dirmag;
 	 	 	 
 	 	 	 a[part1].v[i] = v[0] * dir[i];
 	 	 	 a[part2].v[i] = -v[1] * dir[i]; 
 	 	 	 
 	 	 	 a[part1].v[i] += vzm[i];
 	 	 	 a[part2].v[i] += vzm[i];
 	 	 }
 	 	 
 	 }
 	 
  	 // update prevcoll
  	 
  	 for (i=0; i<N; i++)
 	 {
 	 	 for (j=0; j<N; j++)
 	 	 {
 	 	 	if (i==part1 && j==part2)
 	 	 	{
 	 	 		prevcoll[i][j] = 1;
 	 	 	} else {
 	 	 		prevcoll[i][j] = 0;
 	 	 	}
 	 	 }
 	 }  
 	 
 	//check for change in momentum
 	
 	for (i=0; i<D; i++)
 	{
 		Mom(a, part1, part2, P);
 		finalP[i] = P[i];
 	}
 	 
	for (i=0; i<D; i++)
	{
		changeP[i] = finalP[i] - initialP[i];
	}
	
	if (part1 > 1 && part2 < N-2 && EPcheck)
	{
		cout << part1 << " and " << part2 << ". Change in Px: " << changeP[0] << ". Change in Py: " << changeP[1] << endl;
	}
	
	delete [] vzm;
	delete [] dir;
	delete [] P;
	delete [] initialP;
	delete [] finalP;
	delete [] changeP;
}
 
 void checkcollision(particle a[], bool prevcoll[N][N], bool EPcheck)
 {
	// check which particles are colliding
	
 	 int i, j, k, l, m, n, o, p, part1, part2;
 	 double xdist[N][N], ydist[N][N], totdistsqr[N][N];
 	 
 	 // initialise distances as arbitrary large number
 	 // may require increase if box size increases
 	 
 	 for (k=0; k<N; k++)
 	 {
 	 	 for (l=0; l<N; l++)
 	 	 {
 	 	 	 xdist[k][l] = 100.0;
 	 	 	 ydist[k][l] = 100.0;
 	 	 }
 	 }
 	 
 	 // calculate xdist
 	 
 	 for (i=0; i<N; i++)
 	 {
 	 	 for (j=0; j<N; j++)
 	 	 {
 	 	 	 if (i != j && xdist[j][i] == 100.0)
 	 	 	 {
 	 	 	 	 if (j==N-2 || j==N-1)
 	 	 	 	 {
 	 	 	 	 	 xdist[i][j] = 0.0;
 	 	 	 	 } else {
 	 	 	 	 	 xdist[i][j] = a[i].x[0] - a[j].x[0];
 	 	 	 	 }
 	 	 	 }
 	 	 }
 	 }	 
 	 
 	 // calculate ydist
 	 
 	 for (m=0; m<N; m++)
 	 {
 	 	 for (n=0; n<N; n++)
 	 	 {
 	 	 	 if (m != n && ydist[n][m] == 100.0)
 	 	 	 {
 	 	 	 	 if (m==0 || m==1)
 	 	 	 	 {
 	 	 	 	 	 ydist[m][n] = 0.0;
  	 	 	 	 } else {
 	 	 	 	 	 ydist[m][n] = a[m].x[1] - a[n].x[1];
 	 	 	 	 } 
 	 	 	 }
 	 	 }
 	 }
 	 
 	//total distance
 	
 	for (o=0; o<N; o++)
 	 {
 	 	 for (p=0; p<N; p++)
 	 	 {
 	 	 	 totdistsqr[o][p] = ydist[o][p] * ydist[o][p] + xdist[o][p] * xdist[o][p];
 	 	 	  	 	 	 
 	 	 	 if(totdistsqr[o][p] < (a[o].r * a[o].r) + (a[p].r * a[p].r) && not prevcoll[o][p])
 	 	 	 {
 	 	 	 	 part1 = o;
 	 	 	 	 part2 = p;
 	 	 	 	 collide(a, part1, part2, prevcoll, EPcheck);

 	 	 	 }
 	 	 }
 	 }  	  	 	
 }
 

void Motion(particle a[], double dt, int steps, bool prevcoll[N][N], bool EPcheck)
{
	// calls functions to update positions, check for collisions and print states
	
	int i=0;
	double t=0.0;
	
	for (i=0; i<steps; i++)
	{	
		t += dt;	
		PositionStep(a, dt);
		checkcollision(a, prevcoll, EPcheck);
		
		if (not EPcheck)
		{
			ShowState(a);
		}
	}
}

int main()
{
	particle a[N];
	int i, j;
	
	// number of iterations, if changed may require changes in gnu.particles
	int steps = 10000; 
	
	// time step between iterations must be small enough to prevent particles
	// passing through each other - consider max KE	
	double dt = 0.001; 

	
	// checks if two particles have collided to prevent repeats
	bool prevcoll[N][N]; 
	
	// kinetic energy variables
	double KE1, KE2, dKE; 
	
	// set equal to 1 to print out energy/momentum checks only, else 0
	bool EPcheck=0; 
	
	// set equal to 1 and run '-> partprint' to create file for gnuplot, else 0
	bool gnuplot=0; 
	
 	srand(time(NULL));
	
 	// inititalise prevcoll
 	for (i=0; i<N; i++)
 	 {
 	 	 for (j=0; j<N; j++)
 	 	 {
 	 	 	 prevcoll[i][j]=0;
	 	 }
 	 }
	
 	 // generate N-4 particles with random positions and velocities within the box
 	 // particles may initially overlap but should not do so again once they move apart 
 	 
 	 for (i=2; i<N-2; i++)
 	 {
 	 	 a[i].im = 1.0;
 	 	 a[i].v[0] = 2 * ((2 * ranf()) - 1 );
 	 	 a[i].v[1] = 2 * ((2 * ranf()) - 1);
 	 	 a[i].x[0] = 0.5 + (9 * ranf());
 	 	 a[i].x[1] = 0.5 + (9 * ranf());
 	 	 a[i].r = 0.5; 	 // if changed, x[0] and x[1] may need to be updated
 	 }
 	 
	// walls
	// if positions changed, x[0] and x[1] may need to be updated
	
	a[0].im = 0.0;	//line x=0 (vert)
	a[0].v[0] = 0.0;
	a[0].v[1] = 0.0;
	a[0].x[0] = 0.0;
	a[0].x[1] = 0.0;
	a[0].r = 0.0;
	
	a[1].im = 0.0;	//line x=10 (vert)
	a[1].v[0] = 0.0;
	a[1].v[1] = 0.0;	
	a[1].x[0] = 10.0;
	a[1].x[1] = 0.0;
	a[1].r = 0.0;

	a[N-2].im = 0.0;	//line y=0 (hor)
	a[N-2].v[0] = 0.0;
	a[N-2].v[1] = 0.0;
	a[N-2].x[0] = 0.0;
	a[N-2].x[1] = 0.0;
	a[N-2].r = 0.0;

	
	a[N-1].im = 0.0;	//line y=10 (hor)
	a[N-1].v[0] = 0.0;
	a[N-1].v[1] = 0.0;
	a[N-1].x[0] = 0.0;
	a[N-1].x[1] = 10.0;
	a[N-1].r = 0.0;
	
	if (not gnuplot)
	{
		KE1 = KE(a);
		
		Motion(a, dt, steps, prevcoll, EPcheck);
		
		KE2 = KE(a);
		
		dKE = KE1 - KE2;
		
		if (EPcheck)
		{
			cout << "Change in KE: " << dKE << endl;
		}
		
	} else {
		for (i=1; i<N-3; i++)
		{
			if (i==1)
			{
				cout << "plot 'tmp/file10' every 1::t::t u 1:" << (N-3)<<  " w p ps 5 pt 5,\\";
			} else if (i==N-4)
			{
				cout << "'tmp/file10' every 1::t::t u " << i << ":" << (i+N-4) << " w p ps 5 pt 5";
			} else {
				cout << "'tmp/file10' every 1::t::t u " << i << ":" << (i+N-4) << " w p ps 5 pt 5,\\";
			}
			
			cout << endl;
		}
	}
	
	return 0;	
}
