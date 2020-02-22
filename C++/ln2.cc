// Here begins file ln2.cc

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

using namespace std;

#define ranf() \
  ((double)random()/(1.0+(double)RAND_MAX)) // Uniform from interval [0,1]

int main()
{
	int    outcome = 0, count_in=0 , seed=123, count=0, n=1;
	double fraction_in = 0, prev_fraction = 0, fraction_initial = 0 ;
  
	// Initialise random number generator 
	
	srandom(seed);
 
	while (count < 100000) 
	{
		double x = 1 + ranf();
		double y = ranf();
		outcome = ( x * y < 1.0 ) ? 1 : 0 ; 
		count_in += outcome; 
		fraction_in = static_cast<double>(count_in)/n;
		
		if (count == 0)
		{
			fraction_initial = fraction_in;
		}
		
		if (abs(fraction_in - prev_fraction) < 0.005) 
		{
			count += 1;
		} else {
			count = 0;
		}
		prev_fraction = fraction_in;
		n += 1;
		
		if (count == 100000)
		{
			if (abs(fraction_in - fraction_initial) > 0.005) 
			{
				count = 0;
			}	
		}
	}
  
	cout << "ln(2) is approximately: " << fraction_in << endl;
  
	return 0;
}
