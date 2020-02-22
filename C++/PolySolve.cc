//Here begins the file PolySolve.cc

#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

float Poly(int order, int coef[], float result);

int main()
{
	float root;
	int i, order;
	
	cout << "Enter the order of the polynomial equation you are intersted" << endl; 	
	cin >> order;
	cout << "Enter a guess for the root" << endl; 	
	cin >> root;
	
	int *coef = new int[order];
	
	for (i=0; i<=order; i++)
	{
		cout << "Enter the coefficient x^" << i << endl; 	
		cin >> coef[i];
	}
		
	root = Poly(order, coef, root);
	
	cout << "x =  " << root << endl;
	
	delete [] coef;
	
	return 0;
		
}

float Poly(int order, int coef[], float result)
{
	float *dif_coef = new float[order], func = 0, dif_func = 0, temp_func = 0, temp_dfunc = 0;
	int i, j, k, l, m, n;
	
	for (i=0; i<=order; i++)
	
	{ 	
		dif_coef[i] = coef[i] * i;
	}
	
	for (n=0; n<21; ++n)	
	{
		func = 0;
		dif_func = 0;
		
		for (j=0; j<=order; ++j)
		{
			if (j == 0)
			{
				temp_func = 1;
			} else {
				temp_func = result;
			}
			
			for (k=0; k<j-1; k++) 
			{
				
				temp_func = temp_func * result;
				
			}
			
			func = func + (temp_func * coef[j]);
			
		} 
		
		for (l=0; l <= order; l++)
		{
			if (l < 2)
			{
				temp_dfunc = 1;
			} else {
				temp_dfunc = result;
			}
			
			for (m=0; m<(l-2); m++)
			{
				temp_dfunc = temp_dfunc * result;
		
			}
			
			dif_func = dif_func + (temp_dfunc * dif_coef[l]);
		
		}
		
		result = result - (func/dif_func);
		
	}
	
	delete [] dif_coef;
	
	return result;

}