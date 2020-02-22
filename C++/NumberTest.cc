#include <iostream>
#include <cmath>

using namespace std;

int main()

{
	int num=0, i, prev1=1, prev2=0, fibtot=0;
	bool prime = true, error = false, even = false, fib = false;

	do
	{
		cout << "Enter the positive integer you would like to know about: " <<endl;
		cin >>num;
	
		if (num <= 0)
		{
			error = true;
			cout << "Please make sure the number you entered is a positive" << endl;
		} else {
			error = false;
		}
		
	} while (error);
	
	//Odd or even
	
	if (num%2 == 0) 
	{
		even = true;
	}
	
		if (even)
	{
		cout << num << " is even!" << endl;
	} else {
		cout << num << " is odd!" << endl;
	}
	
	//Prime number test
	
	for (i=2;i<=num/2;++i)
	{
		if (num%i==0)
		{
			prime = false;
			break;
		}
	}		
		
	if (prime && num != 1)
	{
		cout << num << " is prime!" << endl;
	} else {
		cout << num << " is not prime." << endl;
	}
	

	//Fibonacci
		
	while (fibtot < num)
	{
		fibtot = prev1 + prev2;
		if (num == fibtot)
		{
			fib = true;
			break;
		}
		
		prev2 = prev1;
		prev1 = fibtot;
	}
	
	if (fib)
	{
		cout << num << " is part of the Fibonacci series!" << endl;
	} else {
		cout << num << " is not part of the Fibonacci series." << endl;
	}
	
}