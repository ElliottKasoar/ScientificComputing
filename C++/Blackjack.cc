// Here begins file Blackjack.cc

#include <iostream> 
#include <cstdlib> 
#include <ctime> 

using namespace std; 

int main() 

{ 
	int total = 0, card, ans;
	bool stp(false);

	while (not stp) 

	{ 
		cout << "Would you like to draw a card? Please enter 1 for yes and 0 for no" << endl; 
		cin >> ans; 

		if (ans == 0) { 
			stp = true; 
		} else { 
			srand(time(NULL));
			card = (rand()%11)+1;
			total = total + card; 
			cout << "You drew a " << card << "! Your current total is " << total << endl; 

			if (total >= 21) { 
				break; 
			} 
		} 
	} 

	if (total == 21) { 
		cout << "Your final total is " <<total << ". Blackjack!"<< endl; 
	} else if (total > 21) { 
		cout << "Your final total is " <<total << ". Sorry, you lost."<< endl; 
	} else if (total < 21) { 
		cout << "Your final total is " <<total << ". Almost!" << endl; 
	} 
	
	return 0;
}