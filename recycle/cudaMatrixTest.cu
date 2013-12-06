#include <iostream>
#include <cstdio.h>
using namespace std;

const int d = 8;
__global__ void displayMaxtrix(){

}

void diplayMatrix(float *m, int d){
	cout << "m={" << endl;
	for (int i = 0; i < d; i++){
		cout << "\t{";
		for (int j = 0; j < d - 1; j++){
			cout << setw(5) << m[i*d+j] << ", ";
		}
		cout << setw(5) << m[i*d + d - 1] <<
				((i==d-1)? "}" : "},") << endl;
	}
	cout << "};" << endl;
}

int main(int argc, char *argv[]){
	int h_m[64];
	for (int i = 0; i < d; i++){
		for (int j = 0; j < d; j++){
			h_m[i*d + j] = i * 1000 + j;
		}
	}
	diplayMatrix(h_m, d);
	return 0;
}
