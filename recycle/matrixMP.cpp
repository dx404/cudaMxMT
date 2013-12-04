#include <iostream>
using namespace std;

void seq_MxMT(float *mr, float *m, int d){

	for (int i = 0; i < d; i++){
		for (int j = 0; j < d; j++){
			float sum = 0;
			for (int k = 0; k < d; k++){
				sum += m[i*d+k] * m[j*d+k];
			}
			mr[i*d+j] = sum;
		}
	}
}




int main(int argc, char *argv[]){

	return 0;
}
