#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include "seqMatrix.h"
#include <omp.h>
using namespace std;

//The sequential version
void ompMxMT_naive(float *mr, float *m, int d){
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

//omp version is under development
