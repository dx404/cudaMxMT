#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include "seqMatrix.h"
#include <omp.h>
using namespace std;

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

//void ompMxMT_v1(float *mr, float *m, int d){
//	int w = 8;
//	for (int t = 0; t < d; t+=w){
//		for (int j = t; j < t + d; j++){
//			float sum = 0;
//			for (int i = 0; i < d; i++){
//				sum += m[i*d+j] * m[j*d+k];
//			}
//			mr[i*d+j] += sum;
//		}
//	}
//}
