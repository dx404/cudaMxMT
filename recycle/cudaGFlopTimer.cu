#include "cudaGFlopTimer.cuh"

Hi::Hi(){
  x = 5;
}

cudaGFlopTimer::cudaGFlopTimer(){
	cudaEventCreate(&s);
	cudaEventCreate(&t);
}

void cudaGFlopTimer::start(){
	cudaEventRecord(s);
}

void cudaGFlopTimer::stop(){
	cudaEventRecord(t);
	cudaEventSynchronize(t);
	cudaEventElapsedTime(&time, s, t);
}

float cudaGFlopTimer::getElapsedTime(){
	return time;
}

float cudaGFlopTimer::getGFlops(float d){
	return (2.0e-6 * d * d * d) /(time);
}



