#include <cuda.h>
#include "cudaGFlopTimer.cuh"

cudaGFlopTimer::cudaGFlopTimer(){
	cudaEventCreate(&s);
	cudaEventCreate(&t);
}

void cudaGFlopTimer::start(){
	cudaEventRecord(s);
}

inline void cudaGFlopTimer::stop(){
	cudaEventRecord(t);
	cudaEventSynchronize(t);
	cudaEventElapsedTime(&time, s, t);
}

inline float cudaGFlopTimer::getElapsedTime(){
	return time;
}

inline float cudaGFlopTimer::getGFlops(float d){
	return (2.0e-6 * d * d * d) /(time);
}

int main(){
	cudaGFlopTimer *timer = new cudaGFlopTimer();
	timer->start();
}

