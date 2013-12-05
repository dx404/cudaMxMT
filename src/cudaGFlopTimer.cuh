#ifndef CUDAGFLOPTIMER_CUH
#define CUDAGFLOPTIMER_CUH

class cudaGFlopTimer {
private:
	cudaEvent_t s, t;
	float time;
public:
	cudaGFlopTimer();
	void start();
	void stop();
	float getElapsedTime();
	float getGFlops(float d);
};

#endif
