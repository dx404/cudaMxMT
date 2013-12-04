#include <cuda.h>

class cudaGFlopTimer {
private:
	cudaEvent_t s, t;
	float time;
public:
	cudaGFlopTimer();
	inline void start();
	inline void stop();
	inline float getElapsedTime();
	inline float getGFlops(float d);
};
