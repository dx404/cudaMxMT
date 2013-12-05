#include "cudaGFlopTimer.cuh"

class cudaMxMT_agent{
private:
	float impl_version;
	int dim;
	size_t byteSize;
	int init_method;
	bool disp_result;
	const int BW;
	cudaGFlopTimer *cgTimer, *cgTimerCopy;
	bool san_check;
	float L2_diff;

	dim3 gDim;
	dim3 bDim;

	float *hostSrcMatrix;
	float *hostTargetMatrix;
	float *hostCheckMatrix;

	float *deviceSrcMatrix;
	float *deviceTargetMatrix;

public:
	cudaMxMT_agent(int);

	~cudaMxMT_agent();

	void cudaMxMT_init_rand(int);

	void cudaMxMT_calculate(int);

	void check();

	void display_Matrix();

	void printPofile(bool);
};
