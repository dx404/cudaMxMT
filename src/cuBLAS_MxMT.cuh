#include "cudaGFlopTimer.cuh"

void cuBLAS_MxMT_device(float *d_r, float *d_m, int d);

float cuBLAS_MxMT_host(float *h_r, float *h_m, int d);
