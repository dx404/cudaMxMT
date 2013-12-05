#include <iostream>
#include "MxMTagent.cuh"
using namespace std;
/**
 * Author: Duo Donald Zhao
 */

int main(int argc, char *argv[]){
	cout << "Hello, Welcome to the world of CUDA Matrix Multiplication!" << endl;
	if (argc < 0){
		cerr << "usage: " << "<program> <version> <matrix dimension> <check> <display>" << endl;
		return -1;
	}

	float impl_version;
	int dim;
	bool san_check;
	bool disp_result;
	while (cin >> impl_version >> dim >> san_check >> disp_result){
		cudaMxMT_agent *cal = new cudaMxMT_agent(dim);
		cal->cudaMxMT_init_rand(false);
		cal->cudaMxMT_calculate(impl_version);
		if (san_check)
			cal->check();
		if (disp_result)
			cal->display_Matrix();
		cal->printPofile(true);
		delete cal;
	}
	return 0;
}
