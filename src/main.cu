#include <iostream>
#include "MxMTagent.cuh"
using namespace std;
/**
 * Author: Duo Donald Zhao
 */
int main(int argc, char *argv[]){
	cout << "Hello, Welcome to the world of CUDA Matrix Multiplication for M x MT!" << endl;
	cout << "usage: stdin::<impl_version> <dim> <init_method> <san_check> <disp_matrix> <csv_format>" << endl;

	float impl_version;
	int dim;
	int init_mtd;
	bool san_check;
	bool disp_result;
	bool csv_format;

	// Six parameters to fill from the standard input
	while (cin >> impl_version >> dim >> init_mtd >> san_check >> disp_result >> csv_format){
		cudaMxMT_agent *cal = new cudaMxMT_agent(dim);
		cal->cudaMxMT_init_rand(false);
		cal->cudaMxMT_calculate(impl_version);
		if (san_check)
			cal->check();
		if (disp_result)
			cal->display_Matrix();
		cal->printPofile(csv_format);
		delete cal;
	}
	return 0;
}
