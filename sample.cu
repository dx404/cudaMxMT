#include <iostream>
#include "src/cudaGFlopTimer.cuh"
using namespace std;

int main(){
  cudaGFlopTimer *timer = new cudaGFlopTimer();
  timer->start();
  return 0;
}
