#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <math.h>

using namespace std;

#define CUDA_CHECK(f) do {                                                                \
  cudaError_t e = f;                                                                      \
  if (e != cudaSuccess) {                                                                 \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
      exit(1);                                                                            \
  }                                                                                       \
} while (0)

#define ONLY_THREAD if (threadIdx.x == 0)



__global__ void kernel(volatile int *ptr, volatile int *flag) {
  *ptr = 1;
  *flag = 1;
  while (*flag != 2);
}

int main() {
  volatile int *ptr, *flag;
  CUDA_CHECK(cudaMallocManaged(&ptr, sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&flag, sizeof(int)));
  memset((void *) ptr, 0, sizeof(int));
  memset((void *) flag, 0, sizeof(int));

  cout << __LINE__ << endl;
  kernel<<<1,1>>>(ptr, flag);
  cout << __LINE__ << endl;
  
  while (*flag != 1);
  printf("*ptr = %d\n", *ptr);
  *flag = 2;

  CUDA_CHECK(cudaFree((int *) ptr));
  CUDA_CHECK(cudaFree((int *) flag));

  return 0;
}
