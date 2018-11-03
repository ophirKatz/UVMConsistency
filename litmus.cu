#include <stdio.h>
#include <assert.h>
#include <iostream>

#define CUDA_CHECK(f) do {                                                                \
  cudaError_t e = f;                                                                      \
  if (e != cudaSuccess) {                                                                 \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
      exit(1);                                                                            \
  }                                                                                       \
} while (0)

#define START         0
#define START_GPU     1
#define GPU_FINISHED  2
#define FINISH        3


__global__ void kernel(volatile int *x, volatile int *y, volatile int *finished) {
  while (*finished != START_GPU);

  *x = 1;
  *y = 1;

  *finished = GPU_FINISHED;

  while (*finished != FINISH);
}

void consistency(volatile long *x, volatile long *y, volatile int *finished) {
  *finished = START_GPU;
  
  long lv = (*x << 32) & *y;
  if (lv == 1L) {
    std::cout << "Found Inconsistency !" << std::endl;
  } else {
    std::cout << "No Inconsistency Found" << std::endl;
  }
  
  while (*finished != GPU_FINISHED);

  *finished = FINISH;
}

int main() {
  int *x, *y, *finished;
  CUDA_CHECK(cudaMallocManaged(&x, sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&y, sizeof(int)));
  memset((void *) x, 0, sizeof(int));
  memset((void *) y, 0, sizeof(int));

  CUDA_CHECK(cudaMallocManaged(&finished, sizeof(int)));
  memset((void *) finished, START, sizeof(int));

  __sync_synchronize();

  kernel<<<1,1>>>(x, y, finished);
  consistency((long *) x, (long *) y, finished);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree((int *) x));
  CUDA_CHECK(cudaFree((int *) y));
  CUDA_CHECK(cudaFree((int *) finished));
  

  return 0;
}