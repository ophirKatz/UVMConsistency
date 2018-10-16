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

#define START       0
#define CAN_WRITE   1
#define AFTER_WRITE 2
#define CAN_READ    3
#define AFTER_READ  4
#define FINISH      5

__device__ void writer_thread(volatile int *ptr, volatile int *flag) {
  printf("%d\n", __LINE__);
  while (*flag != CAN_WRITE);

  printf("%d\n", __LINE__);
  *ptr = 1;
  *flag = AFTER_WRITE;
  printf("%d\n", __LINE__);
  
  while (*flag != FINISH);
}

__device__ void reader_thread(volatile int *ptr, volatile int *flag, volatile int *out) {
  printf("%d\n", __LINE__);
  while (*flag != CAN_READ);
  
  printf("%d\n", __LINE__);
  *out = *ptr;
  *flag = AFTER_READ;
  printf("%d\n", __LINE__);

  while (*flag != FINISH);
}


__global__ void kernel(volatile int *ptr, volatile int *flag, volatile int *out) {
  if (blockIdx.x == 0) {
    writer_thread(ptr, flag);
  } else {
    reader_thread(ptr, flag, out);
  }
}

int main() {
  volatile int *ptr, *flag, *out;
  CUDA_CHECK(cudaMallocManaged(&ptr, sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&flag, sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&out, sizeof(int)));
  memset((void *) ptr, 0, sizeof(int));
  memset((void *) flag, START, sizeof(int));
  memset((void *) out, 0, sizeof(int));

  kernel<<<1,2>>>(ptr, flag, out);
  
  printf("*ptr before write is %d\n", *ptr);
  *flag = CAN_WRITE;
  while (*flag != AFTER_WRITE);
  *flag = CAN_READ;
  while (*flag != AFTER_READ);
  printf("*ptr after write is %d\n", *out);
  *flag = FINISH;

  CUDA_CHECK(cudaFree((int *) ptr));
  CUDA_CHECK(cudaFree((int *) flag));
  CUDA_CHECK(cudaFree((int *) out));

  return 0;
}
