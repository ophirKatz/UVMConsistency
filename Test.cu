
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <bitset>

#define CUDA_CHECK(f) do {                                                                \
  cudaError_t e = f;                                                                      \
  if (e != cudaSuccess) {                                                                 \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
      exit(1);                                                                            \
  }                                                                                       \
} while (0)

#define ONLY_THREAD if (threadIdx.x == 0)

#define OUT
#define UVMSPACE      volatile


#define START         0
#define GPU_START     1
#define GPU_FINISH    2
#define FINISH        3

#define NUM_SHARED 1000

typedef unsigned long long int ulli;



__global__ void kernel(volatile int *arr, volatile int *finished) {
  // Wait for CPU
  printf("before GPU_START loop\n");
  while (*finished != GPU_START);
  
  printf("Starting loop\n");
  for (int i = 0; i < NUM_SHARED; i++) {
    arr[i] = 1;
		// for (int j = 0; j < 100000; j++);
  }
  printf("After loop\n");
  
  // GPU finished - CPU can finish
  *finished = GPU_FINISH;

  // Wait for CPU to finish
  while (*finished != FINISH);
}

bool is_full(volatile int *arr) {
  int count = 0;
  for (int i = 0; i < NUM_SHARED; i++) {
    count += arr[i];
  }
  return count == NUM_SHARED;
}

void print_arr(volatile int *arr) {
  printf("[");
  for (int i = 0; i < NUM_SHARED; i++) {
    printf("%d,", arr[i]);
  }
  printf("[\n");
}

void CPU() {
  volatile int *arr;
  volatile int *finished;

  CUDA_CHECK(cudaMallocManaged(&arr, sizeof(int) * NUM_SHARED));
  memset((void *) arr, 0, sizeof(int) * NUM_SHARED);

  CUDA_CHECK(cudaMallocManaged(&finished, sizeof(int)));
  memset((void *) finished, START, sizeof(int));

  kernel<<<1,1>>>(arr, finished);

  // GPU can start
  *finished = GPU_START;

  while (!is_full(arr)) {
    print_arr(arr);
  }

  while (*finished != GPU_FINISH);
  // Task is over
  *finished = FINISH;
}

int main() {
  CPU();

  return 0;
}
