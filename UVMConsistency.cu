#include <stdio.h>
#include <vector>
#include <thread>
#include <assert.h>
#include <iostream>

#include "cuda_profiler_api.h"

#define CUDA_CHECK(f) do {                                                                \
  cudaError_t e = f;                                                                      \
  if (e != cudaSuccess) {                                                                 \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
      exit(1);                                                                            \
  }                                                                                       \
} while (0)

namespace UVMConsistency {

#define UVMSPACE      volatile

#define START         0
#define GPU_START     1
#define CPU_LOAD      2
#define GPU_FINISH    4
#define FINISH        5

#define NUM_SHARED 10000

#define NUM_BLOCKS  1
#define SINGLE_THREAD

typedef unsigned long long int ulli;

__global__ void GPU_UVM_Writer_Kernel(UVMSPACE int *kernel_arr, UVMSPACE int *kernel_finished) {
  UVMSPACE int *arr = kernel_arr + blockIdx.x * NUM_SHARED;
  UVMSPACE int *finished = kernel_finished + blockIdx.x;
  
  // Wait for CPU
  while (*finished != GPU_START);

  // Loop and execute writes on shared memory page - sequentially
  for (int i = 0; i < NUM_SHARED; i++) {
    // For Consistency Check
    arr[i] = 1;
     __threadfence_system();
  }
  
  // GPU finished - CPU can finish
  *finished = GPU_FINISH;

  // Wait for CPU to finish
  // while (*finished != FINISH);
}

class Consistency {
private:	// Constructor & Destructor
  Consistency() {
    CUDA_CHECK(cudaMallocManaged(&arr, sizeof(int) * NUM_SHARED * NUM_BLOCKS));
    memset((void *) arr, 0, sizeof(int) * NUM_SHARED * NUM_BLOCKS);

    CUDA_CHECK(cudaMallocManaged(&finished, sizeof(int) * NUM_BLOCKS));
    memset((void *) finished, START, sizeof(int) * NUM_BLOCKS);

    // Writing all the changes of UM to GPU
    __sync_synchronize();
  }

  ~Consistency() {
    CUDA_CHECK(cudaFree((int *) arr));
    CUDA_CHECK(cudaFree((int *) finished));
  }
  
private:	// Logic
  bool is_arr_full(UVMSPACE int *arr) const {
    int count = 0;
    for (int i = 0; i < NUM_SHARED; i++) {
      count += arr[i];
    }
    return count == NUM_SHARED;
  }

  bool check_consistency_on_arr(UVMSPACE long *arr) const {
    // Read shared memory page - sequentially
		static const long maxLong = 4294967296L;
    for (int i = 0; i < NUM_SHARED - 1; i++) {
      long value = *((long *) (arr + i));

      if (value == maxLong) {  // arr[i] == 0 and arr[i + 1] == 1  ==> Inconsistency
        return true;
      }
    }
    return false;
  }
  
  void launch_task() {
    // Start GPU task
    GPU_UVM_Writer_Kernel<<<NUM_BLOCKS,1>>>(arr, finished);
  }

  void check_consistency(UVMSPACE int *arr, UVMSPACE int *finished) const {
    // GPU can start
    *finished = GPU_START;

    // While writes have not finished
    while (!is_arr_full(arr)) {
      // Check if an inconsistency exists in the array
      if (check_consistency_on_arr((long *) arr)) {
        ::std::cout << "Found Inconsistency !" << ::std::endl;
        return;
      }
    }
    ::std::cout << "No Inconsistency Found" << ::std::endl;

    // Wait for GPU
    while (*finished != GPU_FINISH);

    // Task is over
    // *finished = FINISH;
  }

  void finish_task() {
    CUDA_CHECK(cudaDeviceSynchronize());
  }
    
public:
  static void handle_threads(const Consistency &consistency) {
    ::std::vector<std::thread> threads;
		for (int i = 0; i < NUM_BLOCKS; i++) {
      threads.push_back(
        ::std::thread(
          &Consistency::check_consistency,
          &consistency,
          consistency.arr + (i * NUM_SHARED),
          consistency.finished + i
        )
      );
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
  
  static void start() {
		Consistency consistency;

    // Start kernel
		cudaProfilerStart();
    consistency.launch_task();
		cudaProfilerStop();

    // Check GPU consistency
    handle_threads(consistency);

		// Finish task for CPU and GPU
		consistency.finish_task();
  }
private:
  UVMSPACE int *arr;
  UVMSPACE int *finished;
};

} // UVMConsistency namespace

int main() {
	UVMConsistency::Consistency::start();

  return 0;
}
