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
#define GPU_HOLD      3
#define CPU_HOLD      4
#define GPU_HOLD_DONE      55
#define CPU_HOLD_DONE      56
#define GPU_FINISH    5
#define FINISH        6

#define PAGE_SIZE     64 * 1024          // This is the size of a memory page in the tested GPU system [64K]
#define NUM_SHARED    ((2 * (PAGE_SIZE)) / sizeof(int))    // So the array will span at-least 2 memory pages

#define NUM_BLOCKS  1

typedef unsigned long long int ulli;

__global__ void GPU_UVM_Writer_Kernel(UVMSPACE int *kernel_arr, UVMSPACE int *kernel_finished) {
  UVMSPACE int *arr = kernel_arr + blockIdx.x * NUM_SHARED;
  UVMSPACE int *finished = kernel_finished + blockIdx.x;
  
  // Wait for CPU
  while (*finished != GPU_START);

  // Loop and execute writes on shared memory page - sequentially
  for (int i = 0; i < NUM_SHARED; i++) {
    // For Consistency Check
		if (i * sizeof(int) == PAGE_SIZE) {
			*finished = CPU_HOLD;
			while (*finished != GPU_HOLD); // { printf("[kernel]	while finished != GPU_HOLD\n"); }
		}

    arr[i] = 1;	// Write

    __threadfence_system();
		if (i * sizeof(int) == PAGE_SIZE) {
			while (*finished != CPU_HOLD_DONE);
		}
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
			if ((i * sizeof(int)) == PAGE_SIZE) {
				printf("[cpu]	times in if %d\n", *finished);
				while (*finished != CPU_HOLD); // { printf("[CPU]	while finished != CPU_HOLD   finished = %d\n", *finished); }
				*finished = GPU_HOLD;
			}
      long value = *((long *) (arr + i));

      if (value == maxLong) {  // arr[i] == 0 and arr[i + 1] == 1  ==> Inconsistency
        return true;
      }
			if ((i * sizeof(int)) == PAGE_SIZE) {
				*finished = CPU_HOLD_DONE;
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
