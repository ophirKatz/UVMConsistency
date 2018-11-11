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

namespace UVMConsistency {

#define UVMSPACE      volatile

#define START         0
#define GPU_START     1
#define CPU_LOAD      2
#define GPU_FINISH    4
#define FINISH        5

#define NUM_SHARED 100

typedef unsigned long long int ulli;

__global__ void GPU_UVM_Writer_Kernel(UVMSPACE int *arr, UVMSPACE int *finished) {
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
  while (*finished != FINISH);
}

const long V = 1L << 32;

class Consistency {
private:	// Constructor & Destructor
  Consistency() {
    CUDA_CHECK(cudaMallocManaged(&arr, sizeof(int) * NUM_SHARED));
    memset((void *) arr, 0, sizeof(int) * NUM_SHARED);

    CUDA_CHECK(cudaMallocManaged(&finished, sizeof(int)));
    memset((void *) finished, START, sizeof(int));

    // Writing all the changes of UM to GPU
    __sync_synchronize();
  }

  ~Consistency() {
    CUDA_CHECK(cudaFree((int *) arr));
    CUDA_CHECK(cudaFree((int *) finished));
  }
  
private:	// Logic
  bool is_arr_full() {
    int count = 0;
    for (int i = 0; i < NUM_SHARED; i++) {
      count += arr[i];
    }
    return count == NUM_SHARED;
  }

  bool check_consistency(UVMSPACE long *arr) {
    // Read shared memory page - sequentially
    for (int i = 0; i < NUM_SHARED - 1; i += 2) {
      long value = arr[i];	// Will be [00000000;00000001] if arr[i] == 0 and arr[i + 1] == 1
			if (value == V) {
			::std::cout << "arr[i] = " << ((int *) &value)[0] << "arr[i + 1] = " << ((int *) &value)[1] << ::std::endl;

      // if (value == V) {  // arr[i] == 0 and arr[i + 1] == 1  ==> Inconsistency
        return true;
      }
    }
    return false;
  }
  
  void launch_task() {
    // Start GPU task
    GPU_UVM_Writer_Kernel<<<1,1>>>(arr, finished);

    // GPU can start
    *finished = GPU_START;
  }

  void check_consistency() {
    // While writes have not finished
    while (!is_arr_full()) {
      // Check if an inconsistency exists in the array
      if (check_consistency((long *) arr)) {
        ::std::cout << "Found Inconsistency !" << ::std::endl;
        return;
      }
    }
    ::std::cout << "No Inconsistency Found" << ::std::endl;
  }

  void finish_task() {
    while (*finished != GPU_FINISH);
    // Task is over
    *finished = FINISH;

    CUDA_CHECK(cudaDeviceSynchronize());
  }
    
public:
  static void start() {
    Consistency consistency;
    // Start kernel
    consistency.launch_task();

    // Check GPU consistency
    consistency.check_consistency();

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
