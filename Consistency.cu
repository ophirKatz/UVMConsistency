
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


namespace UVMConsistency {


#define START         0
#define GPU_START     1
#define GPU_FINISH    2
#define FINISH        3

#define NUM_SHARED 10

typedef unsigned long long int ulli;


class BitManipulation {
public:
  __device__ static void set_bit(UVMSPACE ulli *mask, int index) {
    *mask = *mask | (1UL << index);
  }

  static int count_set_bits(ulli number) {
    return std::bitset<sizeof(ulli)>(number).count();
  }

  static ulli get_difference(ulli num1, ulli num2) {
    return num1 ^ num2;
  }

  static int get_first_set_bit_index(ulli number) {
    return __builtin_clz(number);
  }
};

class SharedUnit {
private:
  static void *allocate(size_t size) {
    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    CUDA_CHECK(cudaDeviceSynchronize());
    return ptr;
  }

public:

  static void initialize_unit(UVMSPACE SharedUnit *unit, int index, int value) {
    unit->index = index;
    unit->value = value;
    __sync_synchronize();
  }

  SharedUnit() : index(0), value(0) {}

  // Memory management
  void *operator new(size_t size) {
    return allocate(size);
  }

  void *operator new[](size_t size) {
    return allocate(size);
  }

  void operator delete[](void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }

  // Properties
  UVMSPACE int index;
  UVMSPACE int value;
};

__device__ void increment_unit(UVMSPACE SharedUnit *unit, UVMSPACE ulli *mask) {
  atomicAdd((ulli *) &unit->value, (ulli) 1);
  BitManipulation::set_bit(mask, unit->index);
}

__global__ void UVM_increment(UVMSPACE SharedUnit *shared_units, UVMSPACE ulli *mask, UVMSPACE int *finished) {
  // Wait for CPU
  while (*finished != GPU_START);
  
  for (int i = 0; i < NUM_SHARED; i++) {
    UVMSPACE SharedUnit *unit = &shared_units[i];
    increment_unit(unit, mask);
  }
  
  // GPU finished - CPU can finish
  *finished = GPU_FINISH;

  // Wait for CPU to finish
  while (*finished != FINISH);
}

class Consistency {
private:

  Consistency() {
    shared_units = new SharedUnit[NUM_SHARED];
    for (int i = 0; i < NUM_SHARED; i++) {
      const int index = i;
      const int value = 0;
      SharedUnit::initialize_unit(shared_units + i, index, value);
    }

    CUDA_CHECK(cudaMallocManaged(&finished, sizeof(int)));
    memset((void *) finished, START, sizeof(int));

    CUDA_CHECK(cudaMallocManaged(&mask, sizeof(ulli)));
    memset((void *) mask, 0, sizeof(ulli));

    // Writing all the changes of UM to GPU
    __sync_synchronize();
  }

  ~Consistency() {
    delete[] shared_units;
    CUDA_CHECK(cudaFree((int *) finished));
    CUDA_CHECK(cudaFree((ulli *) finished));
  }

  
  // Logic
  // CPU Logic

  static int get_new_unit_changed(UVMSPACE ulli *mask, ulli compared_mask) {
    return BitManipulation::get_first_set_bit_index(
      BitManipulation::get_difference(*mask, compared_mask)
    );
  }

  void launch_task() {
    // Start GPU task
    UVM_increment<<<1,1>>>(shared_units, mask, finished);
  }

  void check_consistency() {
    ulli compared_mask = *mask;
    int last_unit_index = -1;

    // GPU can start
    *finished = GPU_START;

    while (*finished != GPU_FINISH) {
      while (*mask == compared_mask);
      int new_unit_index = Consistency::get_new_unit_changed(mask, compared_mask);
      assert (this->shared_units[new_unit_index].value != 0);
      if (last_unit_index + 1 != new_unit_index) {
        ::std::cout <<  "Error : Last unit was : "  << last_unit_index << 
                        " and current unit is : "   << new_unit_index << ::std::endl;
      }
      last_unit_index = new_unit_index;
    }
  }

  void finish_task() {
    // Task is over
    *finished = FINISH;
  }
  
public:
  static void start() {
    Consistency consistency;
    // Start kernel
    ::std::cout << "Launching kernel" << ::std::endl;
    consistency.launch_task();

    // Check GPU consistency
    ::std::cout << "Start CPU loop" << ::std::endl;
    consistency.check_consistency();

    // Finish task for CPU and GPU
    ::std::cout << "Finish task" << ::std::endl;
    consistency.finish_task();
  }
private:

  UVMSPACE SharedUnit *shared_units;
  UVMSPACE ulli *mask;
  volatile int *finished;
};

};  // namespace UVMConsistency


int main() {
  UVMConsistency::Consistency::start();
  return 0;
}

