#include <stdio.h>


#define GPU_START			1
#define GPU_FINISHED	2
#define FINISH				3

#define NUM_OF_TESTS	100000

__global__ void kernel(volatile int *x, volatile int *y, volatile int *finished) {
	while (*finished != GPU_START);

	for (int i = 1; i <= NUM_OF_TESTS; i++) {
		*x = i;
		__threadfence_system();
		*y = i;
		__threadfence_system();
	}

	*finished = GPU_FINISHED;
}

int main() {
	volatile int *x, *y, *finished;
	cudaMallocManaged((void **) &x, sizeof(int));
	cudaMallocManaged((void **) &y, sizeof(int));
	cudaMallocManaged((void **) &finished, sizeof(int));

	memset((void *) x, 0, sizeof(int));
	memset((void *) y, 0, sizeof(int));
	memset((void *) finished, 0, sizeof(int));

	kernel<<<1,1>>>(x, y, finished);
	*finished = GPU_START;

	int p = 0, q = 0;
	for (int i = 1; i <= NUM_OF_TESTS; i++) {
		while ((p + q) == 0) {
			p = *y;
			q = *x;
		}

		// Perform test
		if ((p == i) && (q == i - 1)) {
			printf("Success\n");
			return 0;
		}
		// Reset for next test
		p = 0;
		q = 0;
	}
	printf("Failure\n");
		
	while (*finished != GPU_FINISHED);

	cudaDeviceSynchronize();

	cudaFree((int *) x);
	cudaFree((int *) y);
	cudaFree((int *) finished);

	return 0;
}


