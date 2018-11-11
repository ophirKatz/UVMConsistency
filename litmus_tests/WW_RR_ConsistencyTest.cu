#include <stdio.h>


#define GPU_START	1
#define GPU_FINISHED	2
#define FINISH	3

__global__ void kernel(volatile int *x, volatile int *y, volatile int *finished) {
	while (*finished != GPU_START);

	*x = 1;
	*y = 1;

	*finished = GPU_FINISHED;

	while (*finished != FINISH);
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
	while (p + q == 0) {
		p = *y;
		q = *x;
	}

	while (*finished != GPU_FINISHED);
	*finished = FINISH;

	if (p == 1 && q == 0) {
		printf("Success\n");
	} else {
		printf("Failure\n");
	}

	cudaDeviceSynchronize();

	cudaFree((int *) x);
	cudaFree((int *) y);
	cudaFree((int *) finished);

	return 0;
}


