#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <cstdlib>

// #define N 1024
#define BLOCK_SIZE 1

class GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;

      public:
 
      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }
 
      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }
 
      void Start()
      {
            cudaEventRecord(start, 0);
      }
 
      void Stop()
      {
            cudaEventRecord(stop, 0);
      }
 
      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};

__global__ void matrix_transpose_naive(int *input, int *output, int N) {

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	int index = indexY * N + indexX;
	int transposedIndex = indexX * N + indexY;

	// this has discoalesced global memory store  
	output[transposedIndex] = input[index];

	// this has discoalesced global memore load
	// output[index] = input[transposedIndex];
}

__global__ void matrix_transpose_shared(int *input, int *output, int N) {

	__shared__ int sharedMemory [BLOCK_SIZE] [BLOCK_SIZE + 1];

	// global index	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	// transposed global memory index
	int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
	int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

	// local index
	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;

	int index = indexY * N + indexX;
	int transposedIndex = tindexY * N + tindexX;

	// reading from global memory in coalesed manner and performing tanspose in shared memory
	sharedMemory[localIndexX][localIndexY] = input[index];

	__syncthreads();

	// writing into global memory in coalesed fashion via transposed data in shared memory
	output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}

//basically just fills the array with index.
void fill_array(int *data, int N) {
	for(int idx=0;idx<(N*N);idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b, int N) {
	printf("\n Original Matrix::\n");
	for(int idx=0;idx<(N*N);idx++) {
		if(idx%N == 0)
			printf("\n");
		printf(" %d ",  a[idx]);
	}
	printf("\n Transposed Matrix::\n");
	for(int idx=0;idx<(N*N);idx++) {
		if(idx%N == 0)
			printf("\n");
		printf(" %d ",  b[idx]);
	}
}

int same_str(const char* str1, const char* str2) {
    return strcmp(str1, str2) == 0;
}

int main(int argc, char** argv) {
	int *a, *b;
        int *d_a, *d_b; // device copies of a, b, c

	if (argc < 3) {
        fprintf(stderr, "expected: <shared|naive> <array size>\n");
        exit(1);
    }

	int is_shared;
	if (same_str(argv[1], "shared")) {
        is_shared = 1;
    } else if(same_str(argv[1], "naive")){
        is_shared = 0;
    } else {
        fprintf(stderr, "Incorrect first argument <shared|naive>\n");
        exit(1);
    }

	int N = atoi(argv[2]);

	int size = N * N *sizeof(int);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); fill_array(a, N);
	b = (int *)malloc(size);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 gridSize(N/BLOCK_SIZE,N/BLOCK_SIZE,1);

	GpuTimer *timer = new GpuTimer();
    timer->Start();
	
	if(is_shared){
		matrix_transpose_shared<<<gridSize,blockSize>>>(d_a,d_b,N);
	} else{
		matrix_transpose_naive<<<gridSize,blockSize>>>(d_a,d_b,N);
	}
	

	// Copy result back to host
	// cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
	// print_output(a,b);

	cudaDeviceSynchronize();
    timer->Stop();
	if(is_shared){
		printf("shared,%d,%f,%d,%d\n", N, timer->Elapsed(),  N/BLOCK_SIZE, BLOCK_SIZE);
	} else {
		printf("naive,%d,%f,%d,%d\n", N, timer->Elapsed(), N/BLOCK_SIZE, BLOCK_SIZE);
	}
	// Copy result back to host
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
	//print_output(a,b,N);

	// terminate memories
	free(a); 
	free(b);
    cudaFree(d_a); 
	cudaFree(d_b); 

	return 0;
}
