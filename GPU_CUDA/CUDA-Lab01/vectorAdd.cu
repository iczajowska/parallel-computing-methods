%%writefile vectorAdd.cu
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
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


__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int same_str(const char* str1, const char* str2) {
    return strcmp(str1, str2) == 0;
}

/**
 * Host main routine
 */
int main(int argc, char** argv)
{
    int isGPU;

    if (argc < 4) {
        fprintf(stderr, "expected: <GPU|CPU> <array size> <thread per block>\n");
        exit(1);
    }

    if (same_str(argv[1], "GPU")) {
        isGPU = 1;
    } else if(same_str(argv[1], "CPU")){
        isGPU = 0;
    } else {
        fprintf(stderr, "Incorrect first argument <GPU|CPU>\n");
        exit(1);
    }



    // Print the vector length to be used, and compute its size
    int numElements = atoi(argv[2]); //50000;
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }


    if(isGPU){
        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

        // Allocate the device input vector A
        float *d_A = NULL;
        err = cudaMalloc((void **)&d_A, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Allocate the device input vector B
        float *d_B = NULL;
        err = cudaMalloc((void **)&d_B, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Allocate the device output vector C
        float *d_C = NULL;
        err = cudaMalloc((void **)&d_C, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the host input vectors A and B in host memory to the device input vectors in
        // device memory
        err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Launch the Vector Add CUDA Kernel
        int threadsPerBlock = atoi(argv[3]);//256;
        int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;


        GpuTimer *timer = new GpuTimer();
        timer->Start();
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        cudaDeviceSynchronize();
        timer->Stop();
        printf("GPU,%d,%f,%d,%d", numElements,timer->Elapsed(),blocksPerGrid, threadsPerBlock);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
        err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        // Free device global memory
        err = cudaFree(d_A);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaFree(d_B);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaFree(d_C);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Reset the device and exit
        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        delete(timer);

    } else {
        clock_t start,end;
        start = clock();
        for(int i=0; i<numElements; i++){
            h_C[i]=h_A[i]+h_B[i];
        }
        end = clock();
        double dif = (double)(end - start)*1000.0 / CLOCKS_PER_SEC;
        printf("CPU,%d,%f,-1,-1", numElements, dif);
    }
 

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf(",1\n");
 

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
