#include<stdio.h>
#include"scrImagePgmPpmPackage.h"

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



//Kernel which calculate the resized image
__global__ void createResizedImage(unsigned char *imageScaledData, int scaled_width, float scale_factor, cudaTextureObject_t texObj)
{
	const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned index = tidY * scaled_width + tidX;
       	
	// Step 4: Read the texture memory from your texture reference in CUDA Kernel
	// printf("index: %d\n", index);
	imageScaledData[index] = tex2D<unsigned char>(texObj,(float)(tidX * scale_factor),(float)(tidY * scale_factor));
}

int main(int argc, char*argv[])
{
	int height=0, width =0, scaled_height=0,scaled_width=0;
	//Define the scaling ratio	
	float scaling_ratio;//=2.0;
	unsigned char*data;
	unsigned char*scaled_data,*d_scaled_data;

	char inputStr[1024]; //= {"aerosmith-double.pgm"};
	char outputStr[1024];// = {"aerosmith-double-scaled.pgm"};

	if (argc < 5) {
        fprintf(stderr, "expected: <filename> <scaling_ratio> <blockDim.x> <blockDim.y>\n");
        exit(1);
    }

	strcpy(inputStr, argv[1]);
	strcat(inputStr, ".pgm");

	strcpy(outputStr, argv[1]);
	strcat(outputStr, "-scaled.pgm");

	scaling_ratio = atof(argv[2]);
	int block_x = atoi(argv[3]); //32
	int block_y = atoi(argv[4]); //32
	int grid_x, grid_y;

	cudaError_t returnValue;

	//Create a channel Description to be used while linking to the tecture
	cudaArray* cu_array;
	cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, kind);

	get_PgmPpmParams(inputStr, &height, &width);	//getting height and width of the current image
	data = (unsigned char*)malloc(height*width*sizeof(unsigned char));
	// printf("\n Reading image width height and width [%d][%d]", height, width);
	scr_read_pgm( inputStr , data, height, width );//loading an image to "inputimage"

	scaled_height = (int)(height*scaling_ratio);
	scaled_width = (int)(width*scaling_ratio);
	scaled_data = (unsigned char*)malloc(scaled_height*scaled_width*sizeof(unsigned char));
	// printf("\n scaled image width height and width [%d][%d]", scaled_height, scaled_width);

	//Allocate CUDA Array
 	returnValue = cudaMallocArray( &cu_array, &channelDesc, width, height);
  if(returnValue != cudaSuccess)
  	printf("\n%s", cudaGetErrorString(returnValue));
	// cudaMemcpyToArray()
	returnValue = (cudaError_t)(returnValue | cudaMemcpyToArray( cu_array, 0, 0, data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice));
	// printf("\n%s", cudaGetErrorString(returnValue));

	if(returnValue != cudaSuccess)
		printf("\n Got error while running CUDA API Array Copy");

	// Step 1. Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cu_array;

	// Step 2. Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Step 3: Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	if(returnValue != cudaSuccess) 
		printf("\n Got error while running CUDA API Bind Texture");
	
	cudaMalloc(&d_scaled_data, scaled_height*scaled_width*sizeof(unsigned char) );

	dim3 dimBlock(block_x, block_y, 1);

	grid_x = scaled_width / dimBlock.x + 1;
	grid_y = scaled_height / dimBlock.y + 1;
	dim3 dimGrid(grid_x, grid_y, 1);
	// printf("\n Launching grid with blocks [%d][%d] \n", dimGrid.x,dimGrid.y);

	GpuTimer *timer = new GpuTimer();
	timer->Start();
	createResizedImage<<<dimGrid, dimBlock>>>(d_scaled_data, scaled_width, 1.0 / scaling_ratio, texObj);
	returnValue = (cudaError_t)(returnValue | cudaDeviceSynchronize());

	returnValue = (cudaError_t)(returnValue |cudaMemcpy (scaled_data , d_scaled_data, scaled_height*scaled_width*sizeof(unsigned char), cudaMemcpyDeviceToHost ));
	if(returnValue != cudaSuccess) 
		printf("\n Got error while running CUDA API kernel");

	timer->Stop();
	printf("%s,%d,%d,%d,%d,%d,%d,%f,%d,%d,%f\n",argv[1],block_x, block_y, dimGrid.x, dimGrid.y, height, width, scaling_ratio, scaled_height, scaled_width, timer->Elapsed());

	// Step 5: Destroy texture object
	cudaDestroyTextureObject(texObj);
	
	scr_write_pgm( outputStr, scaled_data, scaled_height, scaled_width, "####" ); //storing the image with the detections
		
	if(data != NULL)
		free(data);
	if(cu_array !=NULL)
		cudaFreeArray(cu_array);
	if(scaled_data != NULL)
		free(scaled_data);
	if(d_scaled_data!=NULL)
		cudaFree(d_scaled_data);
	
	delete(timer);
	return 0;
}
