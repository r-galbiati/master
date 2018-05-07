// GPU IMPLEMENTATION OF VEGAS ALGORITHM FOR ADAPTIVE MULTIDIMENSIONAL INTEGRATION by Riccardo Galbiati
// July 20th, 2017
// For information GP Lepage, J. Comput. Phys. 27 (1978) 192.


#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

// Needed for cuRAND library
#include <curand.h>
#include <curand_kernel.h>

// Parameters of the configuration
#define D 3                       // Integral dimension
#define N 64                      // Number of intervals throughout the single dimension
#define NCELL int(pow(N,D))       // Total number of subvolumes
#define K 250                     // Number of points in the single subvolume for MC integration
#define M K*NCELL                 // Number of total points in the whole volume. In each subvolume K=M/NCELL.
#define KGRID 100                 // Number of cycles used to refine the grid
#define ALPHA 0.2                 // Rate for the convergence of the grid refining (typically between 0.2 and 2)
#define PERC 1./(N)               // Needed for the grid refining
#define COEFF 20.                 // Paramter of our oscillating (trial) function
#define PI 3.14159265358979323846 // Approximation for PI in C

// Name of the different dimensions
#define X 0
#define Y 1
#define Z 2

// Parameters used for the launch of the different kernels
// Notice that these parameters need to be adjusted according to your system; try different values for different speedups
const int threadsPerBlock = 64;                      // try different values to find the optimal configuration
const int blocksPerGrid = NCELL/threadsPerBlock;     // typically we want this value to be either 1024 or 2048


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// STRUCTURES

// Structure representing a single coordinate in a D-dimensional space
struct coordinates {
  float a[D];
};


// Function to be integrated
struct function {
  // Method to evaluate the function in a single point
  // It is both __host__ and __device__ because it will get called both on the CPU and GPU
  __host__ __device__ float getValue (float x,float y,float z) { return (PI*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z));}
};

// Probability distribution function. We will use an array of these values.
// The two values multiplied must give 1./N
// Notice that in D dimensions we have D first elements and D second elements (a width and a pdf value for each dimension)
struct pdfStep {

  float width[D];                                 // Step width
  float value[D];                                 // Value of p in that step
  float offset[D];                                // Offset for that step

  // Method to get the probability of a step
  __host__ __device__  float getStepArea () {
    float tempArea=1.;
    for(int i=0;i<D;i++) {
      tempArea*=width[i]*value[i];
    }
    return (tempArea);}

  // Method to get the pdf value in that subvolume
  __host__ __device__ float getProb () {
    float temp=1.;
    for(int i=0;i<D;i++) {
      temp*=value[i];
    }
    return (temp);}
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FUNCTIONS

// This GPU kernel is used to initialize the states (that will later be passed to curand as an argument)
__global__ void init(unsigned int seed, curandState_t* states) {

  int tid= threadIdx.x + blockIdx.x * blockDim.x;

  // Initialize the needed states (one for each thread)
  curand_init(seed,  // The seed can be the same for each core; we pass the time so that we get different values every time
	      tid,         // The sequence number needs to be different for each thread unless we want them all to generate the same sequence
	      0,           // The offset represents how much we progress after each call of the sequence; it can be 0
	      &states[tid]);
}

// Function to initialize a pdf
void InitializePdf (pdfStep *temp) {
  for(int i=0; i<NCELL;i++) {
    for(int j=0; j<D;j++) {
      temp[i].width[j]=1./N;
      temp[i].value[j]=1./(temp[i].width[j]*N);
      temp[i].offset[j]=0;
    }
  }
};

// Function to apply an MC integration on each subvolume
// Notice that tempIntegral has a number of elements which is equal to the number of blocks launched (it is used in the context of shared memory)
__global__ void Montecarlo (curandState_t* states, pdfStep* temp, function* tempExample, float* tempIntegral, float* tempIntegral2) {

  // Preliminaries
  __shared__ float cache[threadsPerBlock];                               // Shared memory buffer
  __shared__ float cache2[threadsPerBlock];                              // Shared memory buffer
  int tid = threadIdx.x + blockIdx.x * blockDim.x;                       // Shortcut for the index thread
  int cacheIndex = threadIdx.x;                                          // Shortcut for the buffer index (we only consider the thread independently of the block)

  float tempFloat = 0;                                                   // Each thread has their own temp variable for the summatory
  float tempFloat2 = 0;                                                  // Each thread has their own temp variable for the summatory
  coordinates numbers;

  for(int k=0; k<K;k++) {                                                // We do it K times

    for(int jj=0;jj<D;jj++) {
      numbers.a[jj]=temp[tid].offset[jj]+curand_uniform(&states[tid])*temp[tid].width[jj];        // Adjust the offsets in each dimension
    }

    tempFloat+=tempExample->getValue(numbers.a[X],numbers.a[Y],numbers.a[Z])/temp[tid].getProb();                    // Sum all the contributes
    tempFloat2+=pow(tempExample->getValue(numbers.a[X],numbers.a[Y],numbers.a[Z])/temp[tid].getProb(),2);            // Sum all the squared contributes
  }

  cache[cacheIndex] = tempFloat;                                                   // Assign the value to the shared memory buffer
  cache2[cacheIndex] = tempFloat2;                                                 // Assign the value to the shared memory buffer

  // Sync the threads we are using
  __syncthreads();

  // We realize a reduction, so that we have a single value for each block
  int i = blockDim.x/2;

  while (i != 0) {

    if (cacheIndex < i) {
      cache[cacheIndex]  += cache [cacheIndex + i];
      cache2[cacheIndex] += cache2[cacheIndex + i];
    }

    __syncthreads();
    i /= 2;
  }

  // The 0th thread writes its value in the position of tempIntegral corresponding to the block the index is referred to
  if (cacheIndex == 0){
    tempIntegral[blockIdx.x] = cache[0];
    tempIntegral2[blockIdx.x] = cache2[0];
  }

};

//---------------------------------------------------------------------------------------------------------------------------------------------------------
// Adjust X


// Method which calculates the weights according to which we will adjust the cells in X
// Notice: this time the function call has N as number of blocks and threadsPerBlock as number of threads (for MC it was NCELL)
// Notice the differences between importance and stratified sampling
__global__ void WeightsX (curandState_t* states, function *tempFunction, pdfStep *temp, float *tempColumn, float *tempColumn2) {

  // Preliminaries
  __shared__ float cache[threadsPerBlock];                               // Shared memory buffer
  __shared__ float cache2[threadsPerBlock];                              // Shared memory buffer
  int tid = threadIdx.x + blockIdx.x * blockDim.x;                       // Shortcut for the thread index
  int cacheIndex = threadIdx.x;                                          // Shortcut for the buffer index

  // Temp variables
  float offsetx=0;
  float xvalue=0;

  xvalue=curand_uniform(&states[tid])*temp[blockIdx.x].width[X];             // Random value adjusted according to the cell width
  xvalue+=offsetx;                                                           // Adjusted value with the correct offset
  offsetx+=temp[blockIdx.x].width[X];

  // Each thread in each block writes this value in the cache memory
  cache[cacheIndex] = tempFunction->getValue(xvalue,curand_uniform(&states[tid]),curand_uniform(&states[tid]))/temp[tid].getProb();
  cache2[cacheIndex] = pow(tempFunction->getValue(xvalue,curand_uniform(&states[tid]),curand_uniform(&states[tid]))/temp[tid].getProb(),2);

  // Sync the threads
  __syncthreads();

  // We do a reduction, as in the MC kernel
  // We are left with a single value for each block
  int i = blockDim.x/2;

  while (i != 0) {

    if (cacheIndex < i){
      cache[cacheIndex] += cache[cacheIndex + i];
      cache2[cacheIndex] += cache2[cacheIndex + i];
    }

    __syncthreads();
    i /= 2;
  }

  // The 0th thread writes its value in the tempColumn position corresponding to the block the index is referred to
  if (cacheIndex == 0){
    tempColumn[blockIdx.x] = cache[0];
    tempColumn2[blockIdx.x] = cache2[0];
  }

};

// Method to adjust the cells in X
// Notice that we are working on the CPU here
void AdjustX (pdfStep *temp, float *tempColumn, float *tempColumn2, float &tempColumnTotal, float &tempDeltaTot) {
  // DeltaTot and ColumnTotal need to be equal to 0
  tempDeltaTot=0;
  tempColumnTotal=0;

  // NOTICE THE DIFFERENCE BETWEEN IMPORTANCE AND STRATIFIED SAMPLING
  // Calculate the variance
  for(int i=0; i<N; i++) {
    tempColumn[i]=((1./(M)*tempColumn2[i]) - pow((1./(M)*tempColumn[i]),2))/(M-1);
    tempColumnTotal+= tempColumn[i];
  }

  // Adjust everything according to the rule.
  for(int i=0;i<N;i++) {

    tempColumn[i]=tempColumn[i]/tempColumnTotal;                                       // Consider the percentages on the total of tempColumn

    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[p*N*N+j*N+i].width[X] = temp[p*N*N+j*N+i].width[X]*PERC/tempColumn[i];
	temp[p*N*N+j*N+i].width[X] = pow(log(temp[p*N*N+j*N+i].width[X]+1),ALPHA);
      }
    }

    tempDeltaTot+=temp[0*N*N+0*N+i].width[X];
  }

  // Normalize the cells and adjust the pdfs
  float tempOffset=0;

  for(int i=0;i<N;i++) {
    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[p*N*N+j*N+i].width[X] = temp[p*N*N+j*N+i].width[X]/tempDeltaTot;                         // Renormalize the cells width
	temp[p*N*N+j*N+i].value[X] = 1./(temp[p*N*N+j*N+i].width[X]*N);                               // Adjust the probabilites
      }
    }
  }

  // Adjust the offset for each cell
  for(int i=0;i<N;i++) {
    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[p*N*N+j*N+i].offset[X] = tempOffset;                                                     // Adjust the offset on X
      }
    }

    tempOffset+=temp[0*N*N+0*N+i].width[X];                                                     // Offset for all of them

  }

};


//---------------------------------------------------------------------------------------------------------------------------------------------------------
// Adjust Y
// EVERYTHING (BUT THE INDEXES) IS THE SAME AS FOR THE X

__global__ void WeightsY (curandState_t* states, function *tempFunction, pdfStep *temp, float *tempColumn, float *tempColumn2) {

  // Preliminaries
  __shared__ float cache[threadsPerBlock];
  __shared__ float cache2[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  // Temp variables
  float offsety=0;
  float yvalue=0;

  yvalue=curand_uniform(&states[tid])*temp[blockIdx.x].width[Y];
  yvalue+=offsety;
  offsety+=temp[blockIdx.x].width[Y];


  cache[cacheIndex] = tempFunction->getValue(curand_uniform(&states[tid]),yvalue,curand_uniform(&states[tid]))/temp[tid].getProb();
  cache2[cacheIndex] = pow(tempFunction->getValue(curand_uniform(&states[tid]),yvalue,curand_uniform(&states[tid]))/temp[tid].getProb(),2);

  // Sync the threads
  __syncthreads();


  // Reduction
  int i = blockDim.x/2;

  while (i != 0) {

    if (cacheIndex < i){
      cache[cacheIndex] += cache[cacheIndex + i];
      cache2[cacheIndex] += cache2[cacheIndex + i];
    }

    __syncthreads();
    i /= 2;
  }


  if (cacheIndex == 0){
    tempColumn[blockIdx.x] = cache[0];
    tempColumn2[blockIdx.x] = cache2[0];
  }
};

// This works on the CPU
void AdjustY (pdfStep *temp, float *tempColumn,float *tempColumn2, float &tempColumnTotal, float &tempDeltaTot) {

  tempDeltaTot=0;
  tempColumnTotal=0;

  // Calculate the variance
  for(int i=0; i<N; i++) {
    tempColumn[i]=((1./(M)*tempColumn2[i]) - pow((1./(M)*tempColumn[i]),2))/(M-1);
    tempColumnTotal+= tempColumn[i];
  }

  // Adjust according to the rule
  for(int i=0;i<N;i++) {

    tempColumn[i]=tempColumn[i]/tempColumnTotal;

    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[p*N*N+i*N+j].width[Y] = temp[p*N*N+i*N+j].width[Y]*PERC/tempColumn[i];
	temp[p*N*N+i*N+j].width[Y] = pow(log(temp[p*N*N+i*N+j].width[Y]+1),ALPHA);
      }
    }

    tempDeltaTot+=temp[0*N*N+i*N+0].width[Y];
  }

  // Normalize
  float tempOffset=0;

  for(int i=0;i<N;i++) {
    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[p*N*N+i*N+j].width[Y] = temp[p*N*N+i*N+j].width[Y]/tempDeltaTot;
	temp[p*N*N+i*N+j].value[Y] = 1./(temp[p*N*N+i*N+j].width[Y]*N);
      }
    }
  }

  // Adjust the offset
  for(int i=0;i<N;i++) {
    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[p*N*N+i*N+j].offset[Y] = tempOffset;
      }
    }

    tempOffset+=temp[0*N*N+i*N+0].width[Y];

  }

};


//---------------------------------------------------------------------------------------------------------------------------------------------------------
// Adjust Z
// EVERYTHING (BUT THE INDEXES) IS THE SAME AS FOR THE X

__global__ void WeightsZ (curandState_t* states, function *tempFunction, pdfStep *temp, float *tempColumn,float *tempColumn2) {

  // Preliminaries
  __shared__ float cache[threadsPerBlock];
  __shared__ float cache2[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  // Temp variables
  float offsetz=0;
  float zvalue=0;

  zvalue=curand_uniform(&states[tid])*temp[blockIdx.x].width[Z];
  zvalue+=offsetz;
  offsetz+=temp[blockIdx.x].width[Z];


  cache[cacheIndex] = tempFunction->getValue(curand_uniform(&states[tid]),curand_uniform(&states[tid]),zvalue)/temp[tid].getProb();
  cache2[cacheIndex] = pow(tempFunction->getValue(curand_uniform(&states[tid]),curand_uniform(&states[tid]),zvalue)/temp[tid].getProb(),2);

  // Sync the threads
  __syncthreads();

  // Reduction
  int i = blockDim.x/2;

  while (i != 0) {

    if (cacheIndex < i){
      cache[cacheIndex] += cache[cacheIndex + i];
      cache2[cacheIndex] += cache2[cacheIndex + i];
    }


    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0){
    tempColumn[blockIdx.x] = cache[0];
    tempColumn2[blockIdx.x] = cache2[0];
  }

};

// This works on the CPU
void AdjustZ (pdfStep *temp, float *tempColumn, float *tempColumn2, float &tempColumnTotal, float &tempDeltaTot) {

  tempDeltaTot=0;
  tempColumnTotal=0;

  // Calculate the variance
  for(int i=0; i<N; i++) {
    tempColumn[i]=((1./(M)*tempColumn2[i]) - pow((1./(M)*tempColumn[i]),2))/(M-1);
    tempColumnTotal+= tempColumn[i];
  }

  // Adjust according to the rule
  for(int i=0;i<N;i++) {

    tempColumn[i]=tempColumn[i]/tempColumnTotal;

    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[i*N*N+p*N+j].width[Z] = temp[i*N*N+p*N+j].width[Z]*PERC/tempColumn[i];
	temp[i*N*N+p*N+j].width[Z] = pow(log(temp[i*N*N+p*N+j].width[Z]+1),ALPHA);
      }
    }

    tempDeltaTot+=temp[i*N*N+0*N+0].width[Z];
  }

  // Normalize
  float tempOffset=0;

  for(int i=0;i<N;i++) {
    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[i*N*N+p*N+j].width[Z] = temp[i*N*N+p*N+j].width[Z]/tempDeltaTot;
	temp[i*N*N+p*N+j].value[Z] = 1./(temp[i*N*N+p*N+j].width[Z]*N);
      }
    }
  }

  // Offsets
  for(int i=0;i<N;i++) {
    for(int j=0;j<N;j++) {
      for(int p=0;p<N;p++) {
	temp[i*N*N+p*N+j].offset[Z] = tempOffset;
      }
    }

    tempOffset+=temp[i*N*N+0*N+0].width[Z];

  }

};



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN

int main() {

// Select the graphic card to use

  // Set the device to use
  cudaSetDevice(2);
  // Check
  int* whichOne;
  whichOne = (int*)malloc( sizeof(int));
  cudaGetDevice(whichOne);
  std::cout<<"The device currently used is "<<(*whichOne)<<std::endl;

  // Allocate all the variables we need
  float integral;                                                     // Variable to store the integral estimation
  float variance;                                                     // Variable to store the variance estimation
  float columnTotal;                                                  // Temp variable to adjust the cells
  float deltaTot;                                                     // Temp variable to adjust the cells

  // Allocate the space on the memory to adjust the columns
  float *column;
  float *dev_column;
  column = (float*)malloc( N*sizeof(float));
  cudaMalloc( (void**)&dev_column, N*sizeof(float));                     // Allocate the memory on the GPU

  // Allocate the space on the memory to adjust the columns
  float *column2;
  float *dev_column2;
  column2 = (float*)malloc( N*sizeof(float));
  cudaMalloc( (void**)&dev_column2, N*sizeof(float));                    // Allocate the memory on the GPU

  // Initialize the function to integrate
  function *example;
  example = (function*)malloc( sizeof(function) );
  // Allocate the space on the GPU for the function
  function *dev_example;
  cudaMalloc((void**) &dev_example, sizeof(function));
  cudaMemcpy( dev_example, example, sizeof(function), cudaMemcpyHostToDevice );             // Copy the example values in dev_example


  // Allocate the memory on the CPU and then initialize the pdf
  pdfStep *pdf;
  pdf = (pdfStep*)malloc( NCELL*sizeof(pdfStep) );
  InitializePdf(pdf);
  // Allocate the space on the GPU for the pdf
  pdfStep *dev_pdf;
  cudaMalloc((void**) &dev_pdf, NCELL * sizeof(pdfStep));
  cudaMemcpy( dev_pdf, pdf, NCELL*sizeof(pdfStep), cudaMemcpyHostToDevice );                // Copy the pdf values in dev_pdf


  // curandState_t is a parameter to pass each time we use curand; we use a random value for each thread
  curandState_t* states;
  // Allocate the space for the random states on the GPU
  cudaMalloc((void**) &states, N * sizeof(curandState_t));


  // Measure the GPU performance
  // We start to measure the time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );


  // Initialize all these states
  init<<<blocksPerGrid,threadsPerBlock>>>(time(0), states);


  // Cycle to adjust the grid
  for(int jtemp=0;jtemp<KGRID;jtemp++) {

//---------------------------------------------------------------------------------------------------------------------------------------------------------
    // Adjust X

    WeightsX<<<N,threadsPerBlock>>>(states, dev_example, dev_pdf, dev_column,dev_column2);

    // Copy the array column from the GPU to the CPU
    cudaMemcpy( column, dev_column, N*sizeof(float), cudaMemcpyDeviceToHost );

    // Copy the array column from the GPU to the CPU
    cudaMemcpy( column2, dev_column2, N*sizeof(float), cudaMemcpyDeviceToHost );

    // Complete the summatory on the CPU
    columnTotal = 0;
    for (int i=0; i<N; i++) {
      columnTotal += column[i];
    }

    // Adjust the subvolumes on the CPU
    AdjustX (pdf, column, column2, columnTotal, deltaTot);

    // Copy the pdf values in dev_pdf
    cudaMemcpy( dev_pdf, pdf, NCELL*sizeof(pdfStep), cudaMemcpyHostToDevice );

//---------------------------------------------------------------------------------------------------------------------------------------------------------
    // Adjust Y
    // EVERYTHING (BUT THE INDEXES) IS THE SAME AS FOR THE X

    WeightsY<<<N,threadsPerBlock>>>(states, dev_example, dev_pdf, dev_column,dev_column2);

    cudaMemcpy( column, dev_column, N*sizeof(float), cudaMemcpyDeviceToHost );

    cudaMemcpy( column2, dev_column2, N*sizeof(float), cudaMemcpyDeviceToHost );

    columnTotal = 0;
    for (int i=0; i<N; i++) {
      columnTotal += column[i];
    }

    AdjustY (pdf, column, column2, columnTotal, deltaTot);

    cudaMemcpy( dev_pdf, pdf, NCELL*sizeof(pdfStep), cudaMemcpyHostToDevice );

//---------------------------------------------------------------------------------------------------------------------------------------------------------
    // Adjust Z
    // EVERYTHING (BUT THE INDEXES) IS THE SAME AS FOR THE X

    WeightsZ<<<N,threadsPerBlock>>>(states, dev_example, dev_pdf, dev_column,dev_column2);

    cudaMemcpy( column, dev_column, N*sizeof(float), cudaMemcpyDeviceToHost );

    cudaMemcpy( column2, dev_column2, N*sizeof(float), cudaMemcpyDeviceToHost );

    columnTotal = 0;
    for (int i=0; i<N; i++) {
      columnTotal += column[i];
    }

    AdjustZ (pdf, column, column2, columnTotal, deltaTot);

    cudaMemcpy( dev_pdf, pdf, NCELL*sizeof(pdfStep), cudaMemcpyHostToDevice );
//---------------------------------------------------------------------------------------------------------------------------------------------------------
  }


  // Allocate the space on the memory for the result of the partial summatories of each block
  float *partial_c;
  float *dev_partial_c;
  partial_c = (float*)malloc( blocksPerGrid*sizeof(float));
  // Allocate the memory on the GPU
  cudaMalloc( (void**)&dev_partial_c,blocksPerGrid*sizeof(float)) ;

  // Allocate the space on the memory for the result of the partial summatories(2) of each block
  float *partial_c2;
  float *dev_partial_c2;
  partial_c2 = (float*)malloc( blocksPerGrid*sizeof(float));
  // Allocate the memory on the GPU
  cudaMalloc( (void**)&dev_partial_c2,blocksPerGrid*sizeof(float)) ;


  // The integration itself
  Montecarlo<<<blocksPerGrid,threadsPerBlock>>>(states, dev_pdf, dev_example, dev_partial_c,dev_partial_c2);


  // Copy the C array from the GPU onto the CPU
  cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost );
  // Copy the C2 array from the GPU onto the CPU
  cudaMemcpy( partial_c2, dev_partial_c2, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost );

  // Complete the sum on the CPU
  integral = 0;
  variance = 0;
  for (int i=0; i<blocksPerGrid; i++) {
    integral += partial_c[i];
    variance += partial_c2[i];
  }


  // End of the GPU performance measure
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop );
  printf( "Time used by the GPU: %3.1f ms\n", elapsedTime );


  // The variable integral gets the integral value
  integral = 1./(M)*integral;
  // Print to verify the value
  printf("The estimated integral value is %f\n", integral);

  // The variable variance gets the variance value
  variance = 1./(M)*variance;
  variance = (variance - integral*integral)*1./(M);
  // Print to verify the value
  printf("The estimated variance value is %e\n", variance);
  printf("The value of M is %i\n", M);


  // Free the GPU memory
  cudaFree( states );
  cudaFree( dev_pdf );
  cudaFree( dev_example );
  cudaFree( dev_partial_c );
  cudaFree( dev_partial_c2 );
  cudaFree( dev_column );
  cudaFree( dev_column2 );

  // Free the CPU memory
  free( pdf );
  free( partial_c );
  free( partial_c2 );
  free(example);
  free(column);
  free(column2);
  free(whichOne);

  // Free the memory
  cudaEventDestroy( start );
  cudaEventDestroy( stop );


  return 0;
}
