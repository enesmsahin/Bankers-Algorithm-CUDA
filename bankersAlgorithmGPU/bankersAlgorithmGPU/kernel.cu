
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <iostream>
#include <random>
#include <cmath>

#define CHECK(status)									            \
{														                    \
	if (status != cudaSuccess)							                    \
	{													                    \
		std::cout << "Cuda error: " << cudaGetErrorString(status) << "\n";  \
	}													                    \
}

//QueryPerformanceCounter related stuff
#define NOMINMAX // To prevent interfering of std::min and min function defined by windows
#include <windows.h>

double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
    LARGE_INTEGER li;
    if (!QueryPerformanceFrequency(&li))
        std::cout << "QueryPerformanceFrequency failed!\n";

    PCFreq = double(li.QuadPart) / 1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart - CounterStart) / PCFreq;
}


// GPU kernels

// Checks whether vecA(i) > vecB(i) for at least one i, where vecA and vecB are [1 x vecSize] vectors.
__global__ void checkIfGreater(int* vecA, int* vecB, int vecSize, bool* isGreaterThan)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vecSize)
    {
        if (vecA[tid] > vecB[tid])
        {
            *isGreaterThan = true;
        }
    }
}

// Element-wise subtraction of two vectors. Subtracts vecA from vecB and stores the result in vecA
__global__ void subtractVectors(int* vecA, int* vecB, int vecSize)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vecSize)
    {
        vecA[tid] = vecA[tid] - vecB[tid];
    }
}

// Element-wise addition of two vectors. Adds vecA to vecB and stores the result in vecA
__global__ void addVectors(int* vecA, int* vecB, int vecSize)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vecSize)
    {
        vecA[tid] = vecA[tid] + vecB[tid];
    }
}

// ATOMICALLY Element-wise addition of two vectors. Adds vecA to vecB and stores the result in vecA
__global__ void addVectorsAtomic(int* vecA, int* vecB, int vecSize)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vecSize)
    {
        atomicAdd(&vecA[tid], vecB[tid]);
    }
}


__global__ void safetyState(int* work, int* need, int* allocation, bool* finish, int numResources, int numProcesses, bool* isSafe)
{
    unsigned int tid = threadIdx.x; // Process ID in Banker's Algorithm.
    
    /* workUpdatedFlag is used in order to indicate if a change is made to the work vector or not, i.e. if at least one process finished during that iteration.
    This flag is shared by al processes. If one process is finished during that iteration, corresponding thread sets this flag to true.
    This flag is reset to false at each iteration by thread 0.
    This flag will also be used for determining whether the state is safe or not.See below for explanation. */
    __shared__ bool workUpdatedFlag; 

    bool isFinished = finish[tid]; // Copy finish vector to the registers

    dim3 grid_d, block_d; // grid and block for dynamic parallelism
    
    block_d.x = (unsigned int)fminf(numResources, 1024);
    grid_d.x = static_cast<unsigned int>(std::ceil(static_cast<double>(numResources) / block_d.x));

    bool* isNeedGreaterThanWork; // Flag to be set if need(i) <= work is not satisfied
    cudaMalloc(&isNeedGreaterThanWork, sizeof(bool));

    if (tid == 0)
    {
        workUpdatedFlag = true;
    }

    while (isFinished == false)
    {
        __syncthreads();

        /* If at least one change is made to the work vector by any of the processes (or if it is the first iteration),
        check need(i) <= work(i) for all unfinished processes. If it is false, no process was able to finis during that iteration,
        so safety check terminates */
        if (workUpdatedFlag == true)
        {
            __syncthreads();
            if (tid == 0)
            {
                workUpdatedFlag = false; // Reset work vector's update flag
            }

            *isNeedGreaterThanWork = false;
            // Check if need(i) <= work(i) for all i. If not, isNeedGreaterThanWork becomes true
            checkIfGreater << <grid_d, block_d >> > (&need[tid * numResources], work, numResources, isNeedGreaterThanWork);
            cudaDeviceSynchronize(); // Wait for child kernel to finish

            if (*isNeedGreaterThanWork == false) // need(i) < work is also satisfied, this process can finish given the current states of matrices
            {
                addVectorsAtomic << <grid_d, block_d >> > (work, &allocation[tid * numResources], numResources); // Move resources allocated to this process to the work matrix
                isFinished = true; // Set finish flag of this process to true

                workUpdatedFlag = true; // A change has been made to the work vector, update flag

                cudaDeviceSynchronize(); // Wait for child kernel to finish
            }
        }
        else
        {
            break;
        }
        __syncthreads();
    }
    __syncthreads();


    /* SCENARIOS FOR STATE OF THE workUpdatedFlag flag:

    1 - NONE OF THE PROCESSES IS ABLE TO FINISH AT FIRST ITERATION, flag is set to false. At the 2nd iteration, loop is broken. flag == false

    2 - SOME OF THE PROCESSES FINISHED EARLIER. AT SOME POINT NONE OF THE PROCESSES IS ABLE TO FINISH. 
        So, none of the processes is able to set the flag to true. At the next iteration, loop is broken. flag == false

    3 - ALL OF THE PROCESSES ARE ABLE TO FINISH. At the last iteration, last process finished and set flag = true.
        At the next iteration, none of the processes has isFinished == false since all of them are finished. Loop is exited and flag == true.

    So, if the state is safe, i.e. all processes are able to finish (Scenario - 3), then flag == true. Otherwise, if the state is unsafe, flag == false.
    */
    if (tid == 0)
    {
        if (workUpdatedFlag == true)
        {
            *isSafe = true;
        }
        else
        {
            *isSafe = false;
        }
    }

    finish[tid] = isFinished;

    cudaFree(isNeedGreaterThanWork);
}


// Helper functions
template<class T>
void printMatrix(T* matrix, int nRows, int nCols, const char* matrixName)
{
    std::cout << "\n" << matrixName << ":\n";
    for (int i = 0; i < nRows; i++)
    {
        std::cout << "[ ";
        for (int j = 0; j < nCols; j++)
        {
            std::cout << matrix[i * nCols + j] << " ";
        }
        std::cout << "]\n";
    }
}

int main()
{
    int* available_h, * max_h, * allocation_h, * need_h, * request_h;
    int* available_d, * max_d, * allocation_d, * need_d, * request_d;

    int numProcesses = 5;
    int numResources = 3;

    // Memory allocations
    available_h = new int[numResources] {3, 3, 2};
    max_h = new int[numProcesses * numResources]{ 7, 5, 3,
                                                3, 2, 2,
                                                9, 0, 2,
                                                2, 2, 2,
                                                4, 3, 3 };
    allocation_h = new int[numProcesses * numResources]{ 0, 1, 0,
                                                        2, 0, 0,
                                                        3, 0, 2,
                                                        2, 1, 1,
                                                        0, 0, 2 };
    need_h = new int[numProcesses * numResources];

    request_h = new int[numResources] {1, 0, 2};
    unsigned int requestingProcessId = 1;

    for (int i = 0; i < numProcesses; i++)
    {
        for (int j = 0; j < numResources; j++)
        {
            need_h[i * numResources + j] = max_h[i * numResources + j] - allocation_h[i * numResources + j];
        }
    }

    /*available_h = new int[numResources];
    max_h = new int[numProcesses * numResources];
    allocation_h = new int[numProcesses * numResources];
    need_h = new int[numProcesses * numResources];

    request_h = new int[numResources];
    unsigned int requestingProcessId = 1;*/

    CHECK(cudaMalloc(&available_d, numResources * sizeof(int)));
    CHECK(cudaMalloc(&max_d, numResources * numProcesses * sizeof(int)));
    CHECK(cudaMalloc(&allocation_d, numResources * numProcesses * sizeof(int)));
    CHECK(cudaMalloc(&need_d, numResources * numProcesses * sizeof(int)));
    CHECK(cudaMalloc(&request_d, numResources * sizeof(int)));

    /* RANDOM INITIALIZATION*/

    //int maxResourceAmount = 6;

    //std::random_device rd;
    //std::mt19937 mt(rd());
    //std::uniform_int_distribution<int> dist(0, maxResourceAmount);

    //for (int i = 0; i < numProcesses; i++)
    //{
    //    for (int j = 0; j < numResources; j++)
    //    {
    //        max_h[i * numResources + j] = dist(mt);
    //    }
    //}

    //// allocation_h and need_h matrices. allocation_h can't be greater than max_h. need_h = max_h - allocation_h
    //for (int i = 0; i < numProcesses; i++)
    //{
    //    for (int j = 0; j < numResources; j++)
    //    {
    //        dist.param(std::uniform_int_distribution<int>::param_type(0, max_h[i * numResources + j]));
    //        allocation_h[i * numResources + j] = dist(mt);
    //        need_h[i * numResources + j] = max_h[i * numResources + j] - allocation_h[i * numResources + j];
    //    }
    //}
    //for (int i = 0; i < numResources; i++)
    //{
    //    dist.param(std::uniform_int_distribution<int>::param_type(0, maxResourceAmount / 2));
    //    available_h[i] = dist(mt);
    //}

    /*END OF RANDOM INITIALIZATION*/


    printMatrix<int>(max_h, numProcesses, numResources, "max_h");
    printMatrix<int>(allocation_h, numProcesses, numResources, "allocation_h");
    printMatrix<int>(need_h, numProcesses, numResources, "need_h");
    printMatrix<int>(available_h, 1, numResources, "available_h");
    printMatrix<int>(request_h, 1, numResources, "request_h");

    StartCounter();

    cudaStream_t stream_1, stream_2, stream_3, stream_4;
    CHECK(cudaStreamCreate(&stream_1));
    CHECK(cudaStreamCreate(&stream_2));
    CHECK(cudaStreamCreate(&stream_3));
    CHECK(cudaStreamCreate(&stream_4));


    CHECK(cudaMemcpyAsync(available_d, available_h, numResources * sizeof(int), cudaMemcpyHostToDevice, 0));
    CHECK(cudaMemcpyAsync(max_d, max_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_1));
    CHECK(cudaMemcpyAsync(allocation_d, allocation_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_2));
    CHECK(cudaMemcpyAsync(need_d, need_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_3));
    CHECK(cudaMemcpyAsync(request_d, request_h, numResources * sizeof(int), cudaMemcpyHostToDevice, stream_4));

    /*SINGLE-STREAM MEMCPY OF MATRICES TO THE DEVICE*/
    /*cudaStream_t stream_1, stream_2, stream_3, stream_4;
    CHECK(cudaStreamCreate(&stream_1));
    CHECK(cudaStreamCreate(&stream_2));
    CHECK(cudaStreamCreate(&stream_3));
    CHECK(cudaStreamCreate(&stream_4));

    CHECK(cudaMemcpy(available_d, available_h, numResources * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(max_d, max_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(allocation_d, allocation_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(need_d, need_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(request_d, request_h, numResources * sizeof(int), cudaMemcpyHostToDevice));*/

    CHECK(cudaDeviceSynchronize());

    double execution_duration = GetCounter();
    std::cout << "MEMCPY DURATION: " << execution_duration << " ms\n";

    bool isRequestValid;

    // VALIDATION STATE

    dim3 grid, block;

    block.x = std::min(numResources, 1024);
    grid.x = static_cast<unsigned int>(std::ceil(static_cast<double>(numResources) / block.x));

    bool* isNotValidated_h,* isNotValidated_d; // 2 - dim arrays holding whether Request_i < Need(i) and Request_i < Available_i for all i or not

    isNotValidated_h = new bool[2];

    CHECK(cudaMalloc(&isNotValidated_d, 2 * sizeof(bool)));
    CHECK(cudaMemset((void*)isNotValidated_d, 0, 2 * sizeof(bool)));

    CHECK(cudaDeviceSynchronize());

    checkIfGreater << <grid, block, 0, stream_1 >> > (request_d, &need_d[numResources * requestingProcessId], numResources, &isNotValidated_d[0]);
    checkIfGreater << <grid, block, 0, stream_2 >> > (request_d, available_d, numResources, &isNotValidated_d[1]);

    CHECK(cudaMemcpyAsync(&isNotValidated_h[0], &isNotValidated_d[0], sizeof(bool), cudaMemcpyDeviceToHost, stream_1));
    CHECK(cudaMemcpyAsync(&isNotValidated_h[1], &isNotValidated_d[1], sizeof(bool), cudaMemcpyDeviceToHost, stream_2));

    CHECK(cudaDeviceSynchronize());

    if (isNotValidated_h[0] == true || isNotValidated_h[1] == true)
    {
        isRequestValid = false;
    }
    else
    {
        isRequestValid = true;
    }

    std::cout << "Is Request Valid: " << isRequestValid << "\n";

    // If request is valid, go to modification state and modify the available, allocation, and need matrices as if the allocation was made
   
    /* MODIFICATION STATE */

    subtractVectors << <grid, block, 0, stream_1 >> > (available_d, request_d, numResources); // available(i) = available(i) - request_i
    addVectors << <grid, block, 0, stream_2 >> > (&allocation_d[numResources * requestingProcessId], request_d, numResources); // allocation(i) = allocation(i) + request_i
    subtractVectors << <grid, block, 0, stream_3 >> > (&need_d[numResources * requestingProcessId], request_d, numResources); // need(i) = need(i) - request_i

    CHECK(cudaDeviceSynchronize());

    /* END OF MODIFICATION STATE */

    /* COPY MODIFIED VECTORS BACK TO THE HOST AND PRINT -- JUST FOR PRINTING AND VERIFICATION. NOT ACTIVATED DURING BENCHMARKING */
    int* available_h_modified = new int[numResources];
    int* allocation_h_modified = new int[numProcesses * numResources];
    int* need_h_modified = new int[numProcesses * numResources];
    CHECK(cudaMemcpyAsync(available_h_modified, available_d, numResources * sizeof(int), cudaMemcpyDeviceToHost, stream_1));
    CHECK(cudaMemcpyAsync(allocation_h_modified, allocation_d, numProcesses * numResources * sizeof(int), cudaMemcpyDeviceToHost, stream_2));
    CHECK(cudaMemcpyAsync(need_h_modified, need_d, numProcesses * numResources * sizeof(int), cudaMemcpyDeviceToHost, stream_3));

    CHECK(cudaDeviceSynchronize());

    printMatrix<int>(allocation_h_modified, numProcesses, numResources, "allocation_h_modified");
    printMatrix<int>(need_h_modified, numProcesses, numResources, "need_h_modified");
    printMatrix<int>(available_h_modified, 1, numResources, "available_h_modified");

    delete[] available_h_modified;
    delete[] need_h_modified;
    delete[] allocation_h_modified;

    /* END OF COPY MODIFIED VECTORS BACK TO THE HOST AND PRINT */

    /* SAFETY STATE */

    // Initialize work and finish vectors
    int* work_d; // 1 x numResources
    bool* finish_d; // 1 x numProcesses

    CHECK(cudaMalloc(&work_d, numResources * sizeof(int)));
    CHECK(cudaMalloc(&finish_d, numProcesses * sizeof(bool)));

    /* ************************ May not be necessary, just use available_d as work matrix *************************** */
    CHECK(cudaMemcpyAsync(work_d, available_d, numResources * sizeof(int), cudaMemcpyDeviceToDevice, stream_1));
    CHECK(cudaMemsetAsync((void*)finish_d, 0, numProcesses * sizeof(bool), stream_2));

    CHECK(cudaDeviceSynchronize());

    bool* isSafe_h, * isSafe_d;

    isSafe_h = new bool;
    CHECK(cudaMalloc(&isSafe_d, sizeof(bool)));

    dim3 block_safety;
    block_safety.x = std::min(numProcesses, 1024);

    safetyState << <1, block_safety, (block_safety.x + 1) * sizeof(bool), stream_1 >> > (work_d, need_d, allocation_d, finish_d, numResources, numProcesses, isSafe_d);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(isSafe_h, isSafe_d, sizeof(bool), cudaMemcpyDeviceToHost));

    /* END OF SAFETY STATE */

    bool* finish_h = new bool[numProcesses];

    CHECK(cudaMemcpy(finish_h, finish_d, numProcesses * sizeof(bool), cudaMemcpyDeviceToHost));

    printMatrix<bool>(finish_h, 1, numProcesses, "finish_h");

    std::cout << "IS SAFE: " << *isSafe_h << "\n";

    // Destroy streams
    CHECK(cudaStreamDestroy(stream_1));
    CHECK(cudaStreamDestroy(stream_2));
    CHECK(cudaStreamDestroy(stream_3));
    CHECK(cudaStreamDestroy(stream_4));

    // Free allocated memory spaces
    delete[] available_h;
    delete[] max_h;
    delete[] allocation_h;
    delete[] need_h;
    delete[] request_h;

    delete[] isNotValidated_h;
    delete isSafe_h;


    CHECK(cudaFree(available_d));
    CHECK(cudaFree(max_d));
    CHECK(cudaFree(allocation_d));
    CHECK(cudaFree(need_d));
    CHECK(cudaFree(request_d));

    CHECK(cudaFree(isNotValidated_d));
    CHECK(cudaFree(work_d));
    CHECK(cudaFree(finish_d));
    CHECK(cudaFree(isSafe_d));

    return 0;
}
