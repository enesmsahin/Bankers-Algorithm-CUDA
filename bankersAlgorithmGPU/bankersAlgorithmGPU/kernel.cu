
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include <iostream>
#include <random>
#include <cmath>

#define CHECK(status)									            \
{														            \
	if (status != cudaSuccess)							            \
	{													            \
		std::cout << "Cuda error: " << cudaGetErrorString(status);  \
	}													            \
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


// Helper functions
void printMatrix(int* matrix, int nRows, int nCols, const char* matrixName)
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


    printMatrix(max_h, numProcesses, numResources, "max_h");
    printMatrix(allocation_h, numProcesses, numResources, "allocation_h");
    printMatrix(need_h, numProcesses, numResources, "need_h");
    printMatrix(available_h, 1, numResources, "available_h");
    printMatrix(request_h, 1, numResources, "request_h");

    StartCounter();

    cudaStream_t stream_1, stream_2, stream_3, stream_4;
    CHECK(cudaStreamCreate(&stream_1));
    CHECK(cudaStreamCreate(&stream_2));
    CHECK(cudaStreamCreate(&stream_3));
    CHECK(cudaStreamCreate(&stream_4));


    CHECK(cudaMemcpyAsync((void*)available_d, available_h, numResources * sizeof(int), cudaMemcpyHostToDevice, 0));
    CHECK(cudaMemcpyAsync((void*)max_d, max_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_1));
    CHECK(cudaMemcpyAsync((void*)allocation_d, allocation_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_2));
    CHECK(cudaMemcpyAsync((void*)need_d, need_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_3));
    CHECK(cudaMemcpyAsync((void*)request_d, request_h, numResources * sizeof(int), cudaMemcpyHostToDevice, stream_4));

    /*SINGLE-STREAM MEMCPY OF MATRICES TO THE DEVICE*/
    /*cudaStream_t stream_1, stream_2, stream_3, stream_4;
    CHECK(cudaStreamCreate(&stream_1));
    CHECK(cudaStreamCreate(&stream_2));
    CHECK(cudaStreamCreate(&stream_3));
    CHECK(cudaStreamCreate(&stream_4));

    CHECK(cudaMemcpy((void*)available_d, available_h, numResources * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)max_d, max_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)allocation_d, allocation_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)need_d, need_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)request_d, request_h, numResources * sizeof(int), cudaMemcpyHostToDevice));*/

    CHECK(cudaDeviceSynchronize());

    double execution_duration = GetCounter();
    std::cout << "MEMCPY DURATION: " << execution_duration << " ms\n";

    bool isRequestValid;

    // VALIDATION STATE

    dim3 grid, block;

    block.x = std::min(numProcesses, 1024);
    grid.x = static_cast<unsigned int>(std::ceil(static_cast<double>(numProcesses) / block.x));

    bool* isNotValidated_h,* isNotValidated_d; // 2 - dim arrays holding whether Request_i < Need(i) and Request_i < Available_i for all i or not

    isNotValidated_h = new bool[2];

    CHECK(cudaMalloc(&isNotValidated_d, 2 * sizeof(bool)));
    CHECK(cudaMemset((void*)isNotValidated_d, 0, 2 * sizeof(bool)));

    checkIfGreater << <grid, block, 0, stream_1 >> > (request_d, &need_d[numResources * requestingProcessId], numResources, &isNotValidated_d[0]);
    checkIfGreater << <grid, block, 0, stream_2 >> > (request_d, available_d, numResources, &isNotValidated_d[1]);

    CHECK(cudaMemcpyAsync((void*)&isNotValidated_h[0], &isNotValidated_d[0], sizeof(bool), cudaMemcpyDeviceToHost, stream_1));
    CHECK(cudaMemcpyAsync((void*)&isNotValidated_h[1], &isNotValidated_d[1], sizeof(bool), cudaMemcpyDeviceToHost, stream_2));

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
   
    // MODIFICATION STATE

    subtractVectors << <grid, block, 0, stream_1 >> > (available_d, request_d, numResources); // available(i) = available(i) - request_i
    addVectors << <grid, block, 0, stream_2 >> > (&allocation_d[numResources * requestingProcessId], request_d, numResources); // allocation(i) = allocation(i) + request_i
    subtractVectors << <grid, block, 0, stream_3 >> > (&need_d[numResources * requestingProcessId], request_d, numResources); // need(i) = need(i) - request_i

    CHECK(cudaDeviceSynchronize());

    /* COPY MODIFIED VECTORS BACK TO THE HOST AND PRINT -- JUST FOR PRINTING AND VERIFICATION. NOT ACTIVATED DURING BENCHMARKING */
    int* available_h_modified = new int[numResources];
    int* allocation_h_modified = new int[numProcesses * numResources];
    int* need_h_modified = new int[numProcesses * numResources];
    CHECK(cudaMemcpyAsync((void*)available_h_modified, available_d, numResources * sizeof(int), cudaMemcpyDeviceToHost, stream_1));
    CHECK(cudaMemcpyAsync((void*)allocation_h_modified, allocation_d, numProcesses * numResources * sizeof(int), cudaMemcpyDeviceToHost, stream_2));
    CHECK(cudaMemcpyAsync((void*)need_h_modified, need_d, numProcesses * numResources * sizeof(int), cudaMemcpyDeviceToHost, stream_3));

    CHECK(cudaDeviceSynchronize());

    printMatrix(allocation_h_modified, numProcesses, numResources, "allocation_h_modified");
    printMatrix(need_h_modified, numProcesses, numResources, "need_h_modified");
    printMatrix(available_h_modified, 1, numResources, "available_h_modified");

    /* END OF COPY MODIFIED VECTORS BACK TO THE HOST AND PRINT */

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

    CHECK(cudaFree(available_d));
    CHECK(cudaFree(max_d));
    CHECK(cudaFree(allocation_d));
    CHECK(cudaFree(need_d));
    CHECK(cudaFree(request_d));

    return 0;
}
