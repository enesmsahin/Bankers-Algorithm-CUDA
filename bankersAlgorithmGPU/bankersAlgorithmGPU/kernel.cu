
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <stdlib.h>

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

            workUpdatedFlag = false; // Reset work vector's update flag


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


bool bankersAlgorithmHandler(int*& available_d, int*& max_d, int*& allocation_d, int*& need_d, int*& request_d,
    int requestingProcessId, int numProcesses, int numResources,
    cudaStream_t& stream_1, cudaStream_t& stream_2, cudaStream_t& stream_3, cudaStream_t& stream_4);

// Helper functions

/* Reads matrices from text files into host vectors */
void readMatrices(int*& available, int*& max, int*& allocation, int*& need, int*& request,
    int& numProcesses, int& numResources, int& requestingProcessId);

/* Prints given matrix to the screen */
template<class T>
void printMatrix(T* matrix, int nRows, int nCols, const char* matrixName);

int main()
{
    int* available_h = nullptr, * max_h = nullptr, * allocation_h = nullptr, * need_h = nullptr, * request_h = nullptr;
    int* available_d, * max_d, * allocation_d, * need_d, * request_d;

    int numProcesses;
    int numResources;
    int requestingProcessId;
    
    std::cout << "Matrices are being read from the text files...\n\n";

    readMatrices(available_h, max_h, allocation_h, need_h, request_h, numProcesses, numResources, requestingProcessId);

    std::cout << "Matrices are read from the text files. Execution of the algorithm begins.\n\n";

    /*printMatrix<int>(max_h, numProcesses, numResources, "max_h");
    printMatrix<int>(allocation_h, numProcesses, numResources, "allocation_h");
    printMatrix<int>(need_h, numProcesses, numResources, "need_h");
    printMatrix<int>(available_h, 1, numResources, "available_h");
    printMatrix<int>(request_h, 1, numResources, "request_h");*/

    int numExecutions = 10;
    double totalExecDuration = 0;
    double firstExecDuration;

    for (int i = 0; i < numExecutions; i++)
    {
        StartCounter();

        /* DEVICE MEMORY ALLOCATIONS */
        CHECK(cudaMalloc(&available_d, numResources * sizeof(int)));
        CHECK(cudaMalloc(&max_d, numResources * numProcesses * sizeof(int)));
        CHECK(cudaMalloc(&allocation_d, numResources * numProcesses * sizeof(int)));
        CHECK(cudaMalloc(&need_d, numResources * numProcesses * sizeof(int)));
        CHECK(cudaMalloc(&request_d, numResources * sizeof(int)));



        /* Create CUDA streams that will be used throughout the program */
        cudaStream_t stream_1, stream_2, stream_3, stream_4;
        CHECK(cudaStreamCreate(&stream_1));
        CHECK(cudaStreamCreate(&stream_2));
        CHECK(cudaStreamCreate(&stream_3));
        CHECK(cudaStreamCreate(&stream_4));

        /* Copy initialized host matrices to the allocated device matrices */
        CHECK(cudaMemcpyAsync(available_d, available_h, numResources * sizeof(int), cudaMemcpyHostToDevice, 0));
        CHECK(cudaMemcpyAsync(max_d, max_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_1));
        CHECK(cudaMemcpyAsync(allocation_d, allocation_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_2));
        CHECK(cudaMemcpyAsync(need_d, need_h, numResources * numProcesses * sizeof(int), cudaMemcpyHostToDevice, stream_3));
        CHECK(cudaMemcpyAsync(request_d, request_h, numResources * sizeof(int), cudaMemcpyHostToDevice, stream_4));

        CHECK(cudaDeviceSynchronize());

        bool isRequestServable;

        isRequestServable = bankersAlgorithmHandler(available_d, max_d, allocation_d, need_d, request_d, requestingProcessId, numProcesses, numResources,
            stream_1, stream_2, stream_3, stream_4);

        std::cout << "IS REQUESTED ALLOCATION SERVABLE: " << isRequestServable << "\n\n";

        // Destroy streams
        CHECK(cudaStreamDestroy(stream_1));
        CHECK(cudaStreamDestroy(stream_2));
        CHECK(cudaStreamDestroy(stream_3));
        CHECK(cudaStreamDestroy(stream_4));

        // Free GPU allocations
        CHECK(cudaFree(available_d));
        CHECK(cudaFree(max_d));
        CHECK(cudaFree(allocation_d));
        CHECK(cudaFree(need_d));
        CHECK(cudaFree(request_d));

        double execution_duration = GetCounter();
        std::cout << "EXECUTION DURATION: " << execution_duration << " ms\n";

        if (i == 0)
        {
            firstExecDuration = execution_duration;
            continue;
        }
        totalExecDuration += execution_duration;
    }
    
    std::cout << "\n\nAverage Execution Duration: " << totalExecDuration / (numExecutions - 1) << "\n";
    std::cout << "Average Execution Duration (including first execution): " << (totalExecDuration + firstExecDuration) / (numExecutions) << "\n\n";

    // Free host allocations
    delete[] available_h;
    delete[] max_h;
    delete[] allocation_h;
    delete[] need_h;
    delete[] request_h;


    return 0;
}

bool bankersAlgorithmHandler(int* &available_d, int* &max_d, int* &allocation_d, int* &need_d, int* &request_d,
                                int requestingProcessId, int numProcesses, int numResources,
                                cudaStream_t &stream_1, cudaStream_t& stream_2, cudaStream_t& stream_3, cudaStream_t& stream_4)
{
    // VALIDATION STATE

    dim3 grid, block;

    block.x = std::min(numResources, 1024);
    grid.x = static_cast<unsigned int>(std::ceil(static_cast<double>(numResources) / block.x));

    bool* isNotValidated_h, * isNotValidated_d; // 2 - dim arrays holding whether Request_i < Need(i) and Request_i < Available_i for all i or not

    isNotValidated_h = new bool[2];

    CHECK(cudaMalloc(&isNotValidated_d, 2 * sizeof(bool)));
    CHECK(cudaMemset((void*)isNotValidated_d, 0, 2 * sizeof(bool)));

    CHECK(cudaDeviceSynchronize());

    checkIfGreater << <grid, block, 0, stream_1 >> > (request_d, &need_d[numResources * requestingProcessId], numResources, &isNotValidated_d[0]);
    checkIfGreater << <grid, block, 0, stream_2 >> > (request_d, available_d, numResources, &isNotValidated_d[1]);

    CHECK(cudaMemcpyAsync(&isNotValidated_h[0], &isNotValidated_d[0], sizeof(bool), cudaMemcpyDeviceToHost, stream_1));
    CHECK(cudaMemcpyAsync(&isNotValidated_h[1], &isNotValidated_d[1], sizeof(bool), cudaMemcpyDeviceToHost, stream_2));

    CHECK(cudaDeviceSynchronize());

    if (isNotValidated_h[0] == true)
    {
        std::cout << "Validation Failed! Request_i < Need(i) is not satisfied!\n";
        delete[] isNotValidated_h;
        CHECK(cudaFree(isNotValidated_d));
        return false;
    }
    else if (isNotValidated_h[1] == true)
    {
        std::cout << "Validation Failed! Request_i < Available(i) is not satisfied!\n";
        delete[] isNotValidated_h;
        CHECK(cudaFree(isNotValidated_d));
        return false;
    }

    delete[] isNotValidated_h;
    CHECK(cudaFree(isNotValidated_d));

    // If request is valid, go to modification state and modify the available, allocation, and need matrices as if the allocation was made

    /* MODIFICATION STATE */

    subtractVectors << <grid, block, 0, stream_1 >> > (available_d, request_d, numResources); // available(i) = available(i) - request_i
    addVectors << <grid, block, 0, stream_2 >> > (&allocation_d[numResources * requestingProcessId], request_d, numResources); // allocation(i) = allocation(i) + request_i
    subtractVectors << <grid, block, 0, stream_3 >> > (&need_d[numResources * requestingProcessId], request_d, numResources); // need(i) = need(i) - request_i

    CHECK(cudaDeviceSynchronize());

    /* END OF MODIFICATION STATE */

    /* COPY MODIFIED VECTORS BACK TO THE HOST AND PRINT -- JUST FOR PRINTING AND VERIFICATION. NOT ACTIVATED DURING BENCHMARKING */
    /*int* available_h_modified = new int[numResources];
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
    delete[] allocation_h_modified;*/

    /* END OF COPY MODIFIED VECTORS BACK TO THE HOST AND PRINT */

    /* SAFETY STATE */

    // Initialize work and finish vectors
    int* work_d; // 1 x numResources
    bool* finish_d; // 1 x numProcesses

    CHECK(cudaMalloc(&work_d, numResources * sizeof(int)));
    CHECK(cudaMalloc(&finish_d, numProcesses * sizeof(bool)));

    CHECK(cudaMemcpyAsync(work_d, available_d, numResources * sizeof(int), cudaMemcpyDeviceToDevice, stream_1)); // work = available
    CHECK(cudaMemsetAsync((void*)finish_d, 0, numProcesses * sizeof(bool), stream_2)); // finish[i] = false for all i

    CHECK(cudaDeviceSynchronize());

    bool* isSafe_h, * isSafe_d;

    isSafe_h = new bool;
    CHECK(cudaMalloc(&isSafe_d, sizeof(bool)));

    dim3 block_safety;
    block_safety.x = std::min(numProcesses, 1024);

    safetyState << <1, block_safety>> > (work_d, need_d, allocation_d, finish_d, numResources, numProcesses, isSafe_d);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(isSafe_h, isSafe_d, sizeof(bool), cudaMemcpyDeviceToHost));

    /* END OF SAFETY STATE */

    bool* finish_h = new bool[numProcesses];

    CHECK(cudaMemcpy(finish_h, finish_d, numProcesses * sizeof(bool), cudaMemcpyDeviceToHost));

    //printMatrix<bool>(finish_h, 1, numProcesses, "finish_h");

    delete[] finish_h;

    CHECK(cudaFree(work_d));
    CHECK(cudaFree(finish_d));
    CHECK(cudaFree(isSafe_d));

    if (*isSafe_h == true)
    {
        std::cout << "All processes CAN terminate succesfully if the allocation is made (SAFE STATE). So, the request IS servable!\n\n";
        delete isSafe_h;
        return true;
    }
    else
    {
        std::cout << "All processes CANNOT terminate succesfully if the allocation is made (UNSAFE STATE). So, the request IS NOT servable!\n\n";
        delete isSafe_h;
        return false;
    }
}


void readMatrices(int*& available, int*& max, int*& allocation, int*& need, int*& request,
    int& numProcesses, int& numResources, int& requestingProcessId)
{
    std::ifstream info_file("info.txt");
    std::string info;

    /* PARSE INFO FILE */
    if (info_file.is_open())
    {
        while (info_file >> info)
        {
            if (info == "numProcesses:")
            {
                info_file >> numProcesses;
            }
            else if (info == "numResources:")
            {
                info_file >> numResources;
            }
            else if (info == "requestingProcessId:")
            {
                info_file >> requestingProcessId;
            }
        }
        info_file.close();
    }
    else
    {
        std::cout << "UNABLE TO OPEN INFO FILE!\n";
        exit(1);
    }

    /* HOST MEMORY ALLOCATIONS */
    available = new int[numResources];
    max = new int[numProcesses * numResources];
    allocation = new int[numProcesses * numResources];
    need = new int[numProcesses * numResources];
    request = new int[numResources];

    if ((available == nullptr) || (max == nullptr) || (allocation == nullptr) || (need == nullptr) || (request == nullptr))
    {
        std::cout << "HOST MEMORY ALLOCATION ERROR!\n";
        exit(1);
    }

    std::ifstream available_file("available.txt");
    std::ifstream max_file("max.txt");
    std::ifstream allocation_file("allocation.txt");
    std::ifstream need_file("need.txt");
    std::ifstream request_file("request.txt");

    /* CHECK IF FILES ARE OPENED SUCCESSFULLY */
    if (!available_file.is_open())
    {
        std::cout << "UNABLE TO OPEN AVAILABLE FILE!\n";
        exit(1);
    }
    if (!max_file.is_open())
    {
        std::cout << "UNABLE TO OPEN MAX FILE!\n";
        exit(1);
    }
    if (!allocation_file.is_open())
    {
        std::cout << "UNABLE TO OPEN ALLOCATION FILE!\n";
        exit(1);
    }
    if (!need_file.is_open())
    {
        std::cout << "UNABLE TO OPEN NEED FILE!\n";
        exit(1);
    }
    if (!request_file.is_open())
    {
        std::cout << "UNABLE TO OPEN REQUEST FILE!\n";
        exit(1);
    }

    /* READ MATRICES INTO HOST ARRAYS FROM TEXT FILES */
    for (int i = 0; i < numProcesses * numResources; i++)
    {
        if (i < numResources)
        {
            available_file >> available[i];
            request_file >> request[i];
        }

        max_file >> max[i];
        allocation_file >> allocation[i];
        need_file >> need[i];
    }

    /* CLOSE FILES */
    available_file.close();
    max_file.close();
    allocation_file.close();
    need_file.close();
    request_file.close();
}

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