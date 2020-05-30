#include <iostream>
#include <random>
#include <iterator>
#include <algorithm>



bool bankersAlgorithmSequential(const int numProcesses, const int numResources, int* available, int* max, int* allocation, int* need, int* request, unsigned int requestingProcessId);

// Helper funtions
template<class T>
bool isLessThan(T* smaller, T* greater, const int size);


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

	bool multiThreaded = false;

	int* available, * max, * allocation, * need, * request;

	int numProcesses = 5;
	int numResources = 3;

	// Memory allocations
	available = new int[numResources] {3, 3, 2};
	max = new int[numProcesses * numResources] {7, 5, 3,
												3, 2, 2,
												9, 0, 2,
												2, 2, 2,
												4, 3, 3};
	allocation = new int[numProcesses * numResources] {	0, 1, 0,
														2, 0, 0,
														3, 0, 2,
														2, 1, 1,
														0, 0, 2};
	need = new int[numProcesses * numResources];

	request = new int[numResources] {1, 0, 2};
	unsigned int requestingProcessId = 1;

	// Initialize matrices

	int maxResourceAmount = 6;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<int> dist(0, maxResourceAmount);

	// max matrix
	/*for (int i = 0; i < numProcesses; i++)
	{
		for (int j = 0; j < numResources; j++)
		{
			max[i * numResources + j] = dist(mt);
		}
	}*/
	printMatrix(max, numProcesses, numResources, "max");

	// allocation and need matrices. allocation can't be greater than max. need = max - allocation
	/*for (int i = 0; i < numProcesses; i++)
	{
		for (int j = 0; j < numResources; j++)
		{
			dist.param(std::uniform_int_distribution<int>::param_type(0, max[i * numResources + j]));
			allocation[i * numResources + j] = dist(mt);
			need[i * numResources + j] = max[i * numResources + j] - allocation[i * numResources + j];
		}
	}*/
	for (int i = 0; i < numProcesses; i++)
	{
		for (int j = 0; j < numResources; j++)
		{
			need[i * numResources + j] = max[i * numResources + j] - allocation[i * numResources + j];
		}
	}
	printMatrix(allocation, numProcesses, numResources, "allocation");
	printMatrix(need, numProcesses, numResources, "need");



	/*for (int i = 0; i < numResources; i++)
	{
		dist.param(std::uniform_int_distribution<int>::param_type(0, maxResourceAmount / 2));
		available[i] = dist(mt);
	}*/
	printMatrix(available, 1, numResources, "available");

	bool isRequestServable;

	if (!multiThreaded)
	{
		isRequestServable = bankersAlgorithmSequential();
	}

	return 0;
}

bool bankersAlgorithmSequential(const int numProcesses, const int numResources, int* available, int* max, 
								int* allocation, int* need, int* request, unsigned int requestingProcessId)
{
	/*VALIDATION STATE*/
	
	// Check if Request-i < Need(i) and Request-i < Available(i)
	for (int i = 0; i < numResources; i++)
	{
		if ( (request[i] > need[numResources * requestingProcessId + i]) || (request[i] > available[i]))
		{
			std::cout << "Invalid request for resource id: " << i << ". Current matrix states for resource " << i << ":\n"
				<< "Request[i]: " << request[i] << "\nNeed[i]: " << need[numResources * requestingProcessId + i]
				<< "\nAvailable[i]: " << available[i] << "\n\nSo, request is not feasible !!!\n";
			return false;
		}
	}

	/*MODIFICATION STATE*/
	for (int i = 0; i < numResources; i++)
	{
		available[i] = available[i] - request[i];
		allocation[numResources * requestingProcessId + i] = allocation[numResources * requestingProcessId + i] + request[i];
		need[numResources * requestingProcessId + i] = need[numResources * requestingProcessId + i] - request[i];
	}


	/*SAFETY STATE*/
	int* work = new int[numResources];
	bool* finish = new bool[numProcesses] {false};

	// Initialize work matrix as work = available
	std::copy(available, available + numResources, work);

	// Find an i such that Finish [i] == false and Need(i) <= Work
	for (int i = 0; i < numProcesses; i++)
	{
		if ((finish[i] == false) && isLessThan<int>(&need[numResources * i], work, numResources))
		{

		}
	}

	return true;
}


template<class T>
bool isLessThan(T* smaller, T* greater, const int size)
{
	for (int i = 0; i < size; i++)
	{
		if (smaller[i] > greater[i])
		{
			return false;
		}
	}

	return true;
}
