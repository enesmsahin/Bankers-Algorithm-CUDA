#include <iostream>
#include <random>
#include <iterator>
#include <algorithm>
#include <vector>
#include <fstream>

#include <thread>
#include <cassert>


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


enum class OP
{
	ADD,
	SUB
};

bool bankersAlgorithmSequential(int numProcesses, int numResources, const std::vector<int>& available_orig,
								const std::vector<int>& max, const std::vector<int>& allocation_orig, const std::vector<int>& need_orig,
								const std::vector<int>& request, int requestingProcessId);

bool bankersAlgorithmParallel(int numProcesses, int numResources, const std::vector<int>& available_orig,
							const std::vector<int>& max, const std::vector<int>& allocation_orig, const std::vector<int>& need_orig,
							const std::vector<int>& request, int requestingProcessId);

// Helper funtions

/* Reads matrices from text files into host vectors */
void readMatrices(std::vector<int>& available, std::vector<int>& max, std::vector<int>& allocation, std::vector<int>& need, std::vector<int>& request,
	int& numProcesses, int& numResources, int& requestingProcessId);

/* Prints given matrix to the screen */
template<class T>
void printMatrix(const std::vector<T>& matrix, int nRows, int nCols, const char* matrixName);

int main()
{
	bool multiThreaded = false; // Flag to determine whether multithreaded or single-threaded CPU implementation will be used.

	int numProcesses;
	int numResources;
	int requestingProcessId;

	std::vector<int> available;
	std::vector<int> max;
	std::vector<int> allocation;
	std::vector<int> need;
	std::vector<int> request;

	readMatrices(available, max, allocation, need, request, numProcesses, numResources, requestingProcessId);

	/*printMatrix<int>(max, numProcesses, numResources, "max");
	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");
	printMatrix<int>(request, 1, numResources, "request");*/

	StartCounter();

	bool isRequestServable;

	if (!multiThreaded)
	{
		isRequestServable = bankersAlgorithmSequential(numProcesses, numResources, available, max, allocation, need, request, requestingProcessId);
	}
	else
	{
		isRequestServable = bankersAlgorithmParallel(numProcesses, numResources, available, max, allocation, need, request, requestingProcessId);
	}

	std::cout << "IS REQUESTED ALLOCATION SERVABLE: " << isRequestServable << "\n";

	double execution_duration = GetCounter();
    std::cout << "EXECUTION DURATION: " << execution_duration << " ms\n";

	return 0;
}

bool bankersAlgorithmSequential(int numProcesses, int numResources, const std::vector<int>& available_orig,
								const std::vector<int>& max, const std::vector<int>& allocation_orig, const std::vector<int>& need_orig,
								const std::vector<int>& request, int requestingProcessId)
{

	std::vector<int> available = available_orig;
	std::vector<int> allocation = allocation_orig;
	std::vector<int> need = need_orig;

	int i = -1;
	/*VALIDATION STATE*/
	auto it_req = std::find_if(request.begin(), request.end(), [&](int req_i)
		{
			i++;
			return ((req_i > need.at(numResources * requestingProcessId + i)) || (req_i > available.at(i))); // Check if Request(i) > Need(i) or Request(i) > Available(i), look for an exception.
		});

	if (it_req != request.end()) // If an exception is found, request is not valid and servable.
	{
		int invalidIdx = it_req - request.begin();
		std::cout << "Invalid request for resource id: " << invalidIdx << ". Current matrix states for resource " << invalidIdx << ":\n"
			<< "Request[i]: " << request.at(invalidIdx) << "\nNeed[i]: " << need.at(numResources * requestingProcessId + invalidIdx)
			<< "\nAvailable[i]: " << available.at(invalidIdx) << "\n\nSo, request is not feasible !!!\n";
		return false;
	}

	// If request is valid, go to modification state and modify the available, allocation, and need matrices as if the allocation was made

	/*MODIFICATION STATE*/
	i = -1;
	std::for_each(request.begin(), request.end(), [&](int req_i)
		{
			i++;
			available.at(i) -= req_i;
			allocation.at(numResources * requestingProcessId + i) += req_i;
			need.at(numResources * requestingProcessId + i) -= req_i;
		});


	/*std::cout << "MATRICES AFTER MODIFICATION: \n\n";

	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");*/

	// After modifying matrices, check whether resulting state is safe, i.e. deadlock-free

	/*SAFETY STATE*/
	// Initialize work matrix as work = available
	std::vector<int> work(available.begin(), available.end());
	std::vector<bool> finish(numProcesses, false);

	// Find an i such that Finish [i] == false and Need(i) <= Work
	while(true)
	{
		i = -1;
		auto it = std::find_if(finish.begin(), finish.end(), [&](bool finish_i)
					{
						i++;

						if (finish_i == true)
							return false;

						// Check Need(i) <= Work
						int j = -1;
						auto it = std::find_if(&need.at(numResources * i), &need.at(numResources * i) + numResources, [&](int need_ij)
							{
								j++;
								return need_ij > work.at(j); // If need_ij > work[j] for one resource, returns iterator for need.at(numResources * i + j). exception
							});

						bool need_satisfied = true ? (it == (&need.at(numResources * i) + numResources)) : false; // if the find_if returns last iterator, no exception found, Need(i) <= Work satisfied

						return need_satisfied;
					});

		
		if (it != finish.end()) // If Finish [i] == false and Need(i) <= Work found for an i
		{
			int finishedProcessId = it - finish.begin();
			*it = true;
			std::transform(work.begin(), work.end(), &allocation.at(numResources * finishedProcessId), work.begin(), std::plus<int>());
		}
		else // If Finish [i] == false and Need(i) <= Work not found for an i
		{
			break;
		}
	}

	
	/*std::cout << "\nFINAL VECTORS:\n";

	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");
	printMatrix<bool>(finish, 1, numProcesses, "finish");*/

	auto it_finish = std::find(finish.begin(), finish.end(), false);

	if (it_finish == finish.end())
	{
		std::cout << "All processes can terminate succesfully if allocation is made. So, allocation is servable!\n";
		return true;
	}
	else
	{
		std::cout << "All processes CANNOT terminate succesfully if allocation is made. So, allocation IS NOT servable!\n";
		return false;
	}
}

bool bankersAlgorithmParallel(int numProcesses, int numResources, const std::vector<int>& available_orig, const std::vector<int>& max, const std::vector<int>& allocation_orig, const std::vector<int>& need_orig, const std::vector<int>& request, int requestingProcessId)
{
	std::vector<int> available = available_orig;
	std::vector<int> allocation = allocation_orig;
	std::vector<int> need = need_orig;


	std::vector<int> need_orig_i(need_orig.begin() + numResources * requestingProcessId, need_orig.begin() + numResources * requestingProcessId + numResources); // mx1 need vector for requesting process
	auto isVector1LessThanVector2 = [&](const std::vector<int>& vector1, const std::vector<int>& vector2, int& invalidResourceIdx) {
		assert(vector1.size() == vector2.size());
		int i = -1;
		auto it = std::find_if(vector1.begin(), vector1.end(), [&](int vec1_i)
			{
				i++;
				return vec1_i > vector2.at(i); // Returns true if one element of vector1 is greater than vector2
			});

		if (it == vector1.end())
		{
			invalidResourceIdx = -1;
		}
		else
		{
			invalidResourceIdx = it - vector1.begin();
		}
	};

	/*VALIDATION STATE*/
	int invalidResourceIdxNeed, invalidResourceIdxAvailable;

	std::thread valNeed_th(isVector1LessThanVector2, std::cref(request), std::cref(need_orig_i), std::ref(invalidResourceIdxNeed));
	std::thread valAvailable_th(isVector1LessThanVector2, std::cref(request), std::cref(need_orig_i), std::ref(invalidResourceIdxAvailable));

	valNeed_th.join();
	valAvailable_th.join();

	if ( (invalidResourceIdxNeed != -1) || (invalidResourceIdxAvailable != -1) )
	{
		int invalidIdx = invalidResourceIdxNeed ? (invalidResourceIdxNeed != -1) : invalidResourceIdxAvailable;
		std::cout << "Invalid request for resource id: " << invalidIdx << ". Current matrix states for resource " << invalidIdx << ":\n"
			<< "Request[i]: " << request.at(invalidIdx) << "\nNeed[i]: " << need_orig_i.at(invalidIdx)
			<< "\nAvailable[i]: " << available.at(invalidIdx) << "\n\nSo, request is not feasible !!!\n";
		return false;
	}

	// If request is valid, go to modification state and modify the available, allocation, and need matrices as if the allocation was made

	/*MODIFICATION STATE*/
	auto modifyVectors = [&](std::vector<int>::iterator my_vector_begin, OP op) {
		int i = -1;
		if (op == OP::SUB) // SUBTRACTION IS MADE FOR NEED AND AVAILABLE VECTOR
		{
			std::for_each(request.begin(), request.end(), [&](int req_i)
				{
					i++;
					*(my_vector_begin + i) -= req_i;
				});
		}
		else // ADDITION IS MADE FOR ALLOCATION VECTOR
		{
			std::for_each(request.begin(), request.end(), [&](int req_i)
				{
					i++;
					*(my_vector_begin + i) += req_i;
				});
		}
	};

	std::thread modifyAvailable(modifyVectors, available.begin(), OP::SUB);
	std::thread modifyAllocation(modifyVectors, allocation.begin() + numResources * requestingProcessId, OP::ADD);
	std::thread modifyNeed(modifyVectors, need.begin() + numResources * requestingProcessId, OP::SUB);
	
	modifyAvailable.join();
	modifyAllocation.join();
	modifyNeed.join();


	/*std::cout << "MATRICES AFTER MODIFICATION: \n\n";

	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");*/

	// After modifying matrices, check whether resulting state is safe, i.e. deadlock-free

	/*SAFETY STATE*/
	// Initialize work matrix as work = available
	std::vector<int> work(available.begin(), available.end());
	std::vector<bool> finish(numProcesses, false);

	int i;
	// Find an i such that Finish [i] == false and Need(i) <= Work
	while(true)
	{
		i = -1;
		auto it = std::find_if(finish.begin(), finish.end(), [&](bool finish_i)
					{
						i++;

						if (finish_i == true)
							return false;

						// Check Need(i) <= Work
						int j = -1;
						auto it = std::find_if(&need.at(numResources * i), &need.at(numResources * i) + numResources, [&](int need_ij)
							{
								j++;
								return need_ij > work.at(j); // If need_ij > work[j] for one resource, returns iterator for need.at(numResources * i + j). exception
							});

						bool need_satisfied = true ? (it == (&need.at(numResources * i) + numResources)) : false; // if the find_if returns last iterator, no exception found, Need(i) <= Work satisfied

						return need_satisfied;
					});

		
		if (it != finish.end()) // If Finish [i] == false and Need(i) <= Work found for an i
		{
			int finishedProcessId = it - finish.begin();
			*it = true;
			std::transform(work.begin(), work.end(), &allocation.at(numResources * finishedProcessId), work.begin(), std::plus<int>());
		}
		else // If Finish [i] == false and Need(i) <= Work not found for an i
		{
			break;
		}
	}

	
	/*std::cout << "\nFINAL VECTORS:\n";

	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");
	printMatrix<bool>(finish, 1, numProcesses, "finish");*/

	auto it_finish = std::find(finish.begin(), finish.end(), false);

	if (it_finish == finish.end())
	{
		std::cout << "All processes can terminate succesfully if allocation is made. So, allocation is servable!\n";
		return true;
	}
	else
	{
		std::cout << "All processes CANNOT terminate succesfully if allocation is made. So, allocation IS NOT servable!\n";
		return false;
	}

	return false;
}

void readMatrices(std::vector<int>& available, std::vector<int>& max, std::vector<int>& allocation, std::vector<int>& need, std::vector<int>& request,
	int& numProcesses, int& numResources, int& requestingProcessId)
{
	std::ifstream info_file("info_1.txt");
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
	available.resize(numResources);
	max.resize(numProcesses * numResources);
	allocation.resize(numProcesses * numResources);
	need.resize(numProcesses * numResources);
	request.resize(numResources);


	std::ifstream available_file("available_1.txt");
	std::ifstream max_file("max_1.txt");
	std::ifstream allocation_file("allocation_1.txt");
	std::ifstream need_file("need_1.txt");
	std::ifstream request_file("request_1.txt");

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
void printMatrix(const std::vector<T>& matrix, int nRows, int nCols, const char* matrixName)
{
	//std::cout << "\n" << matrixName << ":\n";
	for (int i = 0; i < nRows; i++)
	{
		//std::cout << "[ ";
		for (int j = 0; j < nCols; j++)
		{
			//std::cout << matrix.at(i * nCols + j) << " ";
		}
		//std::cout << "]\n";
	}
}