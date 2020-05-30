#include <iostream>
#include <random>
#include <iterator>
#include <algorithm>
#include <vector>

#include <thread>
#include <cassert>

enum class OP
{
	ADD,
	SUB
};


bool bankersAlgorithmSequential(unsigned int numProcesses, unsigned int numResources, const std::vector<int>& available_orig,
								const std::vector<int>& max, const std::vector<int>& allocation_orig, const std::vector<int>& need_orig,
								const std::vector<int>& request, unsigned int requestingProcessId);

bool bankersAlgorithmParallel(unsigned int numProcesses, unsigned int numResources, const std::vector<int>& available_orig,
							const std::vector<int>& max, const std::vector<int>& allocation_orig, const std::vector<int>& need_orig,
							const std::vector<int>& request, unsigned int requestingProcessId);

// Helper funtions
template<class T>
void printMatrix(const std::vector<T> & matrix, int nRows, int nCols, const char* matrixName)
{
	std::cout << "\n" << matrixName << ":\n";
	for (int i = 0; i < nRows; i++)
	{
		std::cout << "[ ";
		for (int j = 0; j < nCols; j++)
		{
			std::cout << matrix.at(i * nCols + j) << " ";
		}
		std::cout << "]\n";
	}
}


int main()
{

	bool multiThreaded = false;

	/*std::vector<int> available(numResources);
	std::vector<int> max(numProcesses * numResources);
	std::vector<int> allocation(numProcesses * numResources);
	std::vector<int> need(numProcesses * numResources);
	std::vector<int> request(numResources);*/


	// Memory allocations

	unsigned int numProcesses = 5;
	unsigned int numResources = 3;

	std::vector<int> available = {3, 3, 2};
	std::vector<int> max = {7, 5, 3,
							3, 2, 2,
							9, 0, 2,
							2, 2, 2,
							4, 3, 3 };
	std::vector<int> allocation = { 0, 1, 0,
									2, 0, 0,
									3, 0, 2,
									2, 1, 1,
									0, 0, 2 };
	std::vector<int> need(numProcesses * numResources);

	std::vector<int> request = {1, 0, 2};
	unsigned int requestingProcessId = 1;

	/*unsigned int numProcesses = 2;
	unsigned int numResources = 3;

	std::vector<int> available = { 1, 4, 1 };
	std::vector<int> max = { 1, 3, 1,
							 1, 4, 1};
	std::vector<int> allocation = { 0, 0, 0,
									0, 0, 0};
	std::vector<int> need(numProcesses * numResources);

	std::vector<int> request = { 1, 2, 0 };
	unsigned int requestingProcessId = 0;*/

	std::transform(max.begin(), max.end(), allocation.begin(), need.begin(), std::minus<int>());

	/* RANDOM INITIALIZATION*/

	//std::vector<int> available(numResources);
	//std::vector<int> max(numProcesses * numResources);
	//std::vector<int> allocation(numProcesses * numResources);
	//std::vector<int> need(numProcesses * numResources);

	//std::vector<int> request(numResources);
	//unsigned int requestingProcessId = 1;

	//int maxResourceAmount = 6;

	//std::random_device rd;
	//std::mt19937 mt(rd());
	//std::uniform_int_distribution<int> dist(0, maxResourceAmount);

	//// max matrix
	//for (int i = 0; i < numProcesses * numResources; i++)
	//{
	//	max.at(i) = dist(mt);
	//}
	//
	//// allocation and need matrices. allocation can't be greater than max. need = max - allocation
	//for (int i = 0; i < numProcesses * numResources; i++)
	//{
	//	dist.param(std::uniform_int_distribution<int>::param_type(0, max.at(i)));
	//	allocation.at(i) = dist(mt);
	//	need.at(i) = max.at(i) - allocation.at(i);
	//}
	//
	//for (int i = 0; i < numResources; i++)
	//{
	//	dist.param(std::uniform_int_distribution<int>::param_type(0, maxResourceAmount / 2));
	//	available[i] = dist(mt);
	//}

	/*END OF RANDOM INITIALIZATION*/

	printMatrix<int>(max, numProcesses, numResources, "max");
	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");
	printMatrix<int>(request, 1, numResources, "request");

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

	return 0;
}

bool bankersAlgorithmSequential(unsigned int numProcesses, unsigned int numResources, const std::vector<int>& available_orig,
								const std::vector<int>& max, const std::vector<int>& allocation_orig, const std::vector<int>& need_orig,
								const std::vector<int>& request, unsigned int requestingProcessId)
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


	std::cout << "MATRICES AFTER MODIFICATION: \n\n";

	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");

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

	
	std::cout << "\nFINAL VECTORS:\n";

	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");
	printMatrix<bool>(finish, 1, numProcesses, "finish");

	auto it_finish = std::find(finish.begin(), finish.end(), false);

	if (it_finish == finish.end())
	{
		std::cout << "All processes can terminate succesfully if allocation is made. So, allocation is servable!\n";
		return true;
	}
	else
	{
		std::cout << "All processes CAN NOT terminate succesfully if allocation is made. So, allocation IS NOT servable!\n";
		return false;
	}
}

bool bankersAlgorithmParallel(unsigned int numProcesses, unsigned int numResources, const std::vector<int>& available_orig, const std::vector<int>& max, const std::vector<int>& allocation_orig, const std::vector<int>& need_orig, const std::vector<int>& request, unsigned int requestingProcessId)
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


	std::cout << "MATRICES AFTER MODIFICATION: \n\n";

	printMatrix<int>(allocation, numProcesses, numResources, "allocation");
	printMatrix<int>(need, numProcesses, numResources, "need");
	printMatrix<int>(available, 1, numResources, "available");

	// After modifying matrices, check whether resulting state is safe, i.e. deadlock-free

	return false;
}
