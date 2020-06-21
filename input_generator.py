import numpy as np
from random import randint

numProcesses = 1024
numResources = 4096

maxResourceAmount = 20 # Maximum amount from a resource i. Resource amounts will be between [0, maxResourceAmount]

requestingProcessId = 0

info_file = open("info.txt", "a+")

max_file = open("max.txt", "a+")
allocation_file = open("allocation.txt", "a+")
need_file = open("need.txt", "a+")
available_file = open("available.txt", "a+")
request_file = open("request.txt", "a+")

info_line = ""
info_line += "numProcesses: " + str(numProcesses)
info_line += "\n" + "numResources: " + str(numResources)
info_line += "\n" + "requestingProcessId: " + str(requestingProcessId)
info_line += "\n" + "maxResourceAmount: " + str(maxResourceAmount)

need_vals = []

max_line = ""
allocation_line = ""
need_line = ""
available_line = ""
request_line = ""

# Generate random resource amounts for max, allocation, available and request matrices

for i in range(numProcesses):
    for j in range(numResources):
        max_rnd = randint(0, maxResourceAmount)
        max_line += str(max_rnd) + " "

        allocation_rnd = randint(0, max_rnd) # Allocated resource amount for a process can't be greater than max needs of the process for that resource
        allocation_line += str(allocation_rnd) + " "

        need_rnd = max_rnd - allocation_rnd # need_ij = max_ij - allocation_ij for a process i and resource j
        need_line += str(need_rnd) + " "
        need_vals.append(need_rnd)
    
    # Move to next line in text files for the next process
    max_line += "\n"
    allocation_line += "\n"
    need_line += "\n"
    

for i in range(numResources):
    # available_rnd = randint(0, maxResourceAmount)
    available_rnd = randint(maxResourceAmount, 2 * maxResourceAmount)
    available_line += str(available_rnd) + " "

    request_rnd = randint(0, min(available_rnd, need_vals[requestingProcessId * numResources + i])) # Make sure elements of request vector is less than that of available and need
    
    # if(i == 0):
    #     request_rnd = available_rnd + 1 # Make first element of Request vector greater than firts element of Available (Dataset 1 - Test Case 1)

    # if(i == numResources / 2):
    #     request_rnd = available_rnd + 1 # Make middle element of Request vector greater than middle element of Available (Dataset 1 - Test Case 2)

    # if(i == numResources - 1):
    #     request_rnd = available_rnd + 1 # Make last element of Request vector greater than last element of Available (Dataset 1 - Test Case 3)
    
    request_line += str(request_rnd) + " "


# Write strings to the text files
info_file.write(info_line)

max_file.write(max_line)
allocation_file.write(allocation_line)
need_file.write(need_line)
available_file.write(available_line)
request_file.write(request_line)

# Close text files
info_file.close()

max_file.close()
allocation_file.close()
need_file.close()
available_file.close()
request_file.close()

print("Input generation completed!")