import numpy as np
from random import randint

input_no = 1

numProcesses = 1000
numResources = 10000

maxResourceAmount = 20 # Maximum amount from a resource i. Resource amounts will be between [0, maxResourceAmount]

requestingProcessId = 0

info_file = open("info_" + str(input_no) + ".txt", "a+")

max_file = open("max_" + str(input_no) + ".txt", "a+")
allocation_file = open("allocation_" + str(input_no) + ".txt", "a+")
need_file = open("need_" + str(input_no) + ".txt", "a+")
available_file = open("available_" + str(input_no) + ".txt", "a+")
request_file = open("request_" + str(input_no) + ".txt", "a+")

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
    # Available resource amounts are set to be between [0, maxResourceAmount/2]. This is just for the sake of generality.
    # If it is between [0, maxResourceAmount], it may result in high available amounts and all processes finish easily most of the time.
    # It is done so in order to generate cases where request leads to unsafe states
    available_rnd = randint(maxResourceAmount, 10 * maxResourceAmount)
    available_line += str(available_rnd) + " "

    request_rnd = randint(0, min(available_rnd, need_vals[requestingProcessId * numResources + i])) # maxResourceAmount/6 is used similar reason to the above one, in order to increase chances of request being less than available.
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

print("Input file no: " + str(input_no) + "\nInput generation completed!")