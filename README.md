# Bankers-Algorithm-CUDA
CUDA Implementation of Banker's Algorithm for Deadlock Avoidance

Please read the following instructions carefully in order to execute the code. See [Technical Report](final_report.pdf) for technical details.

## Requirements:

* CUDA **5.0** or higher
* NVIDIA GPU with compute capability **3.5** or higher
* Windows environment (due to benchmarking functions using `<windows.h>`)

## Visual Studio Settings
For the GPU implementation we utilize Dynamic Parallelism. Programs with Dynamic Parallelism require additional project settings in the Visual Studio in order to be compiled and run:

1. Right click project in the solution explorer and go to **Properties**.
2. Under Configuration Properties -> CUDA C/C++ -> Common set **Generate Relocatable Device Code** to **Yes (-rdc=true)**
3. Under Configuration Properties -> CUDA C/C++ -> Device set **Code Generation** to **compute_35,sm_35**
4. Under Configuration Properties -> Linker -> Input -> Additional Dependencies add **cudadevrt.lib**
5. Make sure that execution settings (Debug/Release option) is the same as the configuration in the Properties window.

Source: https://stackoverflow.com/a/59383269/9817067

## Notes about execution of the program
* Program requires `allocation.txt`, `available.txt`, `max.txt`, `need.txt`, `request.txt`, and `info.txt` to be available in the directory of the executable. Modify the paths under `readMatrices()` function accordingly if you would like to store datasets in somewhere else.

* Datasets used during experiments are provided in the `DATASETS.rar` file. You can unzip that file, pick a dataset and copy 6 text files to the execution folder in order to test them. My experimental results of each implementation in all datasets are also provided in _results_ folder of each dataset.

* Note that depending on the dataset size, loading matrices from the disk to the memory may take a long time. Please be aware.

* There is a flag named **`multiThreaded`** in the beginning of the main function of CPU implementation. Set this flag to `true` if you would like to execute partially-parallel version of CPU implementation. When it is set to `false`, all operations are performed sequentially, in a single thread.

* Algorithm outputs whether the requested allocation is servable or not by printing a message to the terminal. In order to print some intermediate data structures you can uncomment necessary lines in the codes or add your own print statements.
