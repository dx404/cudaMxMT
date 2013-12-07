CUDA Matrix Multiplication for M·Mᵀ
--------------------------------------------------------------------------------------------------
	-- A programming assignment for Parallel Computing (COMP 633 Fall 2013, UNC Chapel Hill)
	-- Author: Duo Zhao
	-- The report is under report/ directory

https://github.com/wisdompilot/cudaMxMT	

Project Description: 
--------------------------------------------------------------------------------------------------
	Project 4: Compute M·Mᵀ using CUDA 
	For a given n×n matrix M of floats, implement an efficient matrix product M·Mᵀ. Note that this 
	operation is included in the Cuda Basic Linear Algebra library (CUBLAS syrk). You should investigate
	whether you can build a better implementation based on our discussions of matrix multiplication and
	the related problem on the midterm exam. For this project be sure to report carefully on your
	implementation and the performance obtained as a function of n. Investigate the behavior as you 
	vary Cuda block and thread parameters. Compare your performance with the Cuda BLAS implementation. 

Building and Testing Environment 
-------------------------------------------------------------------------------
	The project is mostly written in CUDA C++ and build in NVCC compiler. The CUDA Tookit version is 5.5.
	The compatibility for PTX and GPU code are both 3.0. More building setups can be found in the Makefile
	under the project directory. 
	
	The code is tested on the supercomputing clusters, Stampede (https://www.tacc.utexas.edu/stampede/).
	The testing node for this project is gpudev. The architecure of the GPU on this code can found at 
	report/deviceQueryResult.txt for details. To run the code on the Stampede cluster, the cuda module is 
	expected to load before building and rebuilding (module load cuda). A interactive submit script 
	(run.sh) has been create for the ease of testing, or invoking from the make command (make run)

Project File Structures and Functionalities
-------------------------------------------------------------------------------		
	Makefile:
		For building the project, the default project is the current directory. To test different block
		sizes, BLOCK_DIM is expected to configure. The supported block size configuration can be found and
		extended at src/MxMTconst.h. The "run" label has the functionality to run the project. 

	run.sh:
		A simple script to submit the executable code to the gpu computing node
		
	src/main.cu:
		The main entry of the program. Each line requires 6 parameters from the stdin. 
		<impl_version>:
			An integer index refers to the implemented kernel functions declared src/cudaMxMT.cuh and 
			defined at src/cudaMxMT.cu
		<dim>
			The dimension of the problem. For this problem, it is the number of rows or columns for the 
			input square matrix
		
		<init_method> 
			The flag for how the input matrix is created.
			0 - each entry is a random number from 0 to 1
			1 - The linearized global position in the array modulo 1000
			<Other Integer> - Fill in all entries with zero. 
			Note: The corresponding function is at src/seqMatrix.cpp::matrixPopulate(float *m, int d, int method)
			
		<san_check>
			0 - No checking of the result from computation 
			1 - Checking the GPU computation result with a CPU sequential version of implementation
				on the same matrix
		
		<disp_matrix> 
			0 - No output of any matrices
			1 - Display the source and target matrices to the standard output

		<csv_format>
			0 - Show the performance information with attributes titles
			1 - Show a single line of the performance information that is comma separated
			
	src/cudaMxMT.cu:
		The major component of the project, which includes all the implementations of the matrix multiplication.
		The implementation detail can be found in the report/comp633pa2report.pdf. Note for the 2nd version, 
		cuda_MxMT_v002, only a matrix with multiples of block width is supported for computation. The 8th and 9th 
		version, cuda_MxMT_v008 and cuda_MxMT_v009, only up to 1024 dimension is supported. The raw data is stored
		at report/Aggregation.xlsx
		
	src/MxMTagent.cu:
		A object-oriented connector to invoke the the kernels in src/cudaMxMT.cu. It allocates the appropriate
		memory space in CPU and GPU and responsible for transferring the data between CPU and GPU. A timer records
		the time and performance at various points
		
	src/cuBLAS_MxMT.cu:
		A wrapper to compute the same problem by calling cuBLAS library function. 
		
	src/cudaGFLopTimer.cu:
		A simple class tailored for performance management
	
	src/seqMatrix.cu:
		The sequential implementation of the same problem as well as some helper functions for result comparison and 
		matrix initialization. 
	
	src/MxMTconst.h:
		The preprocessor for conditional building the project with different block sizes. 
		
