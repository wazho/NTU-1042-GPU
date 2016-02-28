#include <cstdio>
#include <cstdlib>
#include "../utils/SyncedMemory.h"

#define CHECK { auto e = cudaDeviceSynchronize(); if (e != cudaSuccess) { printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e)); abort(); } }

FILE *argumentsInit(int argc, char **argv);
__global__ void transformToUpperCase(char *input_gpu, int fsize);
__global__ void transformToSwapPairs(char *input_gpu, int fsize);

int main(int argc, char **argv) {
	// Initial auguments and set the file that prepare to read.
	FILE *fp = argumentsInit(argc, argv);

	// Get the total chars by file size.
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// Read file.
	MemoryBuffer<char> text(fsize + 1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// GPU settings.
	char *input_gpu = text_smem.get_gpu_rw();
	int gpu_dim   = 32;
	int gpu_times = fsize / gpu_dim + (fsize % gpu_dim ? 1 : 0);

	// HW01 start from here.
	int problem_num = 0;
	printf("\nPlease enter the problem number '1' or '2'. (Else input will leave program)\n");
	scanf("%d", &problem_num);

	// Problem 1.
	if (problem_num == 1)
		transformToUpperCase <<<gpu_times, gpu_dim>>>(input_gpu, fsize);
	// Problem 2.
	else if (problem_num == 2)
		transformToSwapPairs <<<gpu_times, gpu_dim>>>(input_gpu, fsize);
	else
		return 0;

	// Show the data from memory.
	printf("=============== Problem %d start ===============\n", problem_num);
	puts(text_smem.get_cpu_ro());
	printf("===============  Problem %d end  ===============\n\n\n", problem_num);

	// Recursion for this program.
	main(argc, argv);
}

// Open the file via argument.
FILE *argumentsInit(int argc, char **argv) {
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (! fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	return fp;
}

// Convert all characters to be captial.
__global__ void transformToUpperCase(char *input, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < fsize && input[idx] != '\n') {
		if (input[idx] >= 'a' && input[idx] <= 'z') {
			// 32 means the ASCII difference of 'A' to 'a'.
			input[idx] -= 32;
		}
	}
}

// Swap all pairs in all words.
__global__ void transformToSwapPairs(char *input, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Current char.
	char cur_c = input[idx];
	int idx_in_str = 0;
	// 
	if ((cur_c >= 'A' && cur_c <= 'Z') || (cur_c >= 'a' && cur_c <= 'z')) {
		// Get the index in the current string.
		for (int i = idx - 1; i >= 0; i--) {
			char c = input[i];
			if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
				idx_in_str++;
			} else {
				break;
			}
		}
		// Swap by string pairs.
		if (! (idx_in_str % 2) && idx < fsize) {
			char c = input[idx + 1];
			if (c >= 'A' && c <= 'Z') {
				if (cur_c >= 'a' && cur_c <= 'z')
					input[idx] = c + 32, input[idx + 1] = cur_c - 32;
				else
					input[idx] = c, input[idx + 1] = cur_c;
			} else if (c >= 'a' && c <= 'z') {
				if (cur_c >= 'a' && cur_c <= 'z')
					input[idx] = c, input[idx + 1] = cur_c;
				else
					input[idx] = c - 32, input[idx + 1] = cur_c + 32;
			}
		}
	}
}