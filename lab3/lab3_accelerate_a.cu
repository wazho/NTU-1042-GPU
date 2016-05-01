#include <iostream>
#include <cstdio>

#include "lab3.h"
#include "../utils/Timer.h"

enum color { RR = 0, GG = 1, BB = 2 };

#define SAMPLE_POINT_COUNT 3
#define swap(x, y, Type) do { Type tmp = x; x = y; y = tmp; } while (0)
#define isInRange(value, min, max) (min <= value && value < max)

void PoissonImageCloning(const float *background, const float *target, const float *mask, float *output, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox);
__global__ void CalculateFixed(const float *background, const float *target, const float *mask, float *output, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox);
__global__ void JacobiMethod(const float *fixed, const float *mask, const float *target, float *output, const int wt, const int ht);
__global__ void ImageBlending(const float *background, const float *target, const float *mask, float *output, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox);

void PoissonImageCloning(const float *background, const float *target, const float *mask, float *output, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox) {
	// Set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, sizeof(float) * 3 * wt * ht);
	cudaMalloc(&buf1,  sizeof(float) * 3 * wt * ht);
	cudaMalloc(&buf2,  sizeof(float) * 3 * wt * ht);

	// Start the timer.
	cudaDeviceSynchronize();
	Timer timer;
	timer.Start();
 
	// Initialize the iteration.
	dim3 gdim(((wt - 1) / 32 + 1), ((ht - 1) / 16 + 1));
	dim3 bdim(32, 16);
	CalculateFixed <<<gdim, bdim>>> (background, target, mask, fixed, wb, hb, wt, ht, oy, ox);
	cudaMemcpy(buf1, target, sizeof(float) * 3 * wt * ht, cudaMemcpyDeviceToDevice);

	for (int i = 0; i < 6000; i++) {
		JacobiMethod <<<gdim, bdim>>> (fixed, mask, buf1, buf2, wt, ht);
		swap(buf1, buf2, float *);
	}

	// Based background.
	cudaMemcpy(output, background, sizeof(float) * 3 * wb * hb, cudaMemcpyDeviceToDevice);
	ImageBlending <<<gdim, bdim>>> (background, buf1, mask, output, wb, hb, wt, ht, oy, ox);

	// Stop the time and print the time.
	cudaDeviceSynchronize();
	timer.Pause();
	printf_timer(timer);

	// Release memory.
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}

__global__ void CalculateFixed(const float *background, const float *target, const float *mask, float *output, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox) {
	const int xt = blockDim.x * blockIdx.x + threadIdx.x;
	const int yt = blockDim.y * blockIdx.y + threadIdx.y;
	const int curT = wt * yt + xt;
	if (xt >= wt || yt >= ht || mask[curT] < 127.0f)
		return;

	float result[3];
	result[RR] = SAMPLE_POINT_COUNT * target[curT * 3 + RR];
	result[GG] = SAMPLE_POINT_COUNT * target[curT * 3 + GG];
	result[BB] = SAMPLE_POINT_COUNT * target[curT * 3 + BB];

	const int dir[SAMPLE_POINT_COUNT][2] = {{0, 2}, {-1, -1}, {1, -1}};
	for (int i = 0; i < SAMPLE_POINT_COUNT; i++) {
		const int nearX = xt + dir[i][0];
		const int nearY = yt + dir[i][1];
		const int curTN = wt * nearY + nearX;
		const int curIndex = ((isInRange(nearY, 0, ht) && isInRange(nearX, 0, wt)) ? curTN : curT) * 3;
		// Subtract the value of N, W, S and E directions.
		result[RR] -= target[curIndex + RR];
		result[GG] -= target[curIndex + GG];
		result[BB] -= target[curIndex + BB];
		// Add background value.
		if (mask[curTN] < 127.0f || ! (isInRange(nearY, 0, ht) && isInRange(nearX, 0, wt))) {
			const int curBkgIndex = (wb * (nearY + oy) + (nearX + ox)) * 3;
			result[RR] += background[curBkgIndex + RR];
			result[GG] += background[curBkgIndex + GG];
			result[BB] += background[curBkgIndex + BB];
		}
	}

	output[curT * 3 + RR] = result[RR];
	output[curT * 3 + GG] = result[GG];
	output[curT * 3 + BB] = result[BB];
}

__global__ void JacobiMethod(const float *fixed, const float *mask, const float *target, float *output, const int wt, const int ht) {
	const int xt = blockDim.x * blockIdx.x + threadIdx.x;
	const int yt = blockDim.y * blockIdx.y + threadIdx.y;
	const int curT = wt * yt + xt;
	if (xt >= wt || yt >= ht || mask[curT] < 127.0f)
		return;

	float result[3];
	result[RR] = fixed[curT * 3 + RR];
	result[GG] = fixed[curT * 3 + GG];
	result[BB] = fixed[curT * 3 + BB];

	const int dir[SAMPLE_POINT_COUNT][2] = {{0, 2}, {-1, -1}, {1, -1}};
	for (int i = 0; i < SAMPLE_POINT_COUNT; i++) {
		const int nearX = xt + dir[i][0];
		const int nearY = yt + dir[i][1];
		const int curTN = wt * nearY + nearX;
		if (isInRange(nearX, 0, wt) && isInRange(nearY, 0, ht) && mask[curTN] > 127.0f) {
			result[RR] += target[curTN * 3 + RR];
			result[GG] += target[curTN * 3 + GG];
			result[BB] += target[curTN * 3 + BB];
		}
	}

	output[curT * 3 + RR] = result[RR] / SAMPLE_POINT_COUNT;
	output[curT * 3 + GG] = result[GG] / SAMPLE_POINT_COUNT;
	output[curT * 3 + BB] = result[BB] / SAMPLE_POINT_COUNT;
}

__global__ void ImageBlending(const float *background, const float *target, const float *mask, float *output, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox) {
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int curT = wt * yt + xt;
	if (yt < ht && xt < wt && mask[curT] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int curB = (wb * yb + xb) * 3;
		if (isInRange(yb, 0, hb) && isInRange(xb, 0, wb)) {
			output[curB + RR] = target[curT * 3 + RR];
			output[curB + GG] = target[curT * 3 + GG];
			output[curB + BB] = target[curT * 3 + BB];
		}
	}
}