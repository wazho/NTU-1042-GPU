#include <iostream>
#include <cstdio>

#include "lab3.h"
#include "../utils/Timer.h"

// It meas [min, max).
#define isInRange(value, min, max) (min <= value && value < max)

enum color { RR = 0, GG = 1, BB = 2 };

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

	for (int i = 0; i < 10000; i++) {
		JacobiMethod <<<gdim, bdim>>> (fixed, mask, buf1, buf2, wt, ht);
		JacobiMethod <<<gdim, bdim>>> (fixed, mask, buf2, buf1, wt, ht);
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
	if (xt >= wt or yt >= ht or mask[curT] < 127.0f)
		return;

	float result[3];
	result[RR] = 4 * target[curT * 3 + RR];
	result[GG] = 4 * target[curT * 3 + GG];
	result[BB] = 4 * target[curT * 3 + BB];

	const int dir[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
	for (int i = 0; i < 4; ++i) {
		const int nearX = xt + dir[i][0];
		const int nearY = yt + dir[i][1];
		const int curTN = wt * nearY + nearX;
		if (isInRange(nearY, 0, ht) && isInRange(nearX, 0, wt)) {
			result[RR] -= target[curTN * 3 + RR];
			result[GG] -= target[curTN * 3 + GG];
			result[BB] -= target[curTN * 3 + BB];
		} else {
			result[RR] -= target[curT * 3 + RR];
			result[GG] -= target[curT * 3 + GG];
			result[BB] -= target[curT * 3 + BB];
		}

		if (! (isInRange(nearY, 0, ht) && isInRange(nearX, 0, wt)) or mask[curTN] < 127.0f) {
			const int curB = wb * (nearY + oy) + (nearX + ox);
			result[RR] += background[curB * 3 + RR];
			result[GG] += background[curB * 3 + GG];
			result[BB] += background[curB * 3 + BB];
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
	if (xt >= wt or yt >= ht or mask[curT] < 127.0f)
		return;

	float result[3];
	result[RR] = fixed[curT * 3 + RR];
	result[GG] = fixed[curT * 3 + GG];
	result[BB] = fixed[curT * 3 + BB];

	const int dir[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
	for (int i = 0; i < 4; ++i) {
		const int nearX = xt + dir[i][0];
		const int nearY = yt + dir[i][1];
		const int curTN = wt * nearY + nearX;
		if (isInRange(nearX, 0, wt) && isInRange(nearY, 0, ht) && mask[curTN] > 127.0f) {
			result[RR] += target[curTN * 3 + RR];
			result[GG] += target[curTN * 3 + GG];
			result[BB] += target[curTN * 3 + BB];
		}
	}

	output[curT * 3 + RR] = result[RR] / 4;
	output[curT * 3 + GG] = result[GG] / 4;
	output[curT * 3 + BB] = result[BB] / 4;
}

__global__ void ImageBlending(const float *background, const float *target, const float *mask, float *output, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox) {
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int curT = wt * yt + xt;
	if (yt < ht and xt < wt and mask[curT] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int curB = (wb * yb + xb) * 3;
		if (isInRange(yb, 0, hb) && isInRange(xb, 0, wb)) {
			output[curB + RR] = target[curT * 3 + RR];
			output[curB + GG] = target[curT * 3 + GG];
			output[curB + BB] = target[curT * 3 + BB];
		}
	}
}