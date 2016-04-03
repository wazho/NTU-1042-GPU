#include <cstdlib>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "lab2.h"

static const unsigned W = 1920;
static const unsigned H = 1080;
static const unsigned NFRAME = 240; // FPS = 20, time = 6 seconds, frames = 20 * 6 = 120

struct Lab2VideoGenerator::Impl {
    int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
    info.w = W;
    info.h = H;
    info.n_frame = NFRAME;
    info.fps_n = 20;
    info.fps_d = 1;
};

/* Start from here */

// Tree value
#define TREE_COUNTS           20
#define INITIAL_TREE_LENGTH   140    // root y = 0
#define TREE_LENGTH_FARTHEST  30     // root y = H
#define TREE_LENGTH_DECAY     0.67
#define TREE_BRANCHES         14
#define TREE_ROTATION_DEGREE  105.0
#define TREE_ROTATION_SCALE   0.75

// Video value
#define TRANSFORM_FRAME 40
#define GROWUP_FRAME    100

#define abs(value) (value + (value >> sizeof(int) * CHAR_BIT - 1) ^ (value >> sizeof(int) * CHAR_BIT - 1))
#define powerSumBaseTwo(exp) ((1 << (exp + 1)) - 1)
#define toRadians(drgree) ((double) drgree * M_PI / 180)
#define swap(x, y, Type) do { Type tmp = x; x = y; y = tmp; } while (0)
#define toCanvasPos(x, y, mX, mY) ((mY - y) * mX + x)
#define getTreeLength(rootY, depth) (((double) TREE_LENGTH_FARTHEST + (1.0 - ((double) rootY / H)) * (INITIAL_TREE_LENGTH - TREE_LENGTH_FARTHEST)) * pow(TREE_LENGTH_DECAY, depth))
#define growupCoefficient(frame) (frame >= GROWUP_FRAME ? 1 : (double) frame / GROWUP_FRAME)
#define getParent(idx) (ceil((double) idx / 2) - 1)

// Draw one line using Bresenham's line algorithm.
__device__ void drawLine(uint8_t *yuv, int x0, int y0, int x1, int y1) {
    // If the line is steep, then swap two points.
    bool steep = abs(y1 - y0) > abs(x1 - x0);
    if (steep) {
        swap(x0, y0, int);
        swap(x1, y1, int);
    }
    if (x0 > x1) {
        swap(x0, x1, int);
        swap(y0, y1, int);
    }
    // Draw the points by loop.
    int deltaX = x1 - x0;
    int deltaY = abs(y1 - y0);
    int error  = deltaX / 2;
    int yStep  = (y0 < y1) ? 1 : -1;
    for (int x = x0, y = y0; x <= x1; x++) {
        // Draw the point if it is in bound.
        if (! ((steep && (y >= W || x >= H)) || (! steep && (x >= W || y >= H)))) {
            int pos = (steep) ? toCanvasPos(y, x, W, H) : toCanvasPos(x, y, W, H);
            yuv[pos] = 0;
        }
        // Add the error rate.
        error -= deltaY;
        if (error < 0)
            y += yStep, error += deltaX;
    }
}

__device__ void treeBranch(curandState *state, int idx, uint8_t *yuv, int *nodes, double *rotations, int depth, int frame, int direction) {
    // float randJitter = (curand_uniform(state + idx) - 0.5) / 2;
    int parentIdx = getParent(idx);
    rotations[idx] = (TREE_ROTATION_DEGREE / 2 * direction + rotations[parentIdx]) / 2;
    double radian = toRadians(2 * rotations[idx]);
    nodes[2*idx]   = nodes[2*parentIdx]   + cos(radian) * getTreeLength(nodes[1], depth) * growupCoefficient(frame);
    nodes[2*idx+1] = nodes[2*parentIdx+1] + sin(radian) * getTreeLength(nodes[1], depth) * growupCoefficient(frame);
    drawLine(yuv, nodes[2*parentIdx], nodes[2*parentIdx+1], nodes[2*idx], nodes[2*idx+1]);
}

__global__ void fractalTreeKernel(curandState *state, uint8_t *yuv, int *nodes, double *rotations, int frame, int depth, int rootX, int rootY) {
    // Index of thread.
    int idx = (powerSumBaseTwo(depth - 1)) + blockIdx.x * blockDim.x + threadIdx.x;
    // This thread can pass.
    if (idx >= powerSumBaseTwo(depth))
        return;
    // Draw line of branches.
    if (! depth) {
        rotations[0] = 90.0;
        nodes[0]     = rootX;
        nodes[1]     = rootY + getTreeLength(rootY, depth) * growupCoefficient(frame);
        drawLine(yuv, rootX, rootY, nodes[2*idx], nodes[2*idx+1]);
    } else {
        treeBranch(state, idx, yuv, nodes, rotations, depth, frame, (idx % 2) ? 1 : -1);
    }
}

void buildFractalTree(uint8_t *yuv, int currentFrame, int rootX, int rootY) {
    // GPU settings.
    int threads = 32, blocks;
    curandState *deviceState;
    cudaMalloc(&deviceState, sizeof(curandState));
    // 
    int *fractalTreeNodes;
    double *fractalTreeRotations;
    cudaMalloc(&fractalTreeNodes, 2 * sizeof(int) * powerSumBaseTwo(TREE_BRANCHES));
    cudaMalloc(&fractalTreeRotations, sizeof(double) * powerSumBaseTwo(TREE_BRANCHES));
    //
    for (int i = 0; i < TREE_BRANCHES; i++) {
        blocks = ceil(pow(2.0, i) / threads);
        fractalTreeKernel <<< blocks, threads >>> (deviceState, yuv, fractalTreeNodes, fractalTreeRotations, currentFrame, i, rootX, rootY);
    }
    // Release memory.
    cudaFree(deviceState);
    cudaFree(fractalTreeNodes);
    cudaFree(fractalTreeRotations);
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
    // Inital the background.
    cudaMemset(yuv, (impl->t) >= TRANSFORM_FRAME ? 255 : (impl->t) * 255 / TRANSFORM_FRAME, W * H);
    cudaMemset(yuv + W * H, 128, W * H / 2);
    // Random sampling.
    srand(TREE_COUNTS);
    for (int i = 0; i < TREE_COUNTS; i++) {
        int rootX = rand() % W;
        int rootY = rand() % H;
        buildFractalTree(yuv, impl->t, rootX, rootY);
    }
    // Frame added.
    ++(impl->t);
}