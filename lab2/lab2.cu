#include <cstdlib>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "lab2.h"

static const unsigned W = 1920;
static const unsigned H = 1080;
static const unsigned NFRAME = 600; // FPS = 20, time = 6 seconds, frames = 20 * 6 = 120

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

// Tree parameters.
#define TREE_COUNTS                  16
#define TREE_LENGTH                  220    // root y = 0
#define TREE_LENGTH_FARTHEST         30     // root y = H
#define TREE_LENGTH_DECAY            0.70
#define INITIAL_TREE_THICKNESS       7
#define TREE_THICKNESS_DECAY         2
#define TREE_BRANCHES                16     // root y = 0
#define TREE_BRANCHES_FARTHEST       12     // root y = H
#define TREE_ROOT_DEGREE             100.0  // Degree of the root to first branch
#define TREE_ROTATION_JITTER         0.35   // frame / 'JITTER' + 'PADDING_JITTER' = really rotating animation
#define TREE_ROTATION_PADDING_JITTER 0.23
#define TREE_ROTATION_SCALE          0.75

// Video parameters.
#define TRANSFORM_FRAME        40
#define JITTER_FRAME           300
#define GROWUP_FRAME           100
#define FRAME_OF_FOREST_APPEAR NFRAME / 4

// Macro functions.
#define abs(value) (value + (value >> sizeof(int) * CHAR_BIT - 1) ^ (value >> sizeof(int) * CHAR_BIT - 1))
#define powerSumBaseTwo(exp) ((1 << (exp + 1)) - 1)
#define toRadians(drgree) ((double) drgree * M_PI / 180)
#define swap(x, y, Type) do { Type tmp = x; x = y; y = tmp; } while (0)
#define toCanvasPos(x, y, mX, mY) ((mY - y) * mX + x)
#define getTreeLength(rootY, depth, scale) (((double) TREE_LENGTH_FARTHEST + (1.0 - ((double) rootY / H)) * (TREE_LENGTH - TREE_LENGTH_FARTHEST)) * pow(TREE_LENGTH_DECAY, depth) * scale)
#define getBranchCounts(rootY, scale) ((double) TREE_BRANCHES_FARTHEST + (1.0 - ((double) rootY / H)) * (TREE_BRANCHES - TREE_BRANCHES_FARTHEST) * scale)
#define jitterCoefficient(frame) ((frame % JITTER_FRAME >= JITTER_FRAME / 2 ? ((double) (JITTER_FRAME - frame % JITTER_FRAME) / JITTER_FRAME) : ((double) (frame % JITTER_FRAME) / JITTER_FRAME)) * TREE_ROTATION_JITTER + TREE_ROTATION_PADDING_JITTER)
#define growupCoefficient(frame, firstFrame) (frame - firstFrame >= GROWUP_FRAME ? 1 : (double) (frame - firstFrame) / GROWUP_FRAME)
#define getParent(idx) (ceil((double) idx / 2) - 1)

// Macro functions - YUV and RGB converting.
#define clip(x) ((x) > 255 ? 255 : (x) < 0 ? 0 : x)
#define RGB2Y(R, G, B) clip( ( 0.299 * R) + ( 0.587 * G) + ( 0.114 * B)      )
#define RGB2U(R, G, B) clip( (-0.169 * R) + (-0.331 * G) + ( 0.500 * B) + 128)
#define RGB2V(R, G, B) clip( ( 0.500 * R) + (-0.419 * G) + (-0.081 * B) + 128)

// All functions.
void buildFractalTree(uint8_t *yuv, int firstFrame, int currentFrame, int rootX, int rootY, int degree, float scale);
__device__ void drawLine(uint8_t *yuv, int x0, int y0, int x1, int y1, int thickness);
__device__ void treeBranch(curandState *state, int idx, uint8_t *yuv, int *nodes, double *rotations, int depth, int fframe, int frame, int direction, int degree, float scale);
__global__ void fractalTreeKernel(curandState *state, uint8_t *yuv, int *nodes, double *rotations, int fframe, int frame, int depth, int rootX, int rootY, int degree, float scale);

float gForest[TREE_COUNTS * 5] = {   960,   30,   1,  95, 1.25,    275,  120, 120,  85, 0.95,   1740,  150, 130, 100, 0.90,
                                     130,  635, 150,  85, 1.00,    520,  610, 170,  95, 1.15,    920,  690, 165,  95, 0.75,
                                    1300,  600, 145,  95, 0.85,   1630,  750, 175,  95, 0.82,   1890,  685, 170, 100, 1.05,
                                      10,  895, 220,  85, 1.00,    310,  900, 225, 105, 0.85,    550,  930, 195,  95, 1.10,
                                     750,  890, 220,  95, 1.05,   1010,  970, 190,  95, 0.90,   1330,  870, 200,  85, 1.00,
                                    1770,  965, 205,  90, 0.95
                                 };

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
    // Inital the background.
    cudaMemset(yuv, (impl->t) >= TRANSFORM_FRAME ? 255 : (impl->t) * 255 / TRANSFORM_FRAME, W * H);
    cudaMemset(yuv + W * H, 128, W * H / 2);
    // Random sampling to generate forest.
    for (int i = (TREE_COUNTS - 1) * 5; i >= 0; i -= 5)
        buildFractalTree(yuv, gForest[i+2], impl->t, gForest[i], gForest[i+1], gForest[i+3], gForest[i+4]);
    // Frame added.
    ++(impl->t);
}

void buildFractalTree(uint8_t *yuv, int firstFrame, int currentFrame, int rootX, int rootY, int degree, float scale) {
    if (firstFrame > currentFrame)
        return;
    // GPU settings.
    int threads = 32, blocks;
    curandState *deviceState;
    cudaMalloc(&deviceState, sizeof(curandState));
    // Nodes, ratations storage.
    int branches = (int) getBranchCounts(rootY, scale) * growupCoefficient(currentFrame, firstFrame);
    int *fractalTreeNodes;
    double *fractalTreeRotations;
    cudaMalloc(&fractalTreeNodes, 2 * sizeof(int) * powerSumBaseTwo(branches));
    cudaMalloc(&fractalTreeRotations, sizeof(double) * powerSumBaseTwo(branches));
    //
    for (int i = 0; i < branches; i++) {
        blocks = ceil(pow(2.0, i) / threads);
        fractalTreeKernel <<< blocks, threads >>> (deviceState, yuv, fractalTreeNodes, fractalTreeRotations, firstFrame, currentFrame, i, rootX, rootY, degree, scale);
    }
    // Release memory.
    cudaFree(deviceState);
    cudaFree(fractalTreeNodes);
    cudaFree(fractalTreeRotations);
}

// Draw one line using Bresenham's line algorithm.
__device__ void drawLine(uint8_t *yuv, int x0, int y0, int x1, int y1, int thickness) {
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
            // Set the YUV color.
            yuv[pos]       = RGB2Y(83, 53, 10);
            // yuv[pos/4+W*H] = RGB2U(83, 53, 10);
            // yuv[pos/4+W*H*5/4] = RGB2V(83, 53, 10);
        }
        // Add the error rate.
        error -= deltaY;
        if (error < 0)
            y += yStep, error += deltaX;
    }
}

__device__ void treeBranch(curandState *state, int idx, uint8_t *yuv, int *nodes, double *rotations, int depth, int fframe, int frame, int direction, int degree, float scale) {
    // float randJitter = (curand_uniform(state + idx) * 2 - 0.5) * TREE_ROTATION_JITTER;
    int parentIdx = getParent(idx);
    rotations[idx] = (degree / 2 * direction + rotations[parentIdx]) / 2;
    rotations[idx] += rotations[idx] * jitterCoefficient(frame);

    // Here need to fix. the problem about degree.
    // if (frame == 30) printf("[%d] idx:%d,  rotate: %f\n", frame, idx, rotations[idx]);


    double radian = toRadians(2 * rotations[idx]);
    nodes[2*idx]   = nodes[2*parentIdx]   + cos(radian) * getTreeLength(nodes[1], depth, scale) * growupCoefficient(frame, fframe);
    nodes[2*idx+1] = nodes[2*parentIdx+1] + sin(radian) * getTreeLength(nodes[1], depth, scale) * growupCoefficient(frame, fframe);
    drawLine(yuv, nodes[2*parentIdx], nodes[2*parentIdx+1], nodes[2*idx], nodes[2*idx+1], 1);
}

__global__ void fractalTreeKernel(curandState *state, uint8_t *yuv, int *nodes, double *rotations, int fframe, int frame, int depth, int rootX, int rootY, int degree, float scale) {
    // Index of thread.
    int idx = (powerSumBaseTwo(depth - 1)) + blockIdx.x * blockDim.x + threadIdx.x;
    // This thread can pass.
    if (idx >= powerSumBaseTwo(depth))
        return;
    // Draw line of branches.
    if (! depth) {
        rotations[0] = TREE_ROOT_DEGREE;
        nodes[0]     = rootX;
        nodes[1]     = rootY + getTreeLength(rootY, depth, scale) * growupCoefficient(frame, fframe);
        drawLine(yuv, rootX, rootY, nodes[2*idx], nodes[2*idx+1], INITIAL_TREE_THICKNESS);
    } else {
        treeBranch(state, idx, yuv, nodes, rotations, depth, fframe, frame, (idx % 2) ? 1 : -1, degree, scale);
    }
}