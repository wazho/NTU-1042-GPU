#include <cstdlib>
#include <iostream>
#include "lab2.h"

static const unsigned W = 1920;
static const unsigned H = 1080;
// static const unsigned W = 1000;
// static const unsigned H = 750;
// FPS = 20, time = 6 seconds, frames = 20 * 6 = 120
static const unsigned NFRAME = 400;

struct Lab2VideoGenerator::Impl {
    int t = 0;
};

// Tree parameters.
#define TREE_COUNTS            5
#define TREE_LENGTH            270    // root y = 0
#define TREE_LENGTH_FARTHEST   30     // root y = H
#define TREE_LENGTH_DECAY      0.75
#define INITIAL_TREE_THICKNESS 7
#define TREE_THICKNESS_DECAY   2
#define TREE_BRANCHES          17     // root y = 0
#define TREE_BRANCHES_FARTHEST 12     // root y = H
#define TREE_TWIG_FROM         0
#define TREE_LEAF_FROM         6

// Video parameters.
#define TRANSFORM_FRAME 40
#define JITTER_FRAME    100
#define JITTER_RATE     1.35
#define GROWUP_FRAME    150

// static const int COLOR_COLS = 3, COLOR_ROWS = 2;
// static const int g_treePalette[COLOR_COLS * COLOR_ROWS] = {
//     0x53350A, 0x53350A, 0x53350A,
//     0x0F1F00, 0x0F1F00, 0x0F1F00
// };

// [startFrame, rootX, rootY, theta, phi, scale, leafColor]
static const double g_forest[TREE_COUNTS * 7] = {
    1,  500,   10,  45.0,  40.0, 1.00, 1,
    1,  500,   10,  46.0,  40.0, 1.00, 2,
    1, 1500,  525,  45.0, -40.0, 0.80, 1,
    1, 1500,  525,  46.0, -40.0, 0.80, 2,
    1, 1500,  525,  49.0, -40.0, 0.80, 3
};

// Macro functions.
#define abs(value) (value + (value >> sizeof(int) * CHAR_BIT - 1) ^ (value >> sizeof(int) * CHAR_BIT - 1))
#define powerSumBaseTwo(exp) ((1 << (exp + 1)) - 1)
#define toRadians(drgree) ((double) drgree * M_PI / 180)
#define swap(x, y, Type) do { Type tmp = x; x = y; y = tmp; } while (0)
#define toCanvasPos(x, y, mX, mY) ((mY - y) * mX + x)
#define getTreeLength(rootY, depth, scale) (((double) TREE_LENGTH_FARTHEST + (1.0 - ((double) rootY / H)) * (TREE_LENGTH - TREE_LENGTH_FARTHEST)) * pow(TREE_LENGTH_DECAY, depth) * scale)
#define getBranchCounts(rootY, scale) ((double) TREE_BRANCHES_FARTHEST + (1.0 - ((double) rootY / H)) * (TREE_BRANCHES - TREE_BRANCHES_FARTHEST) * scale)
#define jitterCoefficient(frame) (JITTER_RATE * (frame % JITTER_FRAME >= JITTER_FRAME / 2 ? ((double) (JITTER_FRAME - frame % JITTER_FRAME) / JITTER_FRAME) : ((double) (frame % JITTER_FRAME) / JITTER_FRAME)))
#define growupCoefficient(frame, firstFrame) (frame - firstFrame >= GROWUP_FRAME ? 1 : (double) (frame - firstFrame) / GROWUP_FRAME)
#define getParent(idx) (ceil((double) idx / 2) - 1)

// Macro functions - YUV and RGB conversion.
#define clip(x) ((x) > 255 ? 255 : (x) < 0 ? 0 : x)
#define RGB2Y(R, G, B) clip( ( 0.299 * R) + ( 0.587 * G) + ( 0.114 * B)      )
#define RGB2U(R, G, B) clip( (-0.169 * R) + (-0.331 * G) + ( 0.500 * B) + 128)
#define RGB2V(R, G, B) clip( ( 0.500 * R) + (-0.419 * G) + (-0.081 * B) + 128)
#define HEX2R(hex) (hex >> 16 & 0xFF)
#define HEX2G(hex) (hex >>  8 & 0xFF)
#define HEX2B(hex) (hex       & 0xFF)

// Global variable RGB of Background image.
int *g_background;

// All functions.
void readImageFile(int *background);
void buildFractalTree(uint8_t *yuv, int firstFrame, int currentFrame, int rootX, int rootY, double theta, double phi, double scale, int leafColor);
__global__ void drawBackground(uint8_t *yuv, int *background);
__global__ void fractalTreeKernel(uint8_t *yuv, int *nodes, double *rotations, int fframe, int frame, int depth, int rootX, int rootY, double theta, double phi, double scale, int leafColor);
__device__ void treeBranch(int idx, uint8_t *yuv, int *nodes, double *rotations, int depth, int fframe, int frame, int direction, double theta, double phi, double scale, int leafColor);
__device__ void drawLine(uint8_t *yuv, int x0, int y0, int x1, int y1, int thickness, int color);

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
    // GPU settings.
    int threads = 32, blocks;
    cudaFree(0);
    // Inital the foundation of background.
    cudaMemset(yuv, 255, W * H);
    cudaMemset(yuv + W * H, 128, W * H / 2);
    // Set the background photo.
    if (impl->t == 1) {
        g_background = (int *) malloc(W * H * 3 * sizeof(int));
        readImageFile(g_background);
    }
    int *d_background;
    cudaMalloc(&d_background, W * H * 3 * sizeof(int));
    cudaMemcpy(d_background, g_background, W * H * 3 * sizeof(int), cudaMemcpyHostToDevice);
    blocks = ceil(W * H / threads);
    drawBackground <<< blocks, threads >>> (yuv, d_background);
    // Random sampling to generate forest.
    for (int i = (TREE_COUNTS - 1) * 7; i >= 0; i -= 7)
        buildFractalTree(yuv, g_forest[i], impl->t, g_forest[i+1], g_forest[i+2], g_forest[i+3], g_forest[i+4], g_forest[i+5], (int) g_forest[i+6]);
    // Frame added.
    ++(impl->t);
    // Release memory.
    cudaFree(d_background);
}

void readImageFile(int *background) {
    // Open the image file.
    FILE *streamIn = fopen("../resources/background.bmp", "r");
    if (streamIn == (FILE *) 0)
        exit(0);
    // Skip the header info in BMP.
    for (int i = 0; i < 54; i++)
        getc(streamIn);
    // Write RGB each pixel (left-bottom to right-top).
    for (int i = 0; i < W * H * 3; i += 3) {
        background[i+2] = getc(streamIn);
        background[i+1] = getc(streamIn);
        background[i]   = getc(streamIn);
    }
    fclose(streamIn);
    return;
}

__global__ void drawBackground(uint8_t *yuv, int *background) {
    // Index of thread (same with height-y).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // This thread can pass.
    if (idx >= W * H)
        return;
    int x = idx % W;
    int y = idx / W;
    int posY  = toCanvasPos(x, y, W, H);
    int posUV = toCanvasPos(x / 2, y / 2, W / 2, H / 2);
    // Set the YUV color.
    idx *= 3;
    if (! (background[idx] == 255 && background[idx + 1] == 0 && background[idx + 2] == 255)) {
        yuv[posY]                  = RGB2Y(background[idx], background[idx + 1], background[idx + 2]);
        yuv[posUV + W * H]         = RGB2U(background[idx], background[idx + 1], background[idx + 2]);
        yuv[posUV + W * H / 4 * 5] = RGB2V(background[idx], background[idx + 1], background[idx + 2]);
    }
}

void buildFractalTree(uint8_t *yuv, int firstFrame, int currentFrame, int rootX, int rootY, double theta, double phi, double scale, int leafColor) {
    if (firstFrame > currentFrame)
        return;
    // GPU settings.
    int threads = 32, blocks;
    // GPU memory storage.
    int branches = (int) getBranchCounts(rootY, scale) * growupCoefficient(currentFrame, firstFrame);
    // GPU memory - nodes.
    int *d_nodes;
    cudaMalloc(&d_nodes, sizeof(int) * powerSumBaseTwo(branches) * 2);
    cudaMemset(d_nodes, 0, sizeof(int) * powerSumBaseTwo(branches) * 2);
    // GPU memory - ratations.
    double *d_rotations;
    cudaMalloc(&d_rotations, sizeof(double) * powerSumBaseTwo(branches));
    cudaMemset(d_rotations, 0, sizeof(double) * powerSumBaseTwo(branches));
    // // GPU memory - treePalette.
    // int *d_treePalette;
    // cudaMalloc(&d_treePalette, sizeof(int) * COLOR_COLS * COLOR_ROWS);
    // cudaMemcpy(d_treePalette, g_treePalette, sizeof(int) * COLOR_COLS * COLOR_ROWS, cudaMemcpyHostToDevice);
    // Start generate tree.
    for (int i = 0; i < branches; i++) {
        blocks = ceil(pow(2.0, i) / threads);
        fractalTreeKernel <<< blocks, threads >>> (yuv, d_nodes, d_rotations, firstFrame, currentFrame, i, rootX, rootY, theta, phi, scale, leafColor);
        cudaDeviceSynchronize();
    }
    // Release memory.
    cudaFree(d_nodes);
    cudaFree(d_rotations);
    // cudaFree(d_treePalette);
}
__global__ void fractalTreeKernel(uint8_t *yuv, int *nodes, double *rotations, int fframe, int frame, int depth, int rootX, int rootY, double theta, double phi, double scale, int leafColor) {
    // Index of thread.
    int idx = (powerSumBaseTwo((depth - 1))) + blockIdx.x * blockDim.x + threadIdx.x;
    // This thread can pass.
    if (idx >= powerSumBaseTwo(depth))
        return;
    // Draw line of branches.
    if (! depth) {
        rotations[0] = 90;
        nodes[0]     = rootX;
        nodes[1]     = rootY + getTreeLength(rootY, depth, scale) * growupCoefficient(frame, fframe);
        // drawLine(yuv, rootX, rootY, nodes[2 * idx], nodes[2 * idx + 1], INITIAL_TREE_THICKNESS, palette[0]);
        drawLine(yuv, rootX, rootY, nodes[2 * idx], nodes[2 * idx + 1], INITIAL_TREE_THICKNESS, 0x53350A);
    } else {
        treeBranch(idx, yuv, nodes, rotations, depth, fframe, frame, (idx % 2) ? 1 : -1, theta, phi, scale, leafColor);
    }
}

__device__ void treeBranch(int idx, uint8_t *yuv, int *nodes, double *rotations, int depth, int fframe, int frame, int direction, double theta, double phi, double scale, int leafColor) {
    // pIdx is parentIdx.
    int pIdx = getParent(idx);
    double radian;
    int color;
    // Store the degree of this rotation.
    rotations[idx] = rotations[pIdx] + theta * direction + phi;
    rotations[idx] += jitterCoefficient(frame);
    // Store the position of the end point.
    radian = toRadians(rotations[idx]);
    nodes[2 * idx]     = nodes[2 * pIdx]     + cosf(radian) * getTreeLength(nodes[1], depth, scale) * growupCoefficient(frame, fframe);
    nodes[2 * idx + 1] = nodes[2 * pIdx + 1] + sinf(radian) * getTreeLength(nodes[1], depth, scale) * growupCoefficient(frame, fframe);
    // Draw line.
    // if (depth >= TREE_LEAF_FROM) {
        if (leafColor == 1)
            color = 0x0F1F00;
        if (leafColor == 2)
            color = 0x477900;
        // if (leafColor == 3)
        //     color = 0xAD9E00;
    // } else {
    //     color = 0x53350A;
    // }
    drawLine(yuv, nodes[2 * pIdx], nodes[2 * pIdx + 1], nodes[2 * idx], nodes[2 * idx + 1], 1, color);
}

// Draw one line using Bresenham's line algorithm.
__device__ void drawLine(uint8_t *yuv, int x0, int y0, int x1, int y1, int thickness, int color) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    // Get the RGB color.
    int r = HEX2R(color);
    int g = HEX2G(color);
    int b = HEX2B(color);
    // Draw the points by loop.
    int deltaX = x1 - x0;
    int deltaY = abs(y1 - y0);
    int error  = deltaX / 2;
    int yStep  = (y0 < y1) ? 1 : -1;
    for (int x = x0, y = y0; x <= x1; x++) {
        // Draw the point if it is in bound.
        if (! ((steep && (y >= W || x >= H)) || (! steep && (x >= W || y >= H)))) {
            int posY  = (steep) ? toCanvasPos(y, x, W, H) : toCanvasPos(x, y, W, H);
            int posUV = (steep) ? toCanvasPos(y / 2, x / 2, W / 2, H / 2) : toCanvasPos(x / 2, y / 2, W / 2, H / 2);
            // Set the YUV color.
            yuv[posY]                  = RGB2Y(r, g, b);
            yuv[posUV + W * H]         = RGB2U(r, g, b);
            yuv[posUV + W * H / 4 * 5] = RGB2V(r, g, b);
        }
        // Add the error rate.
        error -= deltaY;
        if (error < 0)
            y += yStep, error += deltaX;
    }
}

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) { }

Lab2VideoGenerator::~Lab2VideoGenerator() { }

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
    info.w       = W;
    info.h       = H;
    info.n_frame = NFRAME;
    info.fps_n   = 20;
    info.fps_d   = 1;
};
