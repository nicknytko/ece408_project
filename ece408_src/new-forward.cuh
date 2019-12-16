#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK_WIDTH 21
#define TILE_WIDTH (BLOCK_WIDTH-4)
    
/**
 * Forward convolution stage of the neural network.
 * @param y Input (batch size * output channels * y * x)
 * @param x Output data (batch size * input channels * y * x)
 * @param k Kernel Weights (output channels * input channels * y * x)
 * @param B Number of images in batch
 * @param M Number of features in each output
 * @param C Number of features in each input
 * @param H Image height
 * @param W Image width
 * @param K Filter mask width
 */
__global__ void forward_kernel(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int H_grid = H_out / TILE_WIDTH;
    const int W_grid = (W_out + (TILE_WIDTH - 1)) / TILE_WIDTH;

    (void) H_grid;
    
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define input y
#define output x
#define weights k
#define num_images B
#define num_output_features M
#define num_input_features C
#define num_output_elements H
#define image_width W
#define mask_width K

    /* Put implementation here */

    __shared__ float tile_data[BLOCK_WIDTH][BLOCK_WIDTH];
    
    const int image_idx = blockIdx.x;
    const int image_feature = blockIdx.y;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int image_y = (blockIdx.z / W_grid) * TILE_WIDTH + ty; /* Very crucial to working ! */ 
    const int image_x = (blockIdx.z % W_grid) * TILE_WIDTH + tx;
    const int convolve = (tx < TILE_WIDTH && ty < TILE_WIDTH);

    float acc = 0;
    for (int feature = 0; feature < num_input_features; feature++) {
        __syncthreads();
        
        /* Load values */
        float val = 0;
        if (image_x < W && image_y < H) {
            val = x4d(image_idx, feature, image_y, image_x);
        }
        tile_data[ty][tx] = val;
        
        __syncthreads();

        /* Do convolution step */
        if (convolve) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    acc += tile_data[ty + 0][tx + 0] * k4d(image_feature, feature, 0, 0);
                    acc += tile_data[ty + 0][tx + 1] * k4d(image_feature, feature, 0, 1);
                    acc += tile_data[ty + 0][tx + 2] * k4d(image_feature, feature, 0, 2);
                    acc += tile_data[ty + 0][tx + 3] * k4d(image_feature, feature, 0, 3);
                    acc += tile_data[ty + 0][tx + 4] * k4d(image_feature, feature, 0, 4);
                    acc += tile_data[ty + 1][tx + 0] * k4d(image_feature, feature, 1, 0);
                    acc += tile_data[ty + 1][tx + 1] * k4d(image_feature, feature, 1, 1);
                    acc += tile_data[ty + 1][tx + 2] * k4d(image_feature, feature, 1, 2);
                    acc += tile_data[ty + 1][tx + 3] * k4d(image_feature, feature, 1, 3);
                    acc += tile_data[ty + 1][tx + 4] * k4d(image_feature, feature, 1, 4);
                    acc += tile_data[ty + 2][tx + 0] * k4d(image_feature, feature, 2, 0);
                    acc += tile_data[ty + 2][tx + 1] * k4d(image_feature, feature, 2, 1);
                    acc += tile_data[ty + 2][tx + 2] * k4d(image_feature, feature, 2, 2);
                    acc += tile_data[ty + 2][tx + 3] * k4d(image_feature, feature, 2, 3);
                    acc += tile_data[ty + 2][tx + 4] * k4d(image_feature, feature, 2, 4);
                    acc += tile_data[ty + 3][tx + 0] * k4d(image_feature, feature, 3, 0);
                    acc += tile_data[ty + 3][tx + 1] * k4d(image_feature, feature, 3, 1);
                    acc += tile_data[ty + 3][tx + 2] * k4d(image_feature, feature, 3, 2);
                    acc += tile_data[ty + 3][tx + 3] * k4d(image_feature, feature, 3, 3);
                    acc += tile_data[ty + 3][tx + 4] * k4d(image_feature, feature, 3, 4);
                    acc += tile_data[ty + 4][tx + 0] * k4d(image_feature, feature, 4, 0);
                    acc += tile_data[ty + 4][tx + 1] * k4d(image_feature, feature, 4, 1);
                    acc += tile_data[ty + 4][tx + 2] * k4d(image_feature, feature, 4, 2);
                    acc += tile_data[ty + 4][tx + 3] * k4d(image_feature, feature, 4, 3);
                    acc += tile_data[ty + 4][tx + 4] * k4d(image_feature, feature, 4, 4);
                }
            }
        }
    }

    if (convolve && image_y < H_out && image_x < W_out) {
        y4d(image_idx, image_feature, image_y, image_x) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
#undef input
#undef output
#undef weights
#undef num_images
#undef num_output_features
#undef num_input_features
#undef num_output_elements
#undef image_width
#undef mask_width

}

#define UNROLL_THREADS 1024
    
__global__ void unroll_kernel(int C, int H, int W, int K, int b, float* X, float* X_unroll) {
    int c, s, h_out, w_out, w_base;
    int t = blockIdx.x * UNROLL_THREADS + threadIdx.x;
    int H_out = H - K + 1;
    
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

#define x4d(i3, i2, i1, i0) X[(i3) * (H * W * C) + (i2) * (H * W) + (i1) * (W) + i0]
#define xu2d(i1, i0) X_unroll[(i1) * (H_out * W_out) + i0]
    
    if (t < C * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll; h_out = s / W_out;
        
        w_out = s % W_out;
        int h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;
        
        for (int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                int w_unroll = w_base + p * K + q;
                xu2d(h_unroll, w_unroll) = x4d(b, c, h_out + p, w_out + q);
            }
        }
    }
}
    
void unroll_gpu(int C, int H, int W, int K, int b, float* X, float* X_unroll) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    //6int num_threads = C * H_out * W_out;
    int num_blocks = ceil((C * H_out * W_out) / UNROLL_THREADS);
    unroll_kernel<<<num_blocks, UNROLL_THREADS>>>(C, H, W, K, b, X, X_unroll);
}

#define GEMM_TILE_WIDTH 32
#define matSize(name) (sizeof(float) * num##name##Columns * num##name##Rows)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  __shared__ float subTileM[GEMM_TILE_WIDTH][GEMM_TILE_WIDTH];
  __shared__ float subTileN[GEMM_TILE_WIDTH][GEMM_TILE_WIDTH];
  
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * GEMM_TILE_WIDTH + ty;
  int col = bx * GEMM_TILE_WIDTH + tx;
  float p = 0.0;
  
  int m_bound = (numAColumns + GEMM_TILE_WIDTH - 1) / GEMM_TILE_WIDTH;
  for (int m = 0; m < m_bound; ++m) {
    float val1 = 0;
    float val2 = 0;
    
    if (m * GEMM_TILE_WIDTH + tx < numAColumns) {
      val1 = A[row * numAColumns + m * GEMM_TILE_WIDTH + tx];
    }
    if (m * GEMM_TILE_WIDTH + ty < numBRows) {
      val2 = B[(m * GEMM_TILE_WIDTH + ty) * numBColumns + col];
    }
    subTileM[ty][tx] = val1;
    subTileN[ty][tx] = val2;
    __syncthreads();
    
    for (int k = 0; k < GEMM_TILE_WIDTH; k++) {
      p += subTileM[ty][k] * subTileN[k][tx];
    }
    __syncthreads();
  }
  
  if (row < numCRows && col < numCColumns) {
    C[numCColumns * row + col] = p;
  }
}
 
/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; /* Batch Size */
    const int M = y.shape_[1]; /* Output Channels */
    const int C = x.shape_[1]; /* Input Channels */
    const int H = x.shape_[2]; /* Height */
    const int W = x.shape_[3]; /* Width */
    const int K = w.shape_[3]; /* Filter Size */
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int H_grid = (H_out + (TILE_WIDTH - 1)) / TILE_WIDTH;
    const int W_grid = (W_out + (TILE_WIDTH - 1)) / TILE_WIDTH;
    //const int Z = H_grid * W_grid;

    //dim3 gridDim(B, M, Z);
    //dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    //const int kernel_size = H * C * K * K * sizeof(float);
    //cudaMemcpyToSymbol(constFilter, w.dptr_, kernel_size);

    //printf("Convolution filter is size %d bytes.\n", kernel_size);
    //forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    const int W_unroll = C * K * K;
    const int H_unroll = H_out * W_out;
    float* X_unrolled;
    cudaMalloc(&X_unrolled, sizeof(float) * W_unroll * H_unroll);
    
    for (int n = 0; n < B; n++) {
        unroll_gpu(C, H, W, K, n, x.dptr_, X_unrolled);

        const int numARows = M;
        const int numACols = C * K * K;
        const int numBRows = C * K * K;
        const int numBCols = H_out * W_out;
        const int numCRows = M;
        const int numCCols = H_out * W_out;

        const int tile_width = 32;
        dim3 gridDim(ceil(numCCols / tile_width), ceil(numCRows / tile_width), 1);
        dim3 blockDim(tile_width, tile_width, 1);
        
        matrixMultiplyShared<<<gridDim, blockDim>>>(w.dptr_ + (C*H*W) * n,
                                                    X_unrolled,
                                                    y.dptr_ + (C*W_unroll*H_unroll),
                                                    numARows, numACols,
                                                    numBRows, numBCols,
                                                    numCRows, numCCols);
                             
                             
    }
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
