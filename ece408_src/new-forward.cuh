
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK_WIDTH 20
#define TILE_WIDTH 16
    
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
    const int convolve = (tx >= 0 && tx < TILE_WIDTH && ty >= 0 && ty < TILE_WIDTH);

    float acc = 0;
    for (int feature = 0; feature < num_input_features; feature++) {
        __syncthreads();
        
        /* Load values */
        float val = 0;
        if (image_x < image_width && image_y < image_width) {
            val = x4d(image_idx, feature, image_y, image_x);
        }
        tile_data[ty][tx] = val;
        
        __syncthreads();

        /* Do convolution step */
        if (convolve) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    acc += tile_data[ty + i][tx + j] * k4d(image_feature, feature, i, j);
                }
            }
        }
    }

    if (convolve && image_y < image_width && image_x < image_width) {
        y4d(image_idx, image_feature, image_y, image_x) = acc;
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
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int H_grid = (H_out + (TILE_WIDTH - 1)) / TILE_WIDTH;
    const int W_grid = (W_out + (TILE_WIDTH - 1)) / TILE_WIDTH;
    const int Z = H_grid * W_grid;

    dim3 gridDim(B, M, Z);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    //const int kernel_size = H * C * K * K * sizeof(float);
    //cudaMemcpyToSymbol(constFilter, w.dptr_, kernel_size);

    //printf("Convolution filter is size %d bytes.\n", kernel_size);
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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
