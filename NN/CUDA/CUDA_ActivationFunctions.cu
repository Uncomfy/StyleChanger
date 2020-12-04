#include <math.h>

#include "NN/CUDA/CUDA_ActivationFunctions.h"

namespace NN {
	namespace CUDA {
		namespace AF {
			__global__ void sigmoid_compute(float* input, float* output, int layer_size) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					output[i] = 1.0f / (1.0f + exp(-input[i]));
				}
			}

			__global__ void sigmoid_derivative(float* input_derivative,
										  float* output, float* output_derivative,
										  int layer_size) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					input_derivative[i] += output[i]*(1.0f - output[i]) * output_derivative[i];
				}
			}

			//-----------------------------------------------------------------------------


			__global__ void tanh_compute(float* input, float* output, int layer_size) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					output[i] = tanh(input[i]);
				}
			}

			__global__ void tanh_derivative(float* input_derivative,
									   float* output, float* output_derivative,
									   int layer_size) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					input_derivative[i] += (1.0f - output[i]*output[i]) * output_derivative[i];
				}
			}

			//-----------------------------------------------------------------------------


			__global__ void relu_compute(float* input, float* output, int layer_size) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
				}
			}

			__global__ void relu_derivative(float* input, float* input_derivative,
									   float* output_derivative,
									   int layer_size) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					input_derivative[i] += ((input[i] > 0.0f) ? 1.0f : 0.0f) * output_derivative[i];
				}
			}

			//-----------------------------------------------------------------------------


			__global__ void leaky_relu_compute(float* input, float* output, int layer_size, float alpha) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					output[i] = (input[i] > 0.0f) ? input[i] : alpha*input[i];
				}
			}

			__global__ void leaky_relu_derivative(float* input, float* input_derivative,
											 float* output_derivative,
											 int layer_size, float alpha) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					input_derivative[i] += ((input[i] > 0.0f) ? 1.0f : alpha) * output_derivative[i];
				}
			}

			//-----------------------------------------------------------------------------


			__global__ void sin_compute(float* input, float* output, int layer_size) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					output[i] = sin(input[i]);
				}
			}

			__global__ void sin_derivative(float* input, float* input_derivative,
									  float* output_derivative,
									  int layer_size) {
				int index = blockIdx.x * blockDim.x + threadIdx.x;
				int stride = blockDim.x * gridDim.x;

				for (int i = index; i < layer_size; i += stride) {
					input_derivative[i] += cos(input[i]) * output_derivative[i];
				}
			}
		}
	}
}