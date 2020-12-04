#pragma once

namespace NN {
	namespace CUDA {
		namespace AF {
			__global__ void sigmoid_compute(float* input, float* output, int layer_size);

			__global__ void sigmoid_derivative(float* input_derivative,
										  float* output, float* output_derivative,
										  int layer_size);

			//-----------------------------------------------------------------------------


			__global__ void tanh_compute(float* input, float* output, int layer_size);

			__global__ void tanh_derivative(float* input_derivative,
									   float* output, float* output_derivative,
									   int layer_size);

			//-----------------------------------------------------------------------------


			__global__ void relu_compute(float* input, float* output, int layer_size);

			__global__ void relu_derivative(float* input, float* input_derivative,
											float* output_derivative,
											int layer_size);

			//-----------------------------------------------------------------------------


			__global__ void leaky_relu_compute(float* input, float* output, int layer_size, float alpha);

			__global__ void leaky_relu_derivative(float* input, float* input_derivative,
												 float* output_derivative,
												 int layer_size, float alpha);

			//-----------------------------------------------------------------------------


			__global__ void sin_compute(float* input, float* output, int layer_size);

			__global__ void sin_derivative(float* input, float* input_derivative,
										   float* output_derivative,
										   int layer_size);
		}
	}
}