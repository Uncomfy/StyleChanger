#include <algorithm>
#include "CUDA_func.h"

namespace NN {
	namespace CUDA {
		__global__ void compute_dense_layer(float* input, float* parameters, float* output, int input_size, int output_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int c = index; c < output_size; c += stride) {
				output[c] = 0;
				for (int i = 0; i < input_size; i++) {
					output[c] += input[i] * parameters[c * (input_size + 1) + i];
				}
				output[c] += parameters[c * (input_size + 1) + input_size];
			}
		}


		__global__ void backprop_dense_layer_input_gradient(float* input, float* input_gradient, float* parameters, float* gradient, float* output_gradient, int input_size, int output_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int c = index; c < input_size; c += stride) {
				for (int i = 0; i < output_size; i++) {
					input_gradient[c] += parameters[i * (input_size + 1) + c] * output_gradient[i];
					gradient[i * (input_size + 1) + c] += input[c] * output_gradient[i];
				}
			}
		}

		__global__ void backprop_dense_layer_bias(float* gradient, float* output_gradient, int input_size, int output_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int c = index; c < output_size; c += stride) {
				gradient[c * (input_size + 1) + input_size] += output_gradient[c];
			}
		}

		//-------------------------------------------------------------------------------------------

		__global__ void compute_conv_layer(float* input, float* parameters, float* output,
										int input_width, int input_height, int input_depth,
										int layer_width, int layer_height, int layer_depth,
										int neuron_width, int neuron_height,
										int input_size, int output_size, int layer_size, int neuron_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int weights_offset;
			int input_pos;
			int x, y, z;

			for (int c = index; c < output_size; c += stride) {
				output[c] = 0.0f;

				x = c % layer_width;
				y = (c / layer_width) % layer_height;
				z = c / (layer_width * layer_height);

				weights_offset = z * (neuron_size + 1);

				for (int k = 0; k < input_depth; k++) {
					for (int j = 0; j < neuron_height; j++) {
						for (int i = 0; i < neuron_width; i++) {
							input_pos = ((k * input_height) + (y+j))*input_width + (x+i);
							output[c] += input[input_pos] * parameters[weights_offset + (((k * neuron_height) + j) * neuron_width) + i];
						}
					}
				}

				output[c] += parameters[weights_offset + neuron_size];
			}
		}

		__global__ void backprop_conv_layer_input(float* input, float* input_gradient,
												float* parameters, float* gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int layer_width, int layer_height, int layer_depth,
												int neuron_width, int neuron_height,
												int input_size, int output_size, int layer_size, int neuron_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int weights_offset;
			int output_pos;
			int x, y, z;

			for (int c = index; c < input_size; c += stride) {
				x = c % input_width;
				y = (c / input_width) % input_height;
				z = (c / (input_width * input_height)) % input_depth;

				weights_offset = z * (neuron_width * neuron_height);

				for (int k = 0; k < layer_depth; k++) {
					for (int j = 0; j < neuron_height && y-j >= 0; j++) {
						for (int i = 0; i < neuron_width && x-i >= 0; i++) {
							output_pos = ((k * layer_height) + (y - j)) * layer_width + (x - i);
							input_gradient[c] += output_gradient[output_pos] * parameters[weights_offset + k*(neuron_size + 1) + j*neuron_width + i];
						}
					}
				}
			}
		}

		__global__ void backprop_conv_layer_weights(float* input, float* input_gradient,
													float* parameters, float* gradient, float* output_gradient,
													int input_width, int input_height, int input_depth,
													int layer_width, int layer_height, int layer_depth,
													int neuron_width, int neuron_height,
													int input_size, int output_size, int layer_size, int neuron_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int weights_offset;
			int input_pos;
			int x, y;

			for (int c = index; c < layer_depth*input_depth; c += stride) {
				x = c % input_depth;
				y = c / input_depth;

				weights_offset = (y * (neuron_size+1)) + (x * neuron_height * neuron_width);

				for (int j = 0; j < layer_height; j++) {
					for (int i = 0; i < layer_width; i++) {
						for (int jn = 0; jn < neuron_height; jn++) {
							for (int in = 0; in < neuron_width; in++) {
								input_pos = ((x * input_height) + (j + jn)) * input_width + (i + in);
								gradient[weights_offset + jn * neuron_width + in] += output_gradient[((y * layer_height) + j) * layer_width + i] *
																					input[input_pos];
							}
						}
					}
				}
			}
		}

		__global__ void backprop_conv_layer_bias(float* input, float* input_gradient,
												float* parameters, float* gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int layer_width, int layer_height, int layer_depth,
												int neuron_width, int neuron_height,
												int input_size, int output_size, int layer_size, int neuron_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int c = index; c < layer_depth; c += stride) {
				for (int j = 0; j < layer_height; j++) {
					for (int i = 0; i < layer_width; i++) {
						gradient[c * (neuron_size + 1) + neuron_size] += output_gradient[(c * layer_height + j) * layer_width + i];
					}
				}
			}
		}

		//-------------------------------------------------------------------------------------------

		__global__ void compute_padding_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int output_width, int output_height, int output_depth,
												int offset_width, int offset_height, int offset_depth,
												int input_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y, z, output_pos;

			for (int c = index; c < input_size; c += stride) {
				x = c % input_width;
				y = (c / input_width) % input_height;
				z = c / (input_width * input_height);

				output_pos = (((offset_depth + z) * output_height) + (offset_height + y)) * output_width + (offset_width + x);
				output[output_pos] = input[c];
			}
		}

		__global__ void backprop_padding_layer(float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int output_width, int output_height, int output_depth,
												int offset_width, int offset_height, int offset_depth,
												int input_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y, z, output_pos;

			for (int c = index; c < input_size; c += stride) {
				x = c % input_width;
				y = (c / input_width) % input_height;
				z = c / (input_width * input_height);

				output_pos = (((offset_depth + z) * output_height) + (offset_height + y)) * output_width + (offset_width + x);
				input_gradient[c] += output_gradient[output_pos];
			}
		}

		//-------------------------------------------------------------------------------------------

		__global__ void set_input_layer(float* input, float* output, int input_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int i = index; i < input_size; i += stride) {
				output[i] = input[i];
			}
		}

		//-------------------------------------------------------------------------------------------

		__global__ void compute_pool_max_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int output_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y, z;
			int input_id;
			float max_input = -1e38f;

			for (int c = index; c < output_size; c += stride) {
				x = c % output_width;
				y = (c / output_width) % output_height;
				z = c / (output_width * output_height);

				x *= filter_width;
				y *= filter_height;
				z *= filter_depth;

				for (int k = z; k < z + filter_depth; k++) {
					for (int j = y; j < y + filter_height; j++) {
						for (int i = x; i < x + filter_width; i++) {
							input_id = ((k * input_height) + j) * input_width + i;

							max_input = max(max_input, input[input_id]);
						}
					}
				}

				output[c] = max_input;
			}
		}

		__global__ void backprop_pool_max_layer(float* input, float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int output_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y, z;
			int input_id, mx_in_id;
			float max_input = -1e38f;

			for (int c = index; c < output_size; c += stride) {
				x = c % output_width;
				y = (c / output_width) % output_height;
				z = c / (output_width * output_height);

				x *= filter_width;
				y *= filter_height;
				z *= filter_depth;

				for (int k = z; k < z + filter_depth; k++) {
					for (int j = y; j < y + filter_height; j++) {
						for (int i = x; i < x + filter_width; i++) {
							input_id = ((k * input_height) + j) * input_width + i;

							if (input[input_id] > max_input) {
								max_input = input[input_id];
								mx_in_id = input_id;
							}
						}
					}
				}

				input_gradient[mx_in_id] += output_gradient[c];
			}
		}

		//------------------------------------------------------------------------------------

		__global__ void compute_pool_avg_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int output_size, float filter_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y, z;
			int input_id;

			for (int c = index; c < output_size; c += stride) {
				x = c % output_width;
				y = (c / output_width) % output_height;
				z = c / (output_width * output_height);

				x *= filter_width;
				y *= filter_height;
				z *= filter_depth;

				output[c] = 0.0f;

				for (int k = z; k < z + filter_depth; k++) {
					for (int j = y; j < y + filter_height; j++) {
						for (int i = x; i < x + filter_width; i++) {
							input_id = ((k * input_height) + j) * input_width + i;

							output[c] += input[input_id];
						}
					}
				}

				output[c] /= filter_size;
			}
		}

		__global__ void backprop_pool_avg_layer(float* input, float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int input_size, float filter_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y, z;

			for (int c = index; c < input_size; c += stride) {
				x = c % input_width;
				y = (c / input_width) % input_height;
				z = c / (input_width * input_height);

				x /= filter_width;
				y /= filter_height;
				z /= filter_depth;

				input_gradient[c] += output_gradient[((z * output_height) + y) * output_width + x]/filter_size;
			}
		}

		//------------------------------------------------------------------------------------

		__global__ void compute_upscale_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int output_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y, z;

			for (int c = index; c < output_size; c += stride) {
				x = c % output_width;
				y = (c / output_width) % output_height;
				z = c / (output_width * output_height);

				x /= filter_width;
				y /= filter_height;
				z /= filter_depth;

				output[c] = input[((z*input_height) + y)*input_width + x];
			}
		}

		__global__ void backprop_upscale_layer(float* input, float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int input_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y, z;
			int output_id;

			for (int c = index; c < input_size; c += stride) {
				x = c % input_width;
				y = (c / input_width) % input_height;
				z = c / (input_width * input_height);

				x *= filter_width;
				y *= filter_height;
				z *= filter_depth;

				for (int k = z; k < z + filter_depth; k++) {
					for (int j = y; j < y + filter_height; j++) {
						for (int i = x; i < x + filter_width; i++) {
							output_id = ((k * output_height) + j) * output_width + i;

							input_gradient[c] += output_gradient[output_id];
						}
					}
				}
			}
		}

		//------------------------------------------------------------------------------------

		__global__ void compute_gram_layer(float* input, float* output,
										int input_size, int output_size,
										int vectors, int vector_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y;

			for (int c = index; c < output_size; c += stride) {
				x = c % vectors;
				y = c / vectors;

				output[c] = 0.0f;

				for (int i = 0; i < vector_size; i++) {
					output[c] += input[x * vector_size + i] * input[y * vector_size + i];// / vector_size;
				}
			}

		}

		__global__ void backprop_gram_layer(float* input, float* input_gradient, float* output_gradient,
											int input_size, int output_size,
											int vectors, int vector_size,
											float denominator) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int x, y;

			for (int c = index; c < input_size; c += stride) {
				x = c % vector_size;
				y = c / vector_size;

				for (int i = 0; i < vectors; i++) {
					input_gradient[c] += input[i * vector_size + x] * output_gradient[y * vectors + i];// / vector_size;
				}
			}
		}

		//------------------------------------------------------------------------------------

		__global__ void compute_rearrange_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_size, int sample_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int offset, pos;
			int x, y, z;
			int x_offset, y_offset, z_offset;

			for (int c = index; c < output_size; c += stride) {
				offset = c / sample_size;
				pos = c % sample_size;

				x_offset = offset % filter_width;
				y_offset = (offset / filter_width) % filter_height;
				z_offset = offset / (filter_width * filter_height);

				pos *= filter_width;
				x = (pos % input_width) + x_offset;
				pos /= input_width;
				pos *= filter_height;
				y = (pos % input_height) + y_offset;
				pos /= input_height;
				pos *= filter_depth;
				z = pos + z_offset;

				pos = (z * input_height + y) * input_width + x;

				output[c] = input[pos];
			}
		}

		__global__ void backprop_rearrange_layer(float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_size, int sample_size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int offset, pos;
			int x, y, z;
			int x_offset, y_offset, z_offset;

			for (int c = index; c < output_size; c += stride) {
				offset = c / sample_size;
				pos = c % sample_size;

				x_offset = offset % filter_width;
				y_offset = (offset / filter_width) % filter_height;
				z_offset = offset / (filter_width * filter_height);

				pos *= filter_width;
				x = (pos % input_width) + x_offset;
				pos /= input_width;
				pos *= filter_height;
				y = (pos % input_height) + y_offset;
				pos /= input_height;
				pos *= filter_depth;
				z = pos + z_offset;

				pos = (z * input_height + y) * input_width + x;

				input_gradient[pos] += output_gradient[c];
			}
		}

		//-------------------------------------------------------------------------------------------

		__global__ void adam_optimize(float* parameters, float* gradient, float* mean_gradient, float* mean_squared_gradient, float learning_rate, float batch_size, int parameters_size, float beta1, float beta2, float epsilon, float mean_gradient_adjust, float mean_squared_gradient_adjust) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int i = index; i < parameters_size; i+= stride) {
				gradient[i] /= batch_size;

				mean_gradient[i] = beta1 * mean_gradient[i] + (1.0 - beta1) * gradient[i];

				mean_squared_gradient[i] = beta2 * mean_squared_gradient[i] + (1.0 - beta2) * gradient[i] * gradient[i];

				parameters[i] -= mean_gradient[i] * learning_rate * mean_gradient_adjust / (sqrt(mean_squared_gradient[i] * mean_squared_gradient_adjust) + epsilon);

				gradient[i] = 0.0f;
			}
		}

		//-------------------------------------------------------------------------------------------

		__global__ void sgd_optimize(float* parameters, float* gradient, 
									float learning_rate, float batch_size,
									int parameters_size, float momentum) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int i = index; i < parameters_size; i += stride) {
				gradient[i] /= batch_size;

				parameters[i] -= gradient[i] * learning_rate;

				gradient[i] *= batch_size * momentum;
			}
		}

		//------------------------------------------------------------------------------------

		__global__ void add_arrays(float* input_1, float* input_2, float* output, int size) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int i = index; i < size; i += stride) {
				output[i] = input_1[i] + input_2[i];
			}
		}
	}
}