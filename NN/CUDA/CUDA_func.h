#pragma once

namespace NN {
	namespace CUDA {
		__global__ void compute_dense_layer(float* input, float* parameters, float* output, int input_size, int output_size);

		__global__ void backprop_dense_layer_input_gradient(float* input, float* input_gradient, float* parameters, float* gradient, float* output_gradient, int input_size, int output_size);

		__global__ void backprop_dense_layer_bias(float* gradient, float* output_gradient, int input_size, int output_size);

		//------------------------------------------------------------------------------------

		__global__ void compute_conv_layer(float* input, float* parameters, float* output,
										int input_width, int input_height, int input_depth,
										int layer_width, int layer_height, int layer_depth,
										int neuron_width, int neuron_height,
										int input_size, int output_size, int layer_size, int neuron_size);

		__global__ void backprop_conv_layer_input(float* input, float* input_gradient,
												float* parameters, float* gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int layer_width, int layer_height, int layer_depth,
												int neuron_width, int neuron_height,
												int input_size, int output_size, int layer_size, int neuron_size);

		__global__ void backprop_conv_layer_weights(float* input, float* input_gradient,
													float* parameters, float* gradient, float* output_gradient,
													int input_width, int input_height, int input_depth,
													int layer_width, int layer_height, int layer_depth,
													int neuron_width, int neuron_height,
													int input_size, int output_size, int layer_size, int neuron_size);

		__global__ void backprop_conv_layer_bias(float* input, float* input_gradient,
												float* parameters, float* gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int layer_width, int layer_height, int layer_depth,
												int neuron_width, int neuron_height,
												int input_size, int output_size, int layer_size, int neuron_size);

		//------------------------------------------------------------------------------------

		__global__ void compute_padding_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int output_width, int output_height, int output_depth,
												int offset_width, int offset_height, int offset_depth,
												int input_size);

		__global__ void backprop_padding_layer(float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int output_width, int output_height, int output_depth,
												int offset_width, int offset_height, int offset_depth,
												int input_size);

		//------------------------------------------------------------------------------------

		__global__ void set_input_layer(float* input, float* output, int input_size);

		//------------------------------------------------------------------------------------

		__global__ void compute_pool_max_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int output_size);

		__global__ void backprop_pool_max_layer(float* input, float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int output_size);

		//------------------------------------------------------------------------------------

		__global__ void compute_pool_avg_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int output_size, float filter_size);

		__global__ void backprop_pool_avg_layer(float* input, float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int input_size, float filter_size);

		//------------------------------------------------------------------------------------

		__global__ void compute_upscale_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int output_size);

		__global__ void backprop_upscale_layer(float* input, float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_width, int output_height, int output_depth,
												int input_size);

		//------------------------------------------------------------------------------------

		__global__ void compute_gram_layer(float* input, float* output,
											int input_size, int output_size,
											int vectors, int vector_size);

		__global__ void backprop_gram_layer(float* input, float* input_gradient, float* output_gradient,
											int input_size, int output_size,
											int vectors, int vector_size,
											float denominator);

		//------------------------------------------------------------------------------------

		__global__ void compute_rearrange_layer(float* input, float* output,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_size, int sample_size);

		__global__ void backprop_rearrange_layer(float* input_gradient, float* output_gradient,
												int input_width, int input_height, int input_depth,
												int filter_width, int filter_height, int filter_depth,
												int output_size, int sample_size);

		//------------------------------------------------------------------------------------

		__global__ void adam_optimize(float* parameters, float* gradient, float* mean_gradient, float* mean_squared_gradient, float learning_rate, float batch_size, int parameters_size, float beta1, float beta2, float epsilon, float mean_gradient_adjust, float mean_squared_gradient_adjust);

		//------------------------------------------------------------------------------------

		__global__ void sgd_optimize(float* parameters, float* gradient,
			float learning_rate, float batch_size,
			int parameters_size, float momentum);

		//------------------------------------------------------------------------------------

		__global__ void add_arrays(float* input_1, float* input_2, float* output, int size);
	}
}