#include "../CUDA/CUDA_func.h"
#include "ConvolutionLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        ConvolutionLayer::ConvolutionLayer(std::vector<int> dependencies,
                         int width, int height, int depth,
                         int neuron_width, int neuron_height,
                         int layer_depth) {
            this->dependencies = dependencies;
            this->width = width;
            this->height = height;
            this->depth = depth;
            this->neuron_width = neuron_width;
            this->neuron_height = neuron_height;
            this->layer_depth = layer_depth;

            layer_width = width - (neuron_width - 1);
            layer_height = height - (neuron_height - 1);

            output_size = layer_width * layer_height * layer_depth;
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void ConvolutionLayer::compute() {
            float * prev_layer_output = prev_layer->get_output_iterator();
            /*int output_offset;

            for(int k = 0; k < layer_depth; k++) {
                for(int j = 0; j < layer_height; j++) {
                    for(int i = 0; i < layer_width; i++) {
                        output_offset = get_output_offset(i, j, k);
                        output[output_offset] = 0;

                        for(int kn = 0; kn < depth; kn++) {
                            for(int jn = 0; jn < neuron_height; jn++) {
                                for(int in = 0; in < neuron_width; in++) {
                                    output[output_offset] += parameters[get_parameters_offset(k, in, jn, kn)] *
                                                             prev_layer_output[get_input_offset(i+in, j+jn, kn)];
                                }
                            }
                        }

                        output[output_offset] += parameters[(k+1)*(neuron_width*neuron_height*depth + 1) - 1];
                    }
                }
            }*/

            int input_size = width * height * depth;
            int layer_size = layer_height * layer_width;
            int neuron_size = neuron_width * neuron_height * depth;

            int blockSize = (output_size + 511) / 512;
            NN::CUDA::compute_conv_layer<<<blockSize, 512 >>> (prev_layer_output, parameters, output,
                                                                width, height, depth,
                                                                layer_width, layer_height, layer_depth,
                                                                neuron_width, neuron_height,
                                                                input_size, output_size, layer_size, neuron_size);

            cudaDeviceSynchronize();
        }

        void ConvolutionLayer::backpropagate() {
            float * prev_layer_output = prev_layer->get_output_iterator();
            float * prev_layer_output_gradient = prev_layer->get_output_gradient_iterator();
            /*int output_offset;

            for(int k = 0; k < layer_depth; k++) {
                for(int j = 0; j < layer_height; j++) {
                    for(int i = 0; i < layer_width; i++) {
                        output_offset = get_output_offset(i, j, k);

                        for(int kn = 0; kn < depth; kn++) {
                            for(int jn = 0; jn < neuron_height; jn++) {
                                for(int in = 0; in < neuron_width; in++) {
                                    gradient[get_parameters_offset(k, in, jn, kn)] += output_gradient[output_offset] *
                                                                                      prev_layer_output[get_input_offset(i+in, j+jn, kn)];

                                    prev_layer_output_gradient[get_input_offset(i+in, j+jn, kn)] += output_gradient[output_offset] *
                                                                                                    parameters[get_parameters_offset(k, in, jn, kn)];
                                }
                            }
                        }

                        gradient[(k+1)*(neuron_width*neuron_height*depth + 1) - 1] += output_gradient[output_offset];
                    }
                }
            }*/

            int input_size = width * height * depth;
            int layer_size = layer_height * layer_width;
            int neuron_size = neuron_width * neuron_height * depth;

            int blockSize = (input_size + 511) / 512;
            NN::CUDA::backprop_conv_layer_input <<<blockSize, 512 >>> (prev_layer_output, prev_layer_output_gradient,
                                                                        parameters, gradient, output_gradient,
                                                                        width, height, depth,
                                                                        layer_width, layer_height, layer_depth,
                                                                        neuron_width, neuron_height,
                                                                        input_size, output_size, layer_size, neuron_size);

            cudaDeviceSynchronize();

            blockSize = (layer_depth * depth + 511) / 512;
            NN::CUDA::backprop_conv_layer_weights <<<blockSize, 512 >>> (prev_layer_output, prev_layer_output_gradient,
                                                                        parameters, gradient, output_gradient,
                                                                        width, height, depth,
                                                                        layer_width, layer_height, layer_depth,
                                                                        neuron_width, neuron_height,
                                                                        input_size, output_size, layer_size, neuron_size);

            cudaDeviceSynchronize();

            blockSize = (layer_depth + 511) / 512;
            NN::CUDA::backprop_conv_layer_bias <<<blockSize, 512 >>> (prev_layer_output, prev_layer_output_gradient,
                                                                    parameters, gradient, output_gradient,
                                                                    width, height, depth,
                                                                    layer_width, layer_height, layer_depth,
                                                                    neuron_width, neuron_height,
                                                                    input_size, output_size, layer_size, neuron_size);

            cudaDeviceSynchronize();
        }

        int ConvolutionLayer::get_parameters_size() {
            return layer_depth*(neuron_height*neuron_width*depth + 1);
        }

        void ConvolutionLayer::update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) {
            prev_layer = layer_dependencies[0];
        }

        void ConvolutionLayer::save(NN::File& file) {
            int id = 2;
            file.save(id);

            save_dependencies(file);
            file.save(width);
            file.save(height);
            file.save(depth);
            file.save(neuron_width);
            file.save(neuron_height);
            file.save(layer_depth);
        }

        void ConvolutionLayer::load(NN::File& file) {
            load_dependencies(file);
            file.load(width);
            file.load(height);
            file.load(depth);
            file.load(neuron_width);
            file.load(neuron_height);
            file.load(layer_depth);

            layer_width = width - (neuron_width - 1);
            layer_height = height - (neuron_height - 1);

            output_size = layer_width * layer_height * layer_depth;
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        ConvolutionLayer::~ConvolutionLayer() = default;


        int ConvolutionLayer::get_input_offset(int i, int j, int k) {
            return (k*height + j)*width + i;
        }

        int ConvolutionLayer::get_output_offset(int i, int j, int k) {
            return (k*layer_height + j)*layer_width + i;
        }

        int ConvolutionLayer::get_parameters_offset(int current_depth, int i, int j, int k) {
            return ((current_depth*depth + k)*neuron_height + j)*neuron_width + i + current_depth;
        }
    }
}
