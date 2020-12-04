#include "NN/CUDA/CUDA_func.h"
#include "NN/Layers/DenseLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        DenseLayer::DenseLayer(vector<int> dependencies,
                               int layer_size) {
            this->dependencies = dependencies;

            output_size = layer_size;
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void DenseLayer::compute() {
            float * prev_layer_output = prev_layer->get_output_iterator();

            /*for(int i = 0; i < layer_size; i++) {
                output[i] = 0.0f;

                for(int j = 0; j < neuron_size; j++) {
                    output[i] += parameters[i*(neuron_size + 1) + j] *
                                 prev_layer_output[j];
                }

                output[i] += parameters[i*(neuron_size + 1) + neuron_size];
            }*/

            int block_size = (output_size + 255) / 256;
            NN::CUDA::compute_dense_layer<<<block_size, 256>>>(prev_layer_output, parameters, output, neuron_size, output_size);

            cudaDeviceSynchronize();
        }

        void DenseLayer::backpropagate() {
            float * prev_layer_output = prev_layer->get_output_iterator();
            float * prev_layer_output_gradient = prev_layer->get_output_gradient_iterator();

            /*for(int i = 0; i < neuron_size; i++) {
                prev_layer_output_gradient[i] = 0.0f;
            }

            for(int i = 0; i < layer_size; i++) {
                for(int j = 0; j < neuron_size; j++) {
                    gradient[i*(neuron_size + 1) + j] += output_gradient[i] * prev_layer_output[j];
                    prev_layer_output_gradient[j] += output_gradient[i] * parameters[i*(neuron_size + 1) + j];
                }

                gradient[i*(neuron_size + 1) + neuron_size] += output_gradient[i];
            }*/

            int block_size = (neuron_size + 255) / 256;
            NN::CUDA::backprop_dense_layer_input_gradient <<<block_size, 256 >>> (prev_layer_output, prev_layer_output_gradient, parameters, gradient, output_gradient, neuron_size, output_size);

            cudaDeviceSynchronize();

            block_size = (output_size + 255) / 256;
            NN::CUDA::backprop_dense_layer_bias << <block_size, 256 >> > (gradient, output_gradient, neuron_size, output_size);

            cudaDeviceSynchronize();
        }

        int DenseLayer::get_parameters_size() {
            return (neuron_size + 1) * output_size;
        }

        void DenseLayer::update_dependencies(vector<NN::Layers::Layer *> layer_dependencies) {
            prev_layer = layer_dependencies[0];
            neuron_size = prev_layer->get_output_size();
        }

        void DenseLayer::save(NN::File& file) {
            int id = 3;
            file.save(id);

            save_dependencies(file);
            file.save(output_size);
        }

        void DenseLayer::load(NN::File& file) {
            load_dependencies(file);
            file.load(output_size);

            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        DenseLayer::~DenseLayer() = default;
    }
}
