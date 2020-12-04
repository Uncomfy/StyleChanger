#include "NN/CUDA/CUDA_func.h"
#include "NN/Layers/InputLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        InputLayer::InputLayer(int layer_size) {
            output_size = layer_size;
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0, output_size * sizeof(float));
            cudaMemset(output_gradient, 0, output_size * sizeof(float));
        }

        void InputLayer::compute() {}

        void InputLayer::backpropagate() {}

        int InputLayer::get_parameters_size() {
            return 0;
        }

        void InputLayer::update_dependencies(vector<NN::Layers::Layer *> layer_dependencies) {}

        void InputLayer::save(NN::File& file) {
            file.save(output_size);
        }

        void InputLayer::load(NN::File& file) {
            file.load(output_size);

            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0, output_size * sizeof(float));
            cudaMemset(output_gradient, 0, output_size * sizeof(float));
        }

        void InputLayer::set_input(float* input) {
            /*for(int i = 0; i < layer_size; i++) {
                output[i] = input[i];
            }*/


            memcpy(output, input, output_size * sizeof(float));
        }

        void InputLayer::set_input(vector<float>::iterator input) {
            for(int i = 0; i < output_size; i++) {
                output[i] = *(input+i);
            }
        }
    }
}
