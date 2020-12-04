#include <cmath>

#include "NN/Layers/SoftmaxLayer.h"

using namespace std;

namespace NN {
    namespace Layers {

        SoftmaxLayer::SoftmaxLayer(vector<int> dependencies) {
            this->dependencies = dependencies;
        }

        void SoftmaxLayer::compute() {
            float * prev_layer_output = prev_layer->get_output_iterator();

            float mx = prev_layer_output[0];
            for(int i = 1; i < layer_size; i++) mx = max(mx, prev_layer_output[i]);

            float sum = 0.0;
            for(int i = 0; i < layer_size; i++) {
                sum += exp(prev_layer_output[i] - mx);
            }

            for(int i = 0; i < layer_size; i++) {
                output[i] = exp(prev_layer_output[i]-mx)/sum;
            }
        }

        void SoftmaxLayer::backpropagate() {
            float * prev_layer_output = prev_layer->get_output_iterator();
            float * prev_layer_output_gradient = prev_layer->get_output_gradient_iterator();


            for(int i = 0; i < layer_size; i++) {
                prev_layer_output_gradient[i] = 0.0;

                for(int j = 0; j < layer_size; j++) {
                    prev_layer_output_gradient[i] += output[i]*(float(i==j)-output[j]) * output_gradient[j];
                }
            }
        }

        int SoftmaxLayer::get_parameters_size() {
            return 0;
        }

        void SoftmaxLayer::update_dependencies(vector<NN::Layers::Layer *> layer_dependencies) {
            prev_layer = layer_dependencies[0];
            layer_size = prev_layer->get_output_size();

            output_size = layer_size;
            cudaMallocManaged(&output, layer_size * sizeof(float));
            cudaMallocManaged(&output_gradient, layer_size * sizeof(float));
            cudaMemset(output, 0.0f, layer_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, layer_size * sizeof(float));
        }

        void SoftmaxLayer::save(NN::File& file) {}

        void SoftmaxLayer::load(NN::File& file) {}
    }
}
