#include "NN/Layers/ActivationLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        ActivationLayer::ActivationLayer(vector<int> dependencies,
                                         NN::AF::ActivationFunction * activation_function) {
            this->dependencies = dependencies;
            this->activation_function = activation_function;
        }

        void ActivationLayer::compute() {
            float * prev_layer_output = prev_layer->get_output_iterator();

            activation_function->compute(prev_layer_output, output, output_size);
            cudaDeviceSynchronize();
        }

        void ActivationLayer::backpropagate() {
            float * prev_layer_output = prev_layer->get_output_iterator();
            float * prev_layer_output_gradient = prev_layer->get_output_gradient_iterator();

            activation_function->derivative(prev_layer_output, prev_layer_output_gradient,
                                            output, output_gradient,
                                            output_size);
            cudaDeviceSynchronize();
        }

        int ActivationLayer::get_parameters_size() {
            return 0;
        }

        void ActivationLayer::update_dependencies(vector<NN::Layers::Layer *> layer_dependencies) {
            prev_layer = layer_dependencies[0];
            output_size = prev_layer->get_output_size();

            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void ActivationLayer::save(NN::File& file) {
            int id = 0;
            file.save(id);

            save_dependencies(file);
            activation_function->save(file);
        }

        void ActivationLayer::load(NN::File& file) {
            load_dependencies(file);

            int id;
            file.load(id);

            activation_function = NN::AF::Types::getTypeFromId(id);
            activation_function->load(file);
        }

        ActivationLayer::~ActivationLayer() {
            delete activation_function;
        }
    }
}
