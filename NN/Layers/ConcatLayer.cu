#include "ConcatLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        ConcatLayer::ConcatLayer(vector<int> dependencies) {
            this->dependencies = dependencies;
        }

        void ConcatLayer::compute() {
            int offset = 0;
            int layer_size;
            for (int i = 0; i < layer_dependencies.size(); i++) {
                layer_size = layer_dependencies[i]->get_output_size();
                memcpy(output + offset, layer_dependencies[i]->get_output_iterator(), layer_size * sizeof(float));
                offset += layer_size;
            }
        }

        void ConcatLayer::backpropagate() {
            int offset = 0;
            int layer_size;
            for (int i = 0; i < layer_dependencies.size(); i++) {
                layer_size = layer_dependencies[i]->get_output_size();
                memcpy(layer_dependencies[i]->get_output_gradient_iterator(), output_gradient + offset, layer_size * sizeof(float));
                offset += layer_size;
            }
        }

        int ConcatLayer::get_parameters_size() {
            return 0;
        }

        void ConcatLayer::update_dependencies(vector<NN::Layers::Layer*> layer_dependencies) {
            this->layer_dependencies = layer_dependencies;

            output_size = 0;
            for (int i = 0; i < layer_dependencies.size(); i++) {
                output_size += layer_dependencies[i]->get_output_size();
            }

            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void ConcatLayer::save(NN::File& file) {
            int id = 1;
            file.save(id);

            save_dependencies(file);
        }

        void ConcatLayer::load(NN::File& file) {
            load_dependencies(file);
        }

        ConcatLayer::~ConcatLayer() = default;
    }
}
