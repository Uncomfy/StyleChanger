#include <algorithm>

#include "NN/CUDA/CUDA_func.h"
#include "NN/Layers/SumLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        SumLayer::SumLayer(vector<int> dependencies) {
            this->dependencies = dependencies;
        }

        void SumLayer::compute() {
            memset(output, 0, output_size * sizeof(float));
            int t_size, block_size;
            for (int i = 0; i < layer_dependencies.size(); i++) {
                t_size = min(output_size, layer_dependencies[i]->get_output_size());
                block_size = (t_size + 511) / 512;
                NN::CUDA::add_arrays<<<block_size, 512>>>(layer_dependencies[i]->get_output_iterator(), output, output, t_size);

                cudaDeviceSynchronize();
            }
        }

        void SumLayer::backpropagate() {
            int t_size, block_size;
            for (int i = 0; i < layer_dependencies.size(); i++) {
                t_size = min(output_size, layer_dependencies[i]->get_output_size());
                block_size = (t_size + 511) / 512;
                NN::CUDA::add_arrays<<<block_size, 512>>>(layer_dependencies[i]->get_output_gradient_iterator(), output_gradient, layer_dependencies[i]->get_output_gradient_iterator(), t_size);

                cudaDeviceSynchronize();
            }
        }

        int SumLayer::get_parameters_size() {
            return 0;
        }

        void SumLayer::update_dependencies(vector<NN::Layers::Layer*> layer_dependencies) {
            this->layer_dependencies = layer_dependencies;

            output_size = 0;
            for (int i = 0; i < layer_dependencies.size(); i++) {
                output_size = max(output_size, layer_dependencies[i]->get_output_size());
            }

            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void SumLayer::save(NN::File& file) {
            int id = 10;
            file.save(id);

            save_dependencies(file);
        }

        void SumLayer::load(NN::File& file) {
            load_dependencies(file);
        }

        SumLayer::~SumLayer() = default;
    }
}
