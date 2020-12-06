#include "../CUDA/CUDA_func.h"
#include "GramLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        GramLayer::GramLayer(vector<int> dependencies,
                            int vectors) {
            this->dependencies = dependencies;
            this->vectors = vectors;

            output_size = vectors * vectors;
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void GramLayer::compute() {
            int output_size = vectors * vectors;
            int block_size = (output_size + 511) / 512;
            NN::CUDA::compute_gram_layer <<<block_size, 512 >>> (input, output,
                                                                input_size, output_size,
                                                                vectors, vector_size);

            cudaDeviceSynchronize();
        }

        void GramLayer::backpropagate() {
            int output_size = vectors * vectors;
            int block_size = (input_size + 511) / 512;
            float denominator = float(vectors) * float(vector_size);
            denominator *= denominator;
            denominator = 1.0f;
            NN::CUDA::backprop_gram_layer<<<block_size, 512 >>> (input, input_gradient, output_gradient,
                                                                input_size, output_size,
                                                                vectors, vector_size,
                                                                denominator);

            cudaDeviceSynchronize();
        }

        int GramLayer::get_parameters_size() {
            return 0;
        }

        void GramLayer::update_dependencies(vector<NN::Layers::Layer*> layer_dependencies) {
            input = layer_dependencies[0]->get_output_iterator();
            input_gradient = layer_dependencies[0]->get_output_gradient_iterator();
            input_size = layer_dependencies[0]->get_output_size();
            vector_size = input_size / vectors;
        }

        void GramLayer::save(NN::File& file) {
            int id = 4;
            file.save(id);

            save_dependencies(file);
            file.save(vectors);
        }

        void GramLayer::load(NN::File& file) {
            load_dependencies(file);
            file.load(vectors);

            output_size = vectors * vectors;
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        GramLayer::~GramLayer() = default;
    }
}
