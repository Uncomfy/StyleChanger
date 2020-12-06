#include "../CUDA/CUDA_func.h"
#include "PoolAvgLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        PoolAvgLayer::PoolAvgLayer(std::vector<int> dependencies,
                                    int input_width, int input_height, int input_depth,
                                    int filter_width, int filter_height, int filter_depth) {
            this->dependencies = dependencies;
            this->input_width = input_width;
            this->input_height = input_height;
            this->input_depth = input_depth;
            this->filter_width = filter_width;
            this->filter_height = filter_height;
            this->filter_depth = filter_depth;

            output_width = input_width / filter_width;
            output_height = input_height / filter_height;
            output_depth = input_depth / filter_depth;

            output_size = output_width * output_height * output_depth;
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void PoolAvgLayer::compute() {
            int output_size = output_width * output_height * output_depth;
            float filter_size = filter_width * filter_height * filter_depth;

            int block_size = (output_size + 511) / 512;
            NN::CUDA::compute_pool_avg_layer << <block_size, 512 >> > (input, output,
                input_width, input_height, input_depth,
                filter_width, filter_height, filter_depth,
                output_width, output_height, output_depth,
                output_size, filter_size);
            cudaDeviceSynchronize();
        }

        void PoolAvgLayer::backpropagate() {
            int input_size = input_width * input_height * input_depth;
            float filter_size = filter_width * filter_height * filter_depth;

            int block_size = (input_size + 511) / 512;
            NN::CUDA::backprop_pool_avg_layer << <block_size, 512 >> > (input, input_gradient, output_gradient,
                input_width, input_height, input_depth,
                filter_width, filter_height, filter_depth,
                output_width, output_height, output_depth,
                input_size, filter_size);
            cudaDeviceSynchronize();
        }

        int PoolAvgLayer::get_parameters_size() {
            return 0;
        }

        void PoolAvgLayer::update_dependencies(vector<NN::Layers::Layer*> layer_dependencies) {
            input = layer_dependencies[0]->get_output_iterator();
            input_gradient = layer_dependencies[0]->get_output_gradient_iterator();
        }

        void PoolAvgLayer::save(NN::File& file) {
            int id = 6;
            file.save(id);

            save_dependencies(file);
            file.save(input_width);
            file.save(input_height);
            file.save(input_depth);
            file.save(filter_width);
            file.save(filter_height);
            file.save(filter_depth);
        };

        void PoolAvgLayer::load(NN::File& file) {
            load_dependencies(file);
            file.load(input_width);
            file.load(input_height);
            file.load(input_depth);
            file.load(filter_width);
            file.load(filter_height);
            file.load(filter_depth);

            output_width = input_width / filter_width;
            output_height = input_height / filter_height;
            output_depth = input_depth / filter_depth;

            output_size = output_width * output_height * output_depth;
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        };

        PoolAvgLayer::~PoolAvgLayer() = default;
    }
}