#pragma once

#include <cstdio>
#include <vector>

#include "Layer.h"

namespace NN {
    namespace Layers {
        class InstanceNormLayer : public NN::Layers::Layer {
        public:
            InstanceNormLayer() = default;

            InstanceNormLayer(std::vector<int> dependencies, int layers);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer*> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~InstanceNormLayer();

        private:
            float* input;
            float* input_gradient;

            int layers, layer_size;
            float* sums;
            float* sigmas;
        };
    }
}