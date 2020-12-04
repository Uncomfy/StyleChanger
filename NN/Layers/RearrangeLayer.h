#pragma once

#include <cstdio>
#include <vector>

#include "NN/Layers/Layer.h"

namespace NN {
    namespace Layers {
        class RearrangeLayer : public NN::Layers::Layer {
        public:
            RearrangeLayer() = default;

            RearrangeLayer(std::vector<int> dependencies,
                int input_width, int input_height, int input_depth,
                int filter_width, int filter_height, int filter_depth);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer*> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~RearrangeLayer();

        private:
            float* input;
            float* input_gradient;
            int input_width, input_height, input_depth;
            int filter_width, filter_height, filter_depth;
        };
    }
}