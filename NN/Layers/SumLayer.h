#pragma once

#include <cstdio>
#include <vector>

#include "Layer.h"

namespace NN {
    namespace Layers {
        class SumLayer : public NN::Layers::Layer {
        public:
            SumLayer() = default;

            SumLayer(std::vector<int> dependencies);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer*> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~SumLayer();

        private:
            std::vector<NN::Layers::Layer*> layer_dependencies;
        };
    }
}