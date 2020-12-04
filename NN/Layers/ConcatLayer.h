#pragma once

#include <cstdio>
#include <vector>

#include "NN/Layers/Layer.h"

namespace NN {
    namespace Layers {
        class ConcatLayer : public NN::Layers::Layer {
        public:
            ConcatLayer() = default;

            ConcatLayer(std::vector<int> dependencies);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer*> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~ConcatLayer();

        private:
            std::vector<NN::Layers::Layer*> layer_dependencies;
        };
    }
}