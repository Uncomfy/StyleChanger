#pragma once

#include <cstdio>
#include <vector>

#include "NN/Layers/Layer.h"

namespace NN {
    namespace Layers {
        class GramLayer : public NN::Layers::Layer {
        public:
            GramLayer() = default;

            GramLayer(std::vector<int> dependencies,
                        int vectors);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer*> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~GramLayer();

        private:
            float* input;
            float* input_gradient;
            int input_size;
            int vectors, vector_size;
        };
    }
}