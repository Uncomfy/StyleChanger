#ifndef NN_LAYERS_INPUTLAYER_H_INCLUDED
#define NN_LAYERS_INPUTLAYER_H_INCLUDED

#include <cstdio>
#include <vector>

#include "NN/Layers/Layer.h"

namespace NN {
    namespace Layers {
        class InputLayer : public NN::Layers::Layer {
        public:
            InputLayer() = default;

            InputLayer(int layer_size);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            void set_input(float* input);

            void set_input(std::vector<float>::iterator input);

            ~InputLayer() = default;

        private:
        };
    }
}

#endif // NN_LAYERS_INPUTLAYER_H_INCLUDED
