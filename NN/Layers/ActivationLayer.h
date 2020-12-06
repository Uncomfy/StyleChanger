#ifndef NN_LAYERS_ACTIVATIONLAYER_H_INCLUDED
#define NN_LAYERS_ACTIVATIONLAYER_H_INCLUDED

#include <cstdio>
#include <vector>

#include "../ActivationFunctions.h"
#include "Layer.h"

namespace NN {
    namespace Layers {
        class ActivationLayer : public NN::Layers::Layer {
        public:
            ActivationLayer() = default;

            ActivationLayer(std::vector<int> dependencies,
                            NN::AF::ActivationFunction * activation_function);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~ActivationLayer();

        private:
            NN::Layers::Layer * prev_layer;
            NN::AF::ActivationFunction * activation_function;
        };
    }
}

#endif // NN_LAYERS_ACTIVATIONLAYER_H_INCLUDED
