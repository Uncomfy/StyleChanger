#ifndef NN_LAYERS_SOFTMAXLAYER_H_INCLUDED
#define NN_LAYERS_SOFTMAXLAYER_H_INCLUDED

#include <cstdio>
#include <vector>

#include "NN/Layers/Layer.h"

namespace NN {
    namespace Layers {
        class SoftmaxLayer : public NN::Layers::Layer {
        public:
            SoftmaxLayer() = default;

            SoftmaxLayer(std::vector<int> dependencies);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~SoftmaxLayer() = default;

        private:
            NN::Layers::Layer * prev_layer;
            int layer_size;
        };
    }
}


#endif // NN_LAYERS_SOFTMAXLAYER_H_INCLUDED
