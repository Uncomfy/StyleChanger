#ifndef NN_LAYERS_DENSELAYER_H_INCLUDED
#define NN_LAYERS_DENSELAYER_H_INCLUDED

#include <cstdio>
#include <vector>

#include "Layer.h"

namespace NN {
    namespace Layers {
        class DenseLayer : public NN::Layers::Layer {
        public:
            DenseLayer() = default;

            DenseLayer(std::vector<int> dependencies,
                       int layer_size);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~DenseLayer();

        private:
            NN::Layers::Layer * prev_layer;
            int neuron_size;
        };
    }
}

#endif // NN_LAYERS_DENSELAYER_H_INCLUDED
