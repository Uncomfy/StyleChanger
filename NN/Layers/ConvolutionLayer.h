#ifndef NN_LAYERS_CONVOLUTIONLAYER_H_INCLUDED
#define NN_LAYERS_CONVOLUTIONLAYER_H_INCLUDED

#include <cstdio>
#include <vector>

#include "NN/Layers/Layer.h"

namespace NN {
    namespace Layers {
        class ConvolutionLayer : public NN::Layers::Layer {
        public:
            ConvolutionLayer() = default;

            ConvolutionLayer(std::vector<int> dependencies,
                             int width, int height, int depth,
                             int neuron_width, int neuron_height,
                             int layer_depth);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~ConvolutionLayer();

        private:
            int get_input_offset(int i, int j, int k);

            int get_output_offset(int i, int j, int k);

            int get_parameters_offset(int depth, int i, int j, int k);

            NN::Layers::Layer * prev_layer;
            int width;
            int height;
            int depth;
            int neuron_width;
            int neuron_height;
            int layer_width;
            int layer_height;
            int layer_depth;
        };
    }
}

#endif // NN_LAYERS_CONVOLUTIONLAYER_H_INCLUDED
