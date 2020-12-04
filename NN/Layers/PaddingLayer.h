#ifndef NN_LAYERS_PADDINGLAYER_H_INCLUDED
#define NN_LAYERS_PADDINGLAYER_H_INCLUDED

#include <cstdio>
#include <vector>

#include "NN/Layers/Layer.h"

namespace NN {
    namespace Layers {
        class PaddingLayer : public NN::Layers::Layer {
        public:
            PaddingLayer() = default;

            PaddingLayer(std::vector<int> dependencies,
                         int width,
                         int height,
                         int depth,
                         int left, int right,
                         int top, int bottom,
                         int front, int back);

            void compute() override;

            void backpropagate() override;

            int get_parameters_size() override;

            void update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~PaddingLayer();

        private:
            int get_input_offset(int i, int j, int k);

            int get_output_offset(int i, int j, int k);

            NN::Layers::Layer * prev_layer;
            int width;
            int height;
            int depth;
            int left, right;
            int top, bottom;
            int front, back;
        };
    }
}

#endif // NN_LAYERS_PADDINGLAYER_H_INCLUDED
