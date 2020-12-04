#ifndef NN_LAYERS_LAYER_H_INCLUDED
#define NN_LAYERS_LAYER_H_INCLUDED

#include <cstdio>
#include <vector>

#include "NN/File.h"

namespace NN {
    namespace Layers {
        class Layer {
        public:
            virtual void compute() = 0;

            virtual void backpropagate() = 0;

            virtual int get_parameters_size() = 0;

            virtual void update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) = 0;

            virtual void save(NN::File& file) = 0;

            virtual void load(NN::File& file) = 0;

            virtual ~Layer();

            void set_parameters_iterator(float * parameters,
                                         float * gradient);

            size_t get_output_size();

            float * get_output_iterator();

            float * get_output_gradient_iterator();

            std::vector<int> get_dependencies();

            void nullify_gradient();

            void save_dependencies(NN::File& file);

            void load_dependencies(NN::File& file);

        protected:
            float * output;
            float * output_gradient;

            size_t output_size;

            float * parameters;
            float * gradient;

            std::vector<int> dependencies;
        };
    }
}

#endif // NN_LAYERS_LAYER_H_INCLUDED
