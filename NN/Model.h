#ifndef NN_MODEL_H_INCLUDED
#define NN_MODEL_H_INCLUDED

#include <cstdio>
#include <vector>

#include "Layers/Layer.h"
#include "Layers/InputLayer.h"
#include "Optimizers/Optimizer.h"

namespace NN {
    class Model {
    public:
        Model() = default;

        Model(NN::Layers::InputLayer * input_layer);

        void set_input(float* input);

        void set_input(std::vector<float>::iterator input);

        void set_parameters(float* par);

        void set_output_gradient(float* output_gradient);

        void set_output_gradient(std::vector<float>::iterator output_gradient);

        void set_optimizer(NN::Optimizers::Optimizer * optimizer);

        float * get_output_iterator();

        float* get_input_gradient();

        int get_memory_usage();

        size_t get_output_size();

        int get_parameters_size();

        void add_layer(NN::Layers::Layer * layer);

        void build();

        void randomize(float min_value, float max_value);

        void compute();

        void backpropagate();

        void optimize(float learning_rate, float batch_size);

        void reset_optimizer();

        void save(FILE* file_pointer);

        void load(FILE* file_pointer);

        ~Model();

    private:
        std::vector<NN::Layers::Layer *> layers;

        float * parameters;
        float * gradient;

        int par_size;

        NN::Layers::InputLayer * input_layer;
        NN::Optimizers::Optimizer * optimizer;
    };
}

#endif // NN_MODEL_H_INCLUDED
