#include <cstdlib>
#include <iostream>

#include "NN/File.h"
#include "NN/Layers.h"
#include "NN/Model.h"
#include "NN/Optimizers.h"

using namespace std;

namespace NN {
    Model::Model(NN::Layers::InputLayer * input_layer) {
        this->input_layer = input_layer;
        layers.push_back(input_layer);
    }

    void Model::set_input(float* input) {
        input_layer->set_input(input);
    }

    void Model::set_input(vector<float>::iterator input) {
        input_layer->set_input(input);
    }

    void Model::set_parameters(float* par) {
        cudaMemcpy(parameters, par, par_size * sizeof(float), cudaMemcpyDefault);
    }

    void Model::set_output_gradient(float* output_gradient) {
        float* last_layer_output_gradient = layers[layers.size() - 1]->get_output_gradient_iterator();
        size_t layer_size = layers[layers.size() - 1]->get_output_size();

        cudaMemcpy(last_layer_output_gradient, output_gradient, layer_size * sizeof(float), cudaMemcpyDefault);
    }

    void Model::set_output_gradient(vector<float>::iterator output_gradient) {
        float * last_layer_output_gradient = layers[layers.size()-1]->get_output_gradient_iterator();
        size_t layer_size = layers[layers.size()-1]->get_output_size();

        //cudaDeviceSynchronize();

        for(int i = 0; i < layer_size; i++) {
            last_layer_output_gradient[i] = *(output_gradient+i);
        }
    }

    void Model::set_optimizer(NN::Optimizers::Optimizer * optimizer) {
        this->optimizer = optimizer;

        optimizer->set_parameters(parameters,
                                gradient,
                                par_size);
        optimizer->initialize();
    }

    float * Model::get_output_iterator() {
        return layers[layers.size()-1]->get_output_iterator();
    }

    float* Model::get_input_gradient() {
        return input_layer->get_output_gradient_iterator();
    }

    int Model::get_parameters_size() {
        return par_size;
    }

    void Model::add_layer(NN::Layers::Layer * layer) {
        layers.push_back(layer);
    }

    void Model::build() {
        vector<int> dependencies_ids;
        vector<NN::Layers::Layer *> dependencies;

        int parameters_size = 0;

        for(int i = 1; i < layers.size(); i++) {
            dependencies_ids = layers[i]->get_dependencies();

            dependencies.resize(dependencies_ids.size());
            for(int j = 0; j < dependencies_ids.size(); j++) {
                if (dependencies_ids[j] < 0) {
                    dependencies[j] = layers[i + dependencies_ids[j]];
                }
                else {
                    dependencies[j] = layers[dependencies_ids[j]];
                }
            }

            layers[i]->update_dependencies(dependencies);

            parameters_size += layers[i]->get_parameters_size();
        }

        par_size = parameters_size;

        cudaMallocManaged(&parameters, parameters_size * sizeof(float));
        cudaMallocManaged(&gradient, parameters_size * sizeof(float));
        memset(gradient, 0, parameters_size * sizeof(float));

        parameters_size = 0;

        for(int i = 0; i < layers.size(); i++) {
            layers[i]->set_parameters_iterator(parameters + parameters_size,
                                               gradient + parameters_size);
            parameters_size += layers[i]->get_parameters_size();
        }
    }

    void Model::randomize(float min_value, float max_value) {
        for(int i = 0; i < par_size; i++) {
            parameters[i] = min_value + (max_value - min_value) * float(rand())/RAND_MAX;
        }
    }

    void Model::compute() {
        for(int i = 0; i < layers.size(); i++) {
            layers[i]->compute();
        }
    }

    void Model::backpropagate() {
        for (int i = 0; i < layers.size()-1; i++) {
            layers[i]->nullify_gradient();
        }

        for(int i = layers.size()-1; i >= 0; i--) {
            layers[i]->backpropagate();
        }
    }

    void Model::optimize(float learning_rate, float batch_size) {
        optimizer->optimize(learning_rate, batch_size);
    }

    void Model::reset_optimizer() {
        optimizer->reset();
    }

    int Model::get_memory_usage() {
        int memory_usage = 0;
        for (int i = 0; i < layers.size(); i++) {
            memory_usage += layers[i]->get_output_size() * 2;
        }

        memory_usage += par_size * 2;
        memory_usage *= 4;

        return memory_usage;
    }

    size_t Model::get_output_size() {
        return layers[layers.size() - 1]->get_output_size();
    }

    void Model::save(FILE* file_pointer) {
        bool endianness = NN::File::is_little_endian();

        NN::File file(file_pointer);
        file.save(endianness);

        input_layer->save(file);
        int layers_size = (int)layers.size();
        file.save(layers_size);
        for (int i = 1; i < layers.size(); i++) {
            layers[i]->save(file);
        }

        optimizer->save(file);

        file.save_array(parameters, par_size);
        file.save_array(gradient, par_size);
    }

    void Model::load(FILE* file_pointer) {
        bool endianness;

        NN::File file(file_pointer);
        file.load(endianness);
        bool inverse = NN::File::is_little_endian() ^ endianness;
        file.set_inverse(inverse);

        input_layer = new NN::Layers::InputLayer();
        input_layer->load(file);
        int layers_size;
        file.load(layers_size);
        layers.resize(layers_size);
        layers[0] = input_layer;
        int id;
        for (int i = 1; i < layers_size; i++) {
            file.load(id);
            layers[i] = NN::Layers::Types::getTypeFromId(id);
            layers[i]->load(file);
        }

        build();

        file.load(id);
        optimizer = NN::Optimizers::Types::getTypeFromId(id);
        optimizer->set_parameters(parameters, gradient, par_size);
        optimizer->load(file);

        file.load_array(parameters, par_size);
        file.load_array(gradient, par_size);
    }

    Model::~Model() {
        for (int i = 0; i < layers.size(); i++) {
            delete layers[i];
        }

        layers.clear();
        layers.shrink_to_fit();

        cudaFree(parameters);
        cudaFree(gradient);

        delete optimizer;
    }
}
