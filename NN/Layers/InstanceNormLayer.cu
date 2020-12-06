#include <cmath>

#include "InstanceNormLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        InstanceNormLayer::InstanceNormLayer(vector<int> dependencies, int layers) {
            this->dependencies = dependencies;
            this->layers = layers;

            sums = new float[layers];
            sigmas = new float[layers];
        }

        void InstanceNormLayer::compute() {
            float inv_sqrt_sigma;
            for (int i = 0; i < layers; i++) {
                sums[i] = 0.0f;
                sigmas[i] = 0.0f;

                for (int j = i * layer_size; j < i * layer_size + layer_size; j++) {
                    sums[i] += input[j];
                }
                sums[i] /= layer_size;

                for (int j = i * layer_size; j < i * layer_size + layer_size; j++) {
                    output[j] = (input[j] - sums[i]);
                    sigmas[i] += output[j] * output[j];
                }
                sigmas[i] /= layer_size;

                inv_sqrt_sigma = 1.0f / sqrt(sigmas[i] + 1e-8f);

                for (int j = i * layer_size; j < i * layer_size + layer_size; j++) {
                    output[j] = parameters[i*2] * output[j] * inv_sqrt_sigma + parameters[i*2 + 1];
                }
            }
        }

        void InstanceNormLayer::backpropagate() {
            float inv_sqrt_sigma;
            float dsigma;
            float dsum, t_dsum;

            for (int i = 0; i < layers; i++) {
                inv_sqrt_sigma = 1.0f / sqrt(sigmas[i] + 1e-8f);
                dsigma = 0.0f;
                dsum = 0.0f;
                t_dsum = 0.0f;

                for (int j = i * layer_size; j < i * layer_size + layer_size; j++) {
                    gradient[i*2] += output_gradient[j] * (output[j] - parameters[i*2 + 1]) / parameters[i*2];
                    gradient[i*2 + 1] += output_gradient[j];
                    output_gradient[j] *= parameters[i*2];
                    dsum += output_gradient[j];
                    t_dsum += (input[j] - sums[i]);
                    dsigma += output_gradient[j] * (input[j] - sums[i]);
                }
                dsum *= -inv_sqrt_sigma;
                dsigma *= -0.5f * pow(sigmas[i] + 1e-8f, -1.5f);
                dsum += -2.0f * t_dsum * dsigma / layer_size;

                for (int j = i * layer_size; j < i * layer_size + layer_size; j++) {
                    input_gradient[j] += output_gradient[j] * inv_sqrt_sigma + (2.0f * dsigma * (input[j] - sums[i]) + dsum) / layer_size;
                }
            }

        }

        int InstanceNormLayer::get_parameters_size() {
            return 2*layers;
        }

        void InstanceNormLayer::update_dependencies(vector<NN::Layers::Layer*> layer_dependencies) {
            input = layer_dependencies[0]->get_output_iterator();
            input_gradient = layer_dependencies[0]->get_output_gradient_iterator();
            output_size = layer_dependencies[0]->get_output_size();
            layer_size = output_size / layers;

            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void InstanceNormLayer::save(NN::File& file) {
            int id = 11;
            file.save(id);

            save_dependencies(file);
            file.save(layers);
        }

        void InstanceNormLayer::load(NN::File& file) {
            load_dependencies(file);
            file.load(layers);
            sums = new float[layers];
            sigmas = new float[layers];
        }

        InstanceNormLayer::~InstanceNormLayer() {
            delete[] sums;
            delete[] sigmas;
        }
    }
}
