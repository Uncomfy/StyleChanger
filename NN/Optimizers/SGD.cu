#include <cmath>

#include "../CUDA/CUDA_func.h"
#include "SGD.h"

using namespace std;

namespace NN {
    namespace Optimizers {
        SGD::SGD(float momentum) {
            this->momentum = momentum;
        }

        void SGD::optimize(float learning_rate, float batch_size) {
            /*for(int i = 0; i < parameters_size; i++) {
                gradient[i] /= batch_size;

                mean_gradient[i] = beta1 * mean_gradient[i] + (1.0-beta1) * gradient[i];

                mean_squared_gradient[i] = beta2 * mean_squared_gradient[i] + (1.0-beta2) * gradient[i] * gradient[i];

                parameters[i] -= mean_gradient[i] * learning_rate / (sqrt(mean_squared_gradient[i]) + epsilon);

                gradient[i] = 0.0f;
            }*/

            int blockSize = (parameters_size + 255) / 256;
            NN::CUDA::sgd_optimize << <blockSize, 256 >> > (parameters, gradient,
                learning_rate, batch_size,
                parameters_size, momentum);

            cudaDeviceSynchronize();
        }

        void SGD::set_parameters(float* parameters,
            float* gradient,
            int parameters_size) {
            this->parameters = parameters;
            this->gradient = gradient;
            this->parameters_size = parameters_size;
        }

        void SGD::initialize() {}

        void SGD::reset() {
            cudaMemset(gradient, 0, parameters_size * sizeof(float));
        }

        void SGD::save(NN::File& file) {
            int id = 1;
            file.save(id);

            file.save(momentum);
        }

        void SGD::load(NN::File& file) {
            file.load(momentum);
        }

        SGD::~SGD() = default;
    }
}
