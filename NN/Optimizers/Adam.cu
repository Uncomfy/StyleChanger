#include <cmath>

#include "NN/CUDA/CUDA_func.h"
#include "NN/Optimizers/Adam.h"

using namespace std;

namespace NN {
    namespace Optimizers {
        Adam::Adam(float beta1,
                   float beta2,
                   float epsilon) {
            this->beta1 = beta1;
            this->beta2 = beta2;
            this->epsilon = epsilon;
        }

        void Adam::optimize(float learning_rate, float batch_size) {
            /*for(int i = 0; i < parameters_size; i++) {
                gradient[i] /= batch_size;

                mean_gradient[i] = beta1 * mean_gradient[i] + (1.0-beta1) * gradient[i];

                mean_squared_gradient[i] = beta2 * mean_squared_gradient[i] + (1.0-beta2) * gradient[i] * gradient[i];

                parameters[i] -= mean_gradient[i] * learning_rate / (sqrt(mean_squared_gradient[i]) + epsilon);

                gradient[i] = 0.0f;
            }*/

            float mean_gradient_adjust = 1.0f / (1.0f - pow(beta1, operation_id));
            float mean_squared_gradient_adjust = 1.0f / (1.0f - pow(beta2, operation_id));
            operation_id++;
            int blockSize = (parameters_size + 255) / 256;
            NN::CUDA::adam_optimize <<<blockSize, 256 >>> (parameters, gradient,
                                                           mean_gradient, mean_squared_gradient,
                                                           learning_rate, batch_size,
                                                           parameters_size, beta1, beta2, epsilon,
                                                           mean_gradient_adjust, mean_squared_gradient_adjust);

            cudaDeviceSynchronize();
        }

        void Adam::set_parameters(float * parameters,
                                  float* gradient,
                                  int parameters_size) {
            this->parameters = parameters;
            this->gradient = gradient;
            this->parameters_size = parameters_size;
        }

        void Adam::initialize() {
            cudaMallocManaged(&mean_gradient, parameters_size * sizeof(float));
            cudaMallocManaged(&mean_squared_gradient, parameters_size * sizeof(float));
            cudaMemset(mean_gradient, 0, parameters_size * sizeof(float));
            cudaMemset(mean_squared_gradient, 0, parameters_size * sizeof(float));
        }

        void Adam::reset() {
            memset(mean_gradient, 0, parameters_size * sizeof(float));
            memset(mean_squared_gradient, 0, parameters_size * sizeof(float));
        }

        void Adam::save(NN::File& file) {
            int id = 0;
            file.save(id);

            file.save(beta1);
            file.save(beta2);
            file.save(epsilon);
            file.save(operation_id);

            file.save_array(mean_gradient, parameters_size);
            file.save_array(mean_squared_gradient, parameters_size);
        }

        void Adam::load(NN::File& file) {
            file.load(beta1);
            file.load(beta2);
            file.load(epsilon);
            file.load(operation_id);

            cudaMallocManaged(&mean_gradient, parameters_size * sizeof(float));
            cudaMallocManaged(&mean_squared_gradient, parameters_size * sizeof(float));

            file.load_array(mean_gradient, parameters_size);
            file.load_array(mean_squared_gradient, parameters_size);
        }

        Adam::~Adam() {
            cudaFree(mean_gradient);

            cudaFree(mean_squared_gradient);
        }
    }
}
