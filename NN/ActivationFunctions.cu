#include <algorithm>
#include <cmath>

#include "ActivationFunctions.h"
#include "CUDA/CUDA_ActivationFunctions.h"

using namespace std;

namespace NN {
    namespace AF {
        void Sigmoid::compute(float* input, float* output, int layer_size) {
            int block_size = (layer_size + 255) / 256;
            NN::CUDA::AF::sigmoid_compute <<<block_size, 256>>> (input, output, layer_size);
        }

        void Sigmoid::derivative(float* input, float* input_gradient,
                                 float* output, float* output_gradient,
                                 int layer_size) {
            int block_size = (layer_size + 255) / 256;
            NN::CUDA::AF::sigmoid_derivative <<<block_size, 256>>> (input_gradient,
                                                                    output, output_gradient,
                                                                    layer_size);
        }

        void Sigmoid::save(NN::File& file) {
            int id = 0;
            file.save(id);
        }

        void Sigmoid::load(NN::File& file) {}

        //-------------------------------------------------------
        
        void Tanh::compute(float* input, float* output, int layer_size) {
            int block_size = (layer_size + 255) / 256;
            NN::CUDA::AF::tanh_compute<<<block_size, 256>>>(input, output, layer_size);
        }

        void Tanh::derivative(float* input, float* input_gradient,
                              float* output, float* output_gradient,
                              int layer_size) {
            int block_size = (layer_size + 255) / 256;
            NN::CUDA::AF::tanh_derivative<<<block_size, 256>>>(input_gradient,
                                                               output, output_gradient,
                                                               layer_size);
        }

        void Tanh::save(NN::File& file) {
            int id = 1;
            file.save(id);
        }

        void Tanh::load(NN::File& file) {}

        //-------------------------------------------------------

        void ReLU::compute(float* input, float* output, int layer_size) {
            int block_size = (layer_size + 511) / 512;
            NN::CUDA::AF::relu_compute <<<block_size, 512 >>> (input, output, layer_size);
        }

        void ReLU::derivative(float* input, float* input_gradient,
            float* output, float* output_gradient,
            int layer_size) {
            int block_size = (layer_size + 511) / 512;
            NN::CUDA::AF::relu_derivative <<<block_size, 512 >>> (input, input_gradient,
                                                                  output_gradient,
                                                                  layer_size);
        }

        void ReLU::save(NN::File& file) {
            int id = 2;
            file.save(id);
        }

        void ReLU::load(NN::File& file) {}

        //-------------------------------------------------------
        
        LeakyReLU::LeakyReLU(float alpha) {
            this->alpha = alpha;
        }

        void LeakyReLU::compute(float* input, float* output, int layer_size) {
            int block_size = (layer_size + 511) / 512;
            NN::CUDA::AF::leaky_relu_compute <<<block_size, 512 >>> (input, output, layer_size, alpha);
        }

        void LeakyReLU::derivative(float* input, float* input_gradient,
            float* output, float* output_gradient,
            int layer_size) {
            int block_size = (layer_size + 511) / 512;
            NN::CUDA::AF::leaky_relu_derivative <<<block_size, 512 >>> (input, input_gradient,
                                                                        output_gradient,
                                                                        layer_size, alpha);
        }

        void LeakyReLU::save(NN::File& file) {
            int id = 3;
            file.save(id);

            file.save(alpha);
        }

        void LeakyReLU::load(NN::File& file) {
            file.load(alpha);
        }

        //-------------------------------------------------------
        
        void Sin::compute(float* input, float* output, int layer_size) {
            int block_size = (layer_size + 255) / 256;
            NN::CUDA::AF::sin_compute <<<block_size, 256 >>> (input, output, layer_size);
        }

        void Sin::derivative(float* input, float* input_gradient,
            float* output, float* output_gradient,
            int layer_size) {
            int block_size = (layer_size + 255) / 256;
            NN::CUDA::AF::sin_derivative <<<block_size, 256 >>> (input, input_gradient,
                                                                 output_gradient,
                                                                 layer_size);
        }

        void Sin::save(NN::File& file) {
            int id = 4;
            file.save(id);
        }

        void Sin::load(NN::File& file) {}

        //-------------------------------------------------------
        
        void Softmax::compute(float* input, float* output, int layer_size) {
            float mx = input[0];
            for (int i = 1; i < layer_size; i++) mx = max(mx, input[i]);

            float sum = 0.0;
            for (int i = 0; i < layer_size; i++) {
                sum += exp(input[i] - mx);
            }

            for (int i = 0; i < layer_size; i++) {
                output[i] = exp(input[i] - mx) / sum;
            }
        }

        void Softmax::derivative(float* input, float* input_gradient,
                                 float* output, float* output_gradient,
                                 int layer_size) {

            for (int i = 0; i < layer_size; i++) {
                for (int j = 0; j < layer_size; j++) {
                    input_gradient[i] += output[i] * (float(i == j) - output[j]) * output_gradient[j];
                }
            }
        }

        void Softmax::save(NN::File& file) {
            int id = 5;
            file.save(id);
        }

        void Softmax::load(NN::File& file) {}

        //-------------------------------------------------------

        NN::AF::ActivationFunction * Types::getTypeFromId(int id) {
            switch(id) {
            case 0:
                return new NN::AF::Sigmoid;
            case 1:
                return new NN::AF::Tanh;
            case 2:
                return new NN::AF::ReLU;
            case 3:
                return new NN::AF::LeakyReLU;
            case 4:
                return new NN::AF::Sin;
            case 5:
                return new NN::AF::Softmax;
            default:
                return new NN::AF::Sigmoid;
            }
        }
    }
}
