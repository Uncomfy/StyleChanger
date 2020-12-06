#ifndef NN_AF_H_INCLUDED
#define NN_AF_H_INCLUDED

#include <cstdio>
#include "File.h"

namespace NN {
    namespace AF {
        class ActivationFunction {
        public:
            virtual void compute(float* input, float* output, int layer_size) = 0;
            virtual void derivative(float* input, float* input_gradient,
                                    float* output, float* output_gradient,
                                    int layer_size) = 0;
            virtual void save(NN::File& file) = 0;
            virtual void load(NN::File& file) = 0;

            virtual ~ActivationFunction() = default;
        };

        class Sigmoid : public ActivationFunction {
        public:
            void compute(float* input, float* output, int layer_size) override;
            void derivative(float* input, float* input_gradient,
                            float* output, float* output_gradient,
                            int layer_size) override;
            void save(NN::File& file) override;
            void load(NN::File& file) override;
        };

        class Tanh : public ActivationFunction {
        public:
            void compute(float* input, float* output, int layer_size) override;
            void derivative(float* input, float* input_gradient,
                            float* output, float* output_gradient,
                            int layer_size) override;
            void save(NN::File& file) override;
            void load(NN::File& file) override;
        };

        class ReLU : public ActivationFunction {
        public:
            void compute(float* input, float* output, int layer_size) override;
            void derivative(float* input, float* input_gradient,
                            float* output, float* output_gradient,
                            int layer_size) override;
            void save(NN::File& file) override;
            void load(NN::File& file) override;
        };

        class LeakyReLU : public ActivationFunction {
        public:
            LeakyReLU() = default;

            LeakyReLU(float alpha);
            void compute(float* input, float* output, int layer_size) override;
            void derivative(float* input, float* input_gradient,
                            float* output, float* output_gradient,
                            int layer_size) override;
            void save(NN::File& file) override;
            void load(NN::File& file) override;

        private:
            float alpha;
        };

        class Sin : public ActivationFunction {
        public:
            void compute(float* input, float* output, int layer_size) override;
            void derivative(float* input, float* input_gradient,
                            float* output, float* output_gradient,
                            int layer_size) override;
            void save(NN::File& file) override;
            void load(NN::File& file) override;
        };

        class Softmax : public ActivationFunction {
        public:
            void compute(float* input, float* output, int layer_size) override;
            void derivative(float* input, float* input_gradient,
                            float* output, float* output_gradient,
                            int layer_size) override;
            void save(NN::File& file) override;
            void load(NN::File& file) override;
        };

        class Types{
        public:
            static NN::AF::ActivationFunction * getTypeFromId(int id);

        private:
            Types() = default;
        };
    }
}

#endif // NN_AF_H_INCLUDED
