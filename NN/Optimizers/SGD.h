#pragma once

#include "NN/Optimizers/Optimizer.h"

namespace NN {
    namespace Optimizers {
        class SGD : public NN::Optimizers::Optimizer {
        public:
            SGD() = default;

            SGD(float momentum);

            void optimize(float learning_rate, float batch_size) override;

            void set_parameters(float* parameters,
                float* gradient,
                int parameters_size) override;

            void initialize() override;

            void reset() override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~SGD();

        private:
            float momentum;
        };
    }
}