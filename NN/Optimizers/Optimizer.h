#ifndef NN_OPTIMIZERS_OPTIMIZER_H_INCLUDED
#define NN_OPTIMIZERS_OPTIMIZER_H_INCLUDED

#include <cstdio>
#include <vector>

#include "../File.h"

namespace NN {
    namespace Optimizers {
        class Optimizer {
        public:
            virtual void optimize(float learning_rate, float batch_size) = 0;

            virtual void set_parameters(float * parameters,
                                        float * gradient,
                                        int parameters_size) = 0;

            virtual void initialize() = 0;

            virtual void reset() = 0;

            virtual void save(NN::File& file) = 0;

            virtual void load(NN::File& file) = 0;

            virtual ~Optimizer() = default;

        protected:
            float* parameters;
            float* gradient;
            int parameters_size;
        };
    }
}

#endif // NN_OPTIMIZERS_OPTIMIZER_H_INCLUDED
