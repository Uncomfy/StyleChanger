#ifndef NN_OPTIMIZERS_ADAM_H_INCLUDED
#define NN_OPTIMIZERS_ADAM_H_INCLUDED

#include "Optimizer.h"

namespace NN {
    namespace Optimizers {
        class Adam : public NN::Optimizers::Optimizer {
        public:
            Adam() = default;

            Adam(float beta1,
                 float beta2,
                 float epsilon);

            void optimize(float learning_rate, float batch_size) override;

            void set_parameters(float* parameters,
                                float* gradient,
                                int parameters_size) override;

            void initialize() override;

            void reset() override;

            void save(NN::File& file) override;

            void load(NN::File& file) override;

            ~Adam();

        private:
            float beta1;
            float beta2;
            float epsilon;
            int operation_id = 1;

            float* mean_gradient;
            float* mean_squared_gradient;
        };
    }
}

#endif // NN_OPTIMIZERS_ADAM_H_INCLUDED
