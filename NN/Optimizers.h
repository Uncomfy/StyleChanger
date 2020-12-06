#ifndef NN_OPTIMIZERS_H_INCLUDED
#define NN_OPTIMIZERS_H_INCLUDED

#include "Optimizers/Optimizer.h"

#include "Optimizers/Adam.h"
#include "Optimizers/SGD.h"

namespace NN {
    namespace Optimizers {
        class Types {
        public:
            static NN::Optimizers::Optimizer* getTypeFromId(int id) {
                switch (id) {
                case 0:
                    return new NN::Optimizers::Adam();
                case 1:
                    return new NN::Optimizers::SGD();
                default:
                    return new NN::Optimizers::SGD();
                }
            }

        private:
            Types() = default;
        };
    }
}

#endif // NN_OPTIMIZERS_H_INCLUDED
