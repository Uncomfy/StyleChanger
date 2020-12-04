#ifndef NN_LAYERS_H_INCLUDED
#define NN_LAYERS_H_INCLUDED

#include "NN/Layers/Layer.h"

#include "NN/Layers/ActivationLayer.h"
#include "NN/Layers/ConcatLayer.h"
#include "NN/Layers/ConvolutionLayer.h"
#include "NN/Layers/DenseLayer.h"
#include "NN/Layers/GramLayer.h"
#include "NN/Layers/InputLayer.h"
#include "NN/Layers/InstanceNormLayer.h"
#include "NN/Layers/PaddingLayer.h"
#include "NN/Layers/PoolAvgLayer.h"
#include "NN/Layers/PoolMaxLayer.h"
#include "NN/Layers/RearrangeLayer.h"
#include "NN/Layers/SumLayer.h"
#include "NN/Layers/UpscaleLayer.h"

namespace NN {
	namespace Layers {
        class Types {
        public:
            static NN::Layers::Layer* getTypeFromId(int id) {
                switch (id) {
                case 0:
                    return new NN::Layers::ActivationLayer();
                case 1:
                    return new NN::Layers::ConcatLayer();
                case 2:
                    return new NN::Layers::ConvolutionLayer();
                case 3:
                    return new NN::Layers::DenseLayer();
                case 4:
                    return new NN::Layers::GramLayer();
                case 5:
                    return new NN::Layers::PaddingLayer();
                case 6:
                    return new NN::Layers::PoolAvgLayer();
                case 7:
                    return new NN::Layers::PoolMaxLayer();
                case 8:
                    return new NN::Layers::UpscaleLayer();
                case 9:
                    return new NN::Layers::RearrangeLayer();
                case 10:
                    return new NN::Layers::SumLayer();
                case 11:
                    return new NN::Layers::InstanceNormLayer();
                default:
                    return new NN::Layers::InputLayer();
                }
            }

        private:
            Types() = default;
        };
	}
}
#endif // NN_LAYERS_H_INCLUDED
