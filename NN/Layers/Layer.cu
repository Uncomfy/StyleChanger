#include "NN/Layers/Layer.h"

using namespace std;

namespace NN {
    namespace Layers {
        void Layer::set_parameters_iterator(float* parameters,
                                            float* gradient) {
            this->parameters = parameters;
            this->gradient = gradient;
        }

        size_t Layer::get_output_size() {
            return output_size;
        }

        float * Layer::get_output_iterator() {
            return output;
        }

        float * Layer::get_output_gradient_iterator() {
            return output_gradient;
        }

        vector<int> Layer::get_dependencies() {
            return dependencies;
        }

        void Layer::nullify_gradient() {
            memset(output_gradient, 0, output_size * sizeof(float));
        }

        void Layer::save_dependencies(NN::File& file) {
            int dep_size = dependencies.size();
            file.save(dep_size);
            for (int i = 0; i < dep_size; i++) {
                file.save(dependencies[i]);
            }
        }

        void Layer::load_dependencies(NN::File& file) {
            int dep_size;
            file.load(dep_size);
            dependencies.resize(dep_size);
            for (int i = 0; i < dep_size; i++) {
                file.load(dependencies[i]);
            }
        }

        Layer::~Layer() {
            cudaFree(output);

            cudaFree(output_gradient);

            dependencies.clear();
            dependencies.shrink_to_fit();
        }
    }
}
