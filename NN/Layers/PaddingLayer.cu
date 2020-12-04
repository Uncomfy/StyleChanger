#include "NN/CUDA/CUDA_func.h"
#include "NN/Layers/PaddingLayer.h"

using namespace std;

namespace NN {
    namespace Layers {
        PaddingLayer::PaddingLayer(std::vector<int> dependencies,
                                   int width,
                                   int height,
                                   int depth,
                                   int left, int right,
                                   int top, int bottom,
                                   int front, int back) {
            this->dependencies = dependencies;
            this->width = width;
            this->height = height;
            this->depth = depth;
            this->left = left;
            this->right = right;
            this->top = top;
            this->bottom = bottom;
            this->front = front;
            this->back = back;

            output_size = (left + width + right) * (top + height + bottom) * (front + depth + back);
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        void PaddingLayer::compute() {
            float * prev_layer_output = prev_layer->get_output_iterator();

            /*for(int k = 0; k < depth; k++) {
                for(int j = 0; j < height; j++) {
                    for(int i = 0; i < width; i++) {
                        output[get_output_offset(i, j, k)] = prev_layer_output[get_input_offset(i, j, k)];
                    }
                }
            }*/

            int block_size = (width*height*depth + 511) / 512;
            NN::CUDA::compute_padding_layer <<<block_size, 512 >>> (prev_layer_output, output,
                                                                   width, height, depth,
                                                                   left+width+right, top+height+bottom, front+depth+back,
                                                                   left, top, front,
                                                                   width*height*depth);
            cudaDeviceSynchronize();
        }

        void PaddingLayer::backpropagate() {
            float * prev_layer_output_gradient = prev_layer->get_output_gradient_iterator();

            /*for(int k = 0; k < depth; k++) {
                for(int j = 0; j < height; j++) {
                    for(int i = 0; i < width; i++) {
                        prev_layer_output_gradient[get_input_offset(i, j, k)] = output_gradient[get_output_offset(i, j, k)];
                    }
                }
            }*/

            
            int block_size = (width*height*depth + 511) / 512;
            NN::CUDA::backprop_padding_layer <<<block_size, 512 >>> (prev_layer_output_gradient, output_gradient,
                                                                   width, height, depth,
                                                                   left+width+right, top+height+bottom, front+depth+back,
                                                                   left, top, front,
                                                                   width*height*depth);
            cudaDeviceSynchronize();
        }

        int PaddingLayer::get_parameters_size() {
            return 0;
        }

        void PaddingLayer::update_dependencies(std::vector<NN::Layers::Layer *> layer_dependencies) {
            prev_layer = layer_dependencies[0];
        }

        void PaddingLayer::save(NN::File& file) {
            int id = 5;
            file.save(id);

            save_dependencies(file);
            file.save(width);
            file.save(height);
            file.save(depth);
            file.save(left);
            file.save(right);
            file.save(top);
            file.save(bottom);
            file.save(front);
            file.save(back);
        }

        void PaddingLayer::load(NN::File& file) {
            load_dependencies(file);
            file.load(width);
            file.load(height);
            file.load(depth);
            file.load(left);
            file.load(right);
            file.load(top);
            file.load(bottom);
            file.load(front);
            file.load(back);

            output_size = (left + width + right) * (top + height + bottom) * (front + depth + back);
            cudaMallocManaged(&output, output_size * sizeof(float));
            cudaMallocManaged(&output_gradient, output_size * sizeof(float));
            cudaMemset(output, 0.0f, output_size * sizeof(float));
            cudaMemset(output_gradient, 0.0f, output_size * sizeof(float));
        }

        PaddingLayer::~PaddingLayer() = default;


        int PaddingLayer::get_input_offset(int i, int j, int k) {
            return (k*height + j)*width + i;
        }

        int PaddingLayer::get_output_offset(int i, int j, int k) {
            return ((k+front)*height + (j+top))*width + (i+left);
        }
    }
}
