#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>

#include "CImg/CImg.h"

#include "NN/ActivationFunctions.h"
#include "NN/File.h"
#include "NN/Layers.h"
#include "NN/Model.h"
#include "NN/Optimizers.h"

using namespace std;

int main()
{
    ios::sync_with_stdio(false);
    srand(time(NULL));

    float change_rate = 1.0f;
    float content_coef = 1e-3f;
    float style_coef = 1.0f;

    vector<float> bgr_mean = { 103.939, 116.779, 123.68 };

    NN::Model model(new NN::Layers::InputLayer(128 * 128 * 3)); // 0
    {
        //-----------------------------------------------------------------------------------

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            128, 128, 3,
            1, 1, 1, 1, 0, 0)); // 1
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            130, 130, 3,
            3, 3, 64)); // 2
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 3

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            128, 128, 64,
            1, 1, 1, 1, 0, 0)); // 4
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            130, 130, 64,
            3, 3, 64)); // 5
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 6

        model.add_layer(new NN::Layers::PoolMaxLayer({ -1 },
            128, 128, 64,
            2, 2, 1)); // 7

    //-----------------------------------------------------------------------------------

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            64, 64, 64,
            1, 1, 1, 1, 0, 0)); // 8
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            66, 66, 64,
            3, 3, 128)); // 9
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 10

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            64, 64, 128,
            1, 1, 1, 1, 0, 0)); // 11
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            66, 66, 128,
            3, 3, 128)); // 12
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 13

        model.add_layer(new NN::Layers::PoolMaxLayer({ -1 },
            64, 64, 128,
            2, 2, 1)); // 14

    //-----------------------------------------------------------------------------------

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            32, 32, 128,
            1, 1, 1, 1, 0, 0)); // 15
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            34, 34, 128,
            3, 3, 256)); // 16
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 17

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            32, 32, 256,
            1, 1, 1, 1, 0, 0)); // 18
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            34, 34, 256,
            3, 3, 256)); // 19
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 20

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            32, 32, 256,
            1, 1, 1, 1, 0, 0)); // 21
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            34, 34, 256,
            3, 3, 256)); // 22
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 23

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            32, 32, 256,
            1, 1, 1, 1, 0, 0)); // 24
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            34, 34, 256,
            3, 3, 256)); // 25
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 26

        model.add_layer(new NN::Layers::PoolMaxLayer({ -1 },
            32, 32, 256,
            2, 2, 1)); // 27

    //-----------------------------------------------------------------------------------

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            16, 16, 256,
            1, 1, 1, 1, 0, 0)); // 28
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            18, 18, 256,
            3, 3, 512)); // 29
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 30

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            16, 16, 512,
            1, 1, 1, 1, 0, 0)); // 31
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            18, 18, 512,
            3, 3, 512)); // 32
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 33

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            16, 16, 512,
            1, 1, 1, 1, 0, 0)); // 34
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            18, 18, 512,
            3, 3, 512)); // 35
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 36

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            16, 16, 512,
            1, 1, 1, 1, 0, 0)); // 37
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            18, 18, 512,
            3, 3, 512)); // 38
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 39

        model.add_layer(new NN::Layers::PoolMaxLayer({ -1 },
            16, 16, 512,
            2, 2, 1)); // 40

    //-----------------------------------------------------------------------------------

        model.add_layer(new NN::Layers::PaddingLayer({ -1 },
            8, 8, 512,
            1, 1, 1, 1, 0, 0)); // 41
        model.add_layer(new NN::Layers::ConvolutionLayer({ -1 },
            10, 10, 512,
            3, 3, 512)); // 42
        model.add_layer(new NN::Layers::ActivationLayer({ -1 }, new NN::AF::ReLU())); // 43

        //-----------------------------------------------------------------------------------

        model.add_layer(new NN::Layers::GramLayer({ 2 }, 64)); // 44
        model.add_layer(new NN::Layers::GramLayer({ 9 }, 128)); // 45
        model.add_layer(new NN::Layers::GramLayer({ 16 }, 256)); // 46
        model.add_layer(new NN::Layers::GramLayer({ 29 }, 512)); // 47
        model.add_layer(new NN::Layers::GramLayer({ 42 }, 512)); // 48

        model.add_layer(new NN::Layers::ConcatLayer({ 32, 44, 45, 46, 47, 48 })); // 49
    }

    //-----------------------------------------------------------------------------------

    cout << "Building." << endl;
    model.build();

    model.set_optimizer(new NN::Optimizers::Adam(0.9f, 0.999f, 1e-8f));
    cout << "Done." << endl << endl;

    cout << "Setting up parameters." << endl;
    FILE* fin = fopen("params.data", "rb");
    int par_size = model.get_parameters_size();
    float* pars = new float[par_size];
    fread(pars, 4, par_size, fin);
    fclose(fin);
    model.set_parameters(pars);
    cout << "Done." << endl << endl;

    cout << "Loading images." << endl;
    int gradient_size = model.get_output_size();
    float* gradient = new float[gradient_size];

    cimg_library::CImg<float> styles("StyleSource.png");
    float* style_data;
    cudaMallocManaged(&style_data, 128 * 128 * 3 * sizeof(float));
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 128; j++) {
            for (int x = 0; x < 128; x++) {
                style_data[(k * 128 + j) * 128 + x] = *styles.data(x, j, 0, 2-k) - bgr_mean[k];
            }
        }
    }

    model.set_input(style_data);
    model.compute();

    float* desired_output = new float[gradient_size];
    memcpy(desired_output, model.get_output_iterator(), gradient_size * sizeof(float));

    cimg_library::CImg<float> content_source("ContentSource.png");
    float* content_data;
    cudaMallocManaged(&content_data, 128 * 128 * 3 * sizeof(float));
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 128; j++) {
            for (int x = 0; x < 128; x++) {
                content_data[(k * 128 + j) * 128 + x] = *content_source.data(x, j, 0, 2 - k) - bgr_mean[k];
            }
        }
    }

    model.set_input(content_data);
    model.compute();
    memcpy(desired_output, model.get_output_iterator(), 131072 * sizeof(float));

    float* output_data;
    cudaMallocManaged(&output_data, 128 * 128 * 3 * sizeof(float));
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 128; j++) {
            for (int x = 0; x < 128; x++) {
                output_data[(k * 128 + j) * 128 + x] = 0.0;
            }
        }
    }

    cout << "Done." << endl;

    float tloss = 0.0f;
    float t_grad;
    int counter;
    int iteration = 1;

    NN::Optimizers::Adam ad_r(0.9f, 0.999f, 1e-8f);
    ad_r.set_parameters(output_data, model.get_input_gradient(), 128 * 128 * 3);
    ad_r.initialize();

    cout << "Optimizing..." << endl;

    while (true) {
        memset(gradient, 0, gradient_size * sizeof(float));
        model.set_input(output_data);
        model.compute();

        tloss = 0.0;
        counter = 0;


        float* output = model.get_output_iterator();
        for (; counter < 131072; counter++) {
            t_grad = 1.0f * content_coef * (output[counter] - desired_output[counter]);
            gradient[counter] += t_grad;
            tloss += abs(t_grad);
        }
        for (; counter < 131072 + 64 * 64; counter++) {
            t_grad = 0.2f * style_coef * (output[counter] - desired_output[counter]) / (64.0f * 64.0f * 128.0f * 128.0f * 128.0f * 128.0f);
            gradient[counter] += t_grad;
            tloss += abs(t_grad);
        }
        for (; counter < 131072 + 64 * 64 + 128 * 128; counter++) {
            t_grad = 0.2f * style_coef * (output[counter] - desired_output[counter]) / (128.0f * 128.0f * 64.0f * 64.0f * 64.0f * 64.0f);
            gradient[counter] += t_grad;
            tloss += abs(t_grad);
        }
        for (; counter < 131072 + 64 * 64 + 128 * 128 + 256 * 256; counter++) {
            t_grad = 0.2f * style_coef * (output[counter] - desired_output[counter]) / (256.0f * 256.0f * 32.0f * 32.0f * 32.0f * 32.0f);
            gradient[counter] += t_grad;
            tloss += abs(t_grad);
        }
        for (; counter < 131072 + 64 * 64 + 128 * 128 + 256 * 256 + 512 * 512; counter++) {
            t_grad = 0.2f * style_coef * (output[counter] - desired_output[counter]) / (512.0f * 512.0f * 16.0f * 16.0f * 16.0f * 16.0f);
            gradient[counter] += t_grad;
            tloss += abs(t_grad);
        }
        for (; counter < 131072 + 64 * 64 + 128 * 128 + 256 * 256 + 512 * 512 * 2; counter++) {
            t_grad = 0.2f * style_coef * (output[counter] - desired_output[counter]) / (512.0f * 512.0f * 8.0f * 8.0f * 8.0f * 8.0f);
            gradient[counter] += t_grad;
            tloss += abs(t_grad);
        }

        model.set_output_gradient(gradient);
        model.backpropagate();

        ad_r.optimize(change_rate, 1.0f);

        cimg_library::CImg<float> image_output(128, 128, 1, 3);

        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 128; j++) {
                for (int x = 0; x < 128; x++) {
                    *image_output.data(x, j, 0, 2 - k) = min(max(output_data[(k * 128 + j) * 128 + x] + bgr_mean[k], 0.0f), 255.0f);
                }
            }
        }

        image_output.save("StyleOutput.png");

        cout << "Iteration " << iteration++ << endl;
    }

    return 0;
}
