# StyleChanger
Image-to-image style tranfering solution written with C++ and CUDA based on [this](https://arxiv.org/abs/1508.06576) article and [VGG19](https://keras.io/api/applications/vgg/).

## How to use
Create two 128x128 images:
* "StyleSource.png" - source image of the style.
* "ContentSource.png" - destination image for the style.<br/>
Put these two pictures in the program folder and run the program.<br/>
You'll be able to see the output in the "StyleOutput.png" image file.

## Acknowledgements
* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
* [CImg](https://cimg.eu/), used for processing images.
* [ImageMagick](https://imagemagick.org/), required for CImg to work properly.
