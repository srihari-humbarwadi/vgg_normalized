## VGG Normalized
This script rescales weights of `Conv2D` layers such that all the `Conv2D` layer activations of `VGG` model have unit mean.
This is helpful when training models like "style-transfer" that make use of `VGG` based losses.

#### Usage: 
`python3 convert.py --model_name vgg19 --image_dir ILSVRC2012_img_val --output_weights_path normalized_vgg19.h5`


```
@inproceedings{Gatys2016,
  doi = {10.1109/cvpr.2016.265},
  url = {https://doi.org/10.1109/cvpr.2016.265},
  year = {2016},
  month = jun,
  publisher = {{IEEE}},
  author = {Leon A. Gatys and Alexander S. Ecker and Matthias Bethge},
  title = {Image Style Transfer Using Convolutional Neural Networks},
  booktitle = {2016 {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})}
}
```
