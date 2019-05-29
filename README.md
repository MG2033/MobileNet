# MobileNet
An implementation of `Google MobileNet` introduced in TensorFlow. According to the authors, `MobileNet` is a computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power. It can be used for different applications including: Object-Detection, Finegrain Classification, Face Attributes and Large Scale Geo-Localization.

Link to the original paper: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

This implementation was made to be clearer than TensorFlow original implementation. It was also made to be an example of a common DL software architecture. The weights/biases/parameters from the pretrained ImageNet model that was implemented by TensorFlow are dumped to a dictionary in pickle format file (`pretrained_weights/mobilenet_v1.pkl`) to allow a less restrictive way of loading them.


## Depthwise Separable Convolution
<div align="center">
<img src="https://github.com/MG2033/MobileNet/blob/master/figures/dws.png"><br><br>
</div>

## ReLU6
The paper uses ReLU6 as an activation function. ReLU6 was first introduced in [Convolutional Deep Belief Networks on CIFAR-10](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf) as a ReLU with clipping its output at 6.0.

## Usage
### Main Dependencies
 ```
 Python 3 and above
 tensorflow 1.3.0
 numpy 1.13.1
 tqdm 4.15.0
 easydict 1.7
 matplotlib 2.0.2
 pillow 5.0.0
 ```
### Train and Test
1. Prepare your data, and modify the data_loader.py/DataLoader/load_data() method.
2. Modify the config/test.json to meet your needs.

Note: If you want to test that the model is pretrained and working properly, I've added some test images from different classes in directory 'data/test_images'. All of them are classified correctly.

### Run
```
python3 main.py --config config/test.json

```
The file 'test.json' is just an example of a file. If you run it as is, it will test the model against the images in directory 'data/test_images'. You can create your own configuration file for training/testing.

## Benchmarking
The paper has achieved 569 Mult-Adds. In my implementation, I have achieved approximately 1140 MFLOPS. The paper counts multiplication+addition as one unit. My result verifies the paper as roughly dividing 1140 by 2 is equal to 569 unit.

To calculate the FLOPs in TensorFlow, make sure to set the batch size equal to 1, and execute the following line when the model is loaded into memory.
```
tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')
```
I've already implemented this function. It's called ```calculate_flops()``` in `utils.py`. Use it directly if you want.

## Updates
* Inference and training are working properly.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

