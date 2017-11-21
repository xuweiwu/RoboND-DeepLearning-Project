## Project: Deep Learning - Follow Me
### [Rubric Points](https://review.udacity.com/#!/rubrics/1155/view)
---
[//]: # (Image References)

[image1]: ./doc/misc/model_architecture.png

### Network Architecture
The implemented fully converlutional network consists of 3 encoders, an intermediate 1x1 convolutional layer, 3 decoders, and a final convolutional layer. The network achitecture is illustrated in the following figure. 

![image1]
*network architecture*

Each encoder uses two depthwise separable convolutional layers internally to extract feature maps. As mentioned in [1], the depthwise separable convolution can be seen as an extreme version of the Inception module, which decouples the mapping of cross-channels correlations and spatial correlations to make the learning process more efficient. Therefor, separable convolutions can also take advantage of this factoring of the mapping to learn richer representation with less parameters. In each encoder, the first layer expands the channel dimension by a factor of 2 and maintains the 2 spatial dimensions (width and height), while the second layer preserves the channel dimension and downsamples the spatial dimensions by using stride 2. This kind of generic module was learned from the Xception Architecture in [1], where the downsampling was performed instead by a max-pooling layer. 

In the corresponding decoder network, each decoder upsamples its input feature maps by a factor of 2 using bilinear interpolation. Then two separable convolutions are performed making the upsampling process learnable. Unlike the encodr network, both separable convolutions in the decoder module preserve all 3 dimensions of its input maps. Note that two skip connections are established and concatenated to the input layer of the last decoder, which produces dense feature maps with the same spatial dimensions as the input images (160 x 160). The first skip is produced by 2x upsampling the output layer of encoder 1 and the second skip by 4x upsampling the output layer of encoder 2. As explained in [2], adding these skips from encoders with higher spatial resolutions helps the decoders to reproduce finer spatial boundaries in the final segmentation masks. However, the balance between the learned semantic information and the appearance information from the skips should be considered carefully, otherwise the learned prediction features can be overwhelmed in the final segmentation output.

The encoder network and the corresponding decoder network are connected through a 1x1 convolutional layer with filters of spatial size 1x1, which replaces the fully-connected layer in classification convolutional network in order to retain the spatial information of the feature maps. In image classification tasks the extracted feature maps are flattened and fed to the fully-connected layer(s) followed directly by a classifier. During this process the information about spatial coordinates is generally lost. For semantic segmentation tasks, pixelwise classifications are performed upon the dense two-dimensional feature maps, and hence the spatial information is vital for such tasks.

Note that except for the final convolutional layer all separable convolutional layers and the 1x1 convolutional layer are followed by batch normalization and ReLU activation. The included batch normalizations can enhance the convergence performance of the total network and make the training process more efficient. The final convolutional layer uses a softmax classifier to produce a pixelwise classification with 3 classes.

A set of network architectures are compared during the implementation. The two highest final scores are achieved by the same network arhitecture with different depths. As shown in the following table, using deeper network significantly improved the precisions of the final segmentation masks.

network architecture | encoder1 | encoder2 | encoder3 | decoder1 | decoder2 | decoder3 | final IoU |final scores
--- | --- | --- | --- | --- | --- | --- | --- | ---
network-128 | 32 | 64 | 128 | 64 | 32 | 32 | 0.490 | 0.338
network-256 | 64 | 128 | 256 | 128 | 64 | 64 | 0.536 | 0.403

### Data Collection

By observing the single scores for the three group of predictions, one can notice that the score measuring how well the neural network can detect the target from far away has the most significant influence on the final score. As the network might not be trained good enough this kind of predictions by using the default dataset, a large number of false negatives will be recorded and greatly lower down the final score.

Therefore, the main strategy for data collections is to capture more images when the drone is patrolling far away from the target. Through additional data collections the final dataset has 11038 images and corresponding masks. Using the 80/20 rule this dataset was divided into two portions: 8830 images and corresponding masks for the training set and 2208 for the validation set. 

### Hyperparameter Tunning

A set of batch sizes between 20 to 32 was tried during the implementation. By observing the rate of changes in the loss one can determine if the chosen batch size is resonable. If the loss generally bounces back and forth within one epoch, it means that the gradient estimations made by the mini batches have a high variance and the selected batch size might be too slow. Since additonal data are collected for training, the batch size is finally chosen as 32 to take advantage of the larger dataset. The noisy changes in the loss can only be seen in the final 10 epochs, which is mainly due to overfitting of the network.

The learning rate is set to a small value:7.5e-5, since the network has a deep architecture with 14 convolutional layers. The Adam optimizer is set as default optimizer, which can internally take care of the learning rate decay during the learning process.

The number of epochs is set to 50. In the last 10 epochs the loss cannot monotonically decrease due to overfitting. However, the gap between the training and validation accuracy can be maintained at a low and constant level, indicating the weak effect of the overfitting.

The hyperparameters for steps per epoch and validation steps are chosen according to the size of training and validation set while keeping a constant batch size.

### Limitations
The current model is trained by labeled data where the target is marked as red in the images captured by the camera. During the data collection, labeled images are automatically generated for the 3 classes (cam2 for other humans, cam3 for the target, and cam4 for the background), and combined to the single set of mask images for training. Therefor, in order to make the model work well for following another object, new data will need to be collected where the target object and other classes are correctly labeld using similar strategies. 

The network architecture itself can mostly remain unchanged, only the channel depth of the final layer might need to be changed according to a new number of classes.

### Future Enhancements

A major enhancement can be made by collecting more camera images with higher resolutions. As explained above, the network produces relatively bad predictions when the target is far away in the images. In this situation, the amount of total pixels matching the target is very limited, and the network might treat them as noises. With higher resolutions the network can make a better observation of the target and increases the true positives in its predictions as a result.

Improvement in performance of the training might also be made by tunning the regularization parameters.   