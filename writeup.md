[//]: # (Image References)

[image1]: ./examples/train_dist.png "Train Example Class distribution"
[image2]: ./examples/first5.png "Test Data 1"
[image3]: ./examples/second5.png "Test Data 2"
[image4]: ./examples/inference.png "Inference"


#**Traffic Sign Recognition** 

## 1. Dataset Summary and Exploration

The German Traffic Signs Dataset given contains:

* 34799 training examples;
* 12630 testing examples;
* 4410 validation examples;

The Image shape is (32, 32, 3) and the number of different traffic signs is 43 (number of classes).

The training dataset is not very well distributed, having much more examples of certain classes than the others, as the following graph shows.

![alt text][image1]

## 2. Design and Test a Model Architecture

#### 2.1 Preprocessing
The pre-processing is just normalization.
This is done in the 4th code cell using cv2.normalize() function.
The idea behind it is that all images should contribute equaly to the training. Without normalization, images with higher brightness would contribute differently to the gradient than those with lower brightness, this is equivalent to adjust the learning rate for each example. Instead the input is scaled and centered to zero. Centering around zero is important for convergence of the bias.
I didn't use grayscale because from intuition, color information is important to classify traffic signs.

#### 2.2 Model Architecture
Following the suggestion, I continued with LeNet-5 architecture, adapting the input layer to 32x32x3 images and the output to 43 possible classes.

Later decided to increase the fully connected layers and increase the dropout probability to 0.5 to make the net generalize better.

#### 2.3 Model Training
The hyper parameters were mostly kept, BATCH_SIZE = 128 and learning rate = 0.001.
Only the number of EPOCHS was increased to 100. On my last implementation, with only 10 epochs, it seemed that the net was still converging to a better solution when training stopped. This can be seen on the results of 9th code cell.
After training it looks that 30 to 50 ephocs would have been enough for this net. 

#### 2.4 Solution Approach
First Training yielded already 93% accuracy in the validation set.
To improve it, I tried using the methods explained in the classes, dropout and inception.
The dropout worked quite ok with a probability of dropping of 40%, increasing the validation accuracy to 94%
Unfortunately I was not able to implement inception out of the box, and as I'm really behind schedule decided to quit on that option.

On test set this model did not perform as well, having only 92.5% accuracy. This let me to thing the model was not generalizing good enough and I increased the size of the fully connected layers and the probability of dropout to 50% the model reaches accuracy of 96% on validation set and 94.3% on test set.

The final model has the following archiecture:

| Layer         		|     Description | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 |
| RELU					| |
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 |
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x20 |
 RELU					| |
 | Max pooling	      	| 2x2 stride,  outputs 5x5x20 |
| Fully connected		| Input 500, output 140|
| Fully connected		| Input 140, output 84|
| Fully connected		| Input 84, output 43|

## 3. Test a Model on New Images

#### 3.1 Acquiring New Images
I acquired and treated (scale and crop) the images to make them 32x32x3.
The first 5 pictures I added are the following:
![alt text][image2]
The result was 100% correct, so I added more 5:
![alt text][image3]

#### 3.2 Performance on New Images
The performance, was 100% on first 5 images and including all images is 90% because it fails the "No Entry" sign which is confused with a "Priority road" sign. This would be a pretty dangerous mistake in a real life situation.
The confusion is quite strange because both signs have different colors and different shapes, something not very intuitive.

The confusion is probably related to the size of the "No entry" sign in the example. The sign is sideways and quite far away in the image, compared to the examples in the training set which are more centered.

```
One may say the model is overfitting since the performance on the test set was superior (94%) and now is only 90%, but this 90% is actually not representative since the set of new images is very small.
```

* 2 - Speed limit (50km/h)
* 35 - Ahead only
* 1 - Speed limit (30km/h)
* 13 - Yield
* 25 - Road work
* 14 - Stop
* 12 - Priority road
* 12 - Priority road
* 20 - Dangerous curve to the right
* 28 - Children crossing
![alt text][image4]

#### 3.3 Model Certainty - Softmax Probabilities 

The model is pretty sure on all its predictions, despite the fact that some are not correct. 

* It predicts for the first sign, classID=2, which is "Speed Limit 50" and softmax probabilities are [1, 0, 0, 0, 0] for classes [ 2,  0,  1, 3,  4] so the model is 100% sure of its decision. It has similar behavior for all the other images.

