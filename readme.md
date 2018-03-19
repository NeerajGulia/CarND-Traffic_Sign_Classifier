# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/actualVsGray.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/testImages.png          "Traffic Signs"
[image5]: ./examples/33_TurnRightAhead.jpg  "Wrongly identified image"
[image6]: ./examples/loss_graph.png  "Loss Graph"
[image7]: ./examples/accuracy_graph.png  "Accuracy Graph"
[image9]: ./examples/VisualizationTrainingData.png "Visualization"

## Rubric Points
---
### Writeup / README

### Data Set Summary & Exploration


I used the python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799 
* The size of the validation set is: 4410 
* The size of test set is : 12630 
* The shape of a traffic sign image is : 32, 32, 3
* The number of unique classes/labels in the data set is : 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distrubuted for each class

![Visualization of Training data][image9]


### Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Grayscale
As a first step, I decided to convert the images to grayscale because color is not a significant differentiator for traffic sign classification. Moreover converting the image to grayscale will reduce the channels from 3 to 1 thus faster execution.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

#### Normalization
As a last step, I normalized the image data to make the computations easier and faster.

#### Augmentation
I did not choose to do Augmentation, since without the Augmentation I was getting good accuracy. 

But augmentation should be used whenever the training data is less or the distribution of the data is not even. Like if some classes have higher number of training data while some has less. Using Augmentation the training data can be increased

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                   | Description                                    | Output          |
|-------------------------|------------------------------------------------|-----------------|
| Conv Layer 1            | With Valid Padding, stride 1 and filter size 3 | (?, 30, 30, 6)  |
| Relu                    | Activation function                            |                 |
| Conv Layer 2            | With Valid Padding, stride 1 and filter size 3 | (?, 28, 28, 16) |
| Max Pool                | 2 stide to reduce the output to half           | (?, 14, 14, 16) |
| Relu                    | Activation function                            |                 |
| Drop Out                | with value .5 in Training and 1 in evaluation  |                 |
| Conv Layer 3            | With Valid Padding, stride 1 and filter size 3 | (?, 12, 12, 26) |
| Max Pool                | 2 stide to reduce the output to half           | (?, 6, 6, 26)   |
| Relu                    | Activation function                            |                 |
| Drop Out                | with value .5 in Training and 1 in evaluation  |                 |
| Conv Layer 4            | With Valid Padding, stride 1 and filter size 5 | (?, 6, 6, 42)   |
| Flatten                 | Flattened the Conv layer 4                     | (?, 1512)       |
| Fully connected Layer 1 |                                                | (?, 320)        |
| Relu                    | Activation function                            |                 |
| Drop Out                | with value .7 in Training and 1 in evaluation  |                 |
| Fully connected Layer 2 |                                                | (?, 150)        |
| Relu                    | Activation function                            |                 |
| Drop Out                | with value .7 in Training and 1 in evaluation  |                 |
| Fully connected Layer 3 |                                                | (?, 80)         |
| Relu                    | Activation function                            |                 |
| Drop Out                | with value .7 in Training and 1 in evaluation  |                 |
| Fully connected Layer 4 |                                                | (?, 43)         |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used followings:
1. Epoch = 90
2. Batch Size = 128
3. Learning Rate = 0.001
4. Optimizer Algorithm: Adam Optimizer 
5. Cross entropy = Softmax cross entropy with logits
6. loss operation = reduce mean of cross entropy
7. Prediction = Argmax
8. Accuracy = Reduced mean
9. mean = 0
10. Sigma = 0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of : 0.99
* validation set accuracy of : 0.983
* test set accuracy of : 0.969

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?   
  LeNet was chosen as the first architecture, because it was a tried and tested architecture for identifying the numerics

* What were some problems with the initial architecture?   
  The LeNet architecture was very simple for the traffic signs.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  Added two additional conv layers and two Fully connected layers. Also added dropouts to reduce the overfitting. Used normalization and conversion of images to grayscale.

* Which parameters were tuned? How were they adjusted and why?
  Initially I choose same value of keep probe for conv and fully connected layers. But later while doing trial and error settled for different keep probe values for conv (.5) and fully connnected layers (.7).
  Also, tried many iterations with different batch sizes and epoch and finally settled with 90 Epoch and 128 batch size.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  The most important decision was to increase the layers and to add dropout layer in between. Normalization and conversion to gray scale also played an important role in achieving the good accuracy.
 
## Evaluation on Training, Validation

![Loss Graph][image6]
![Accuracy Graph][image7]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are few German traffic signs that I found on the web:

![Traffic Signs][image4] 

#### Output of evaluation during training
EPOCH 1, T Loss: 3.407, V Loss: 3.487, T Acc: 0.094, V Acc: 0.082      
EPOCH 2, T Loss: 2.152, V Loss: 2.260, T Acc: 0.354, V Acc: 0.326      
EPOCH 3, T Loss: 1.097, V Loss: 1.188, T Acc: 0.632, V Acc: 0.589      
EPOCH 4, T Loss: 0.603, V Loss: 0.674, T Acc: 0.804, V Acc: 0.771      
EPOCH 5, T Loss: 0.402, V Loss: 0.457, T Acc: 0.879, V Acc: 0.857      
EPOCH 6, T Loss: 0.272, V Loss: 0.321, T Acc: 0.907, V Acc: 0.887      
EPOCH 7, T Loss: 0.201, V Loss: 0.270, T Acc: 0.939, V Acc: 0.905      
EPOCH 8, T Loss: 0.151, V Loss: 0.209, T Acc: 0.946, V Acc: 0.916      
EPOCH 9, T Loss: 0.121, V Loss: 0.175, T Acc: 0.961, V Acc: 0.939      
EPOCH 10, T Loss: 0.105, V Loss: 0.165, T Acc: 0.967, V Acc: 0.943     
EPOCH 11, T Loss: 0.087, V Loss: 0.162, T Acc: 0.965, V Acc: 0.939     
EPOCH 12, T Loss: 0.076, V Loss: 0.139, T Acc: 0.968, V Acc: 0.947     
EPOCH 13, T Loss: 0.072, V Loss: 0.147, T Acc: 0.972, V Acc: 0.948     
EPOCH 14, T Loss: 0.063, V Loss: 0.121, T Acc: 0.975, V Acc: 0.955     
EPOCH 15, T Loss: 0.059, V Loss: 0.128, T Acc: 0.980, V Acc: 0.953     
EPOCH 16, T Loss: 0.050, V Loss: 0.121, T Acc: 0.979, V Acc: 0.958     
EPOCH 17, T Loss: 0.046, V Loss: 0.114, T Acc: 0.979, V Acc: 0.960     
EPOCH 18, T Loss: 0.040, V Loss: 0.127, T Acc: 0.981, V Acc: 0.957     
EPOCH 19, T Loss: 0.039, V Loss: 0.111, T Acc: 0.981, V Acc: 0.960     
EPOCH 20, T Loss: 0.039, V Loss: 0.139, T Acc: 0.980, V Acc: 0.953     
EPOCH 21, T Loss: 0.032, V Loss: 0.115, T Acc: 0.985, V Acc: 0.958     
EPOCH 22, T Loss: 0.030, V Loss: 0.120, T Acc: 0.983, V Acc: 0.958     
EPOCH 23, T Loss: 0.025, V Loss: 0.086, T Acc: 0.992, V Acc: 0.971     
EPOCH 24, T Loss: 0.024, V Loss: 0.100, T Acc: 0.993, V Acc: 0.965     
EPOCH 25, T Loss: 0.019, V Loss: 0.084, T Acc: 0.996, V Acc: 0.975     
EPOCH 26, T Loss: 0.018, V Loss: 0.120, T Acc: 0.995, V Acc: 0.967     
EPOCH 27, T Loss: 0.014, V Loss: 0.079, T Acc: 0.997, V Acc: 0.978     
EPOCH 28, T Loss: 0.011, V Loss: 0.116, T Acc: 0.998, V Acc: 0.970     
EPOCH 29, T Loss: 0.012, V Loss: 0.077, T Acc: 0.997, V Acc: 0.977     
EPOCH 30, T Loss: 0.013, V Loss: 0.115, T Acc: 0.996, V Acc: 0.966     
EPOCH 31, T Loss: 0.011, V Loss: 0.121, T Acc: 0.996, V Acc: 0.968     
EPOCH 32, T Loss: 0.008, V Loss: 0.121, T Acc: 0.997, V Acc: 0.972     
EPOCH 33, T Loss: 0.007, V Loss: 0.101, T Acc: 0.999, V Acc: 0.974     
EPOCH 34, T Loss: 0.007, V Loss: 0.090, T Acc: 0.998, V Acc: 0.976     
EPOCH 35, T Loss: 0.007, V Loss: 0.102, T Acc: 0.999, V Acc: 0.971     
EPOCH 36, T Loss: 0.005, V Loss: 0.069, T Acc: 0.999, V Acc: 0.980     
EPOCH 37, T Loss: 0.005, V Loss: 0.066, T Acc: 0.999, V Acc: 0.982     
EPOCH 38, T Loss: 0.005, V Loss: 0.059, T Acc: 0.999, V Acc: 0.982     
EPOCH 39, T Loss: 0.004, V Loss: 0.078, T Acc: 0.999, V Acc: 0.979     
EPOCH 40, T Loss: 0.006, V Loss: 0.105, T Acc: 0.998, V Acc: 0.971     
EPOCH 41, T Loss: 0.006, V Loss: 0.095, T Acc: 0.998, V Acc: 0.975     
EPOCH 42, T Loss: 0.003, V Loss: 0.075, T Acc: 0.999, V Acc: 0.978     
EPOCH 43, T Loss: 0.011, V Loss: 0.208, T Acc: 0.996, V Acc: 0.955     
EPOCH 44, T Loss: 0.004, V Loss: 0.077, T Acc: 0.999, V Acc: 0.977     
EPOCH 45, T Loss: 0.003, V Loss: 0.059, T Acc: 0.999, V Acc: 0.982     
EPOCH 46, T Loss: 0.005, V Loss: 0.072, T Acc: 0.998, V Acc: 0.980     
EPOCH 47, T Loss: 0.003, V Loss: 0.098, T Acc: 0.999, V Acc: 0.975     
EPOCH 48, T Loss: 0.005, V Loss: 0.123, T Acc: 0.999, V Acc: 0.971     
EPOCH 49, T Loss: 0.003, V Loss: 0.076, T Acc: 0.999, V Acc: 0.980     
EPOCH 50, T Loss: 0.002, V Loss: 0.068, T Acc: 0.999, V Acc: 0.981     
EPOCH 51, T Loss: 0.004, V Loss: 0.129, T Acc: 0.999, V Acc: 0.970     
EPOCH 52, T Loss: 0.002, V Loss: 0.090, T Acc: 1.000, V Acc: 0.980     
EPOCH 53, T Loss: 0.003, V Loss: 0.073, T Acc: 0.999, V Acc: 0.982     
EPOCH 54, T Loss: 0.002, V Loss: 0.096, T Acc: 0.999, V Acc: 0.977     
EPOCH 55, T Loss: 0.002, V Loss: 0.090, T Acc: 0.999, V Acc: 0.978     
EPOCH 56, T Loss: 0.003, V Loss: 0.114, T Acc: 0.999, V Acc: 0.973     
EPOCH 57, T Loss: 0.002, V Loss: 0.090, T Acc: 1.000, V Acc: 0.978     
EPOCH 58, T Loss: 0.001, V Loss: 0.090, T Acc: 1.000, V Acc: 0.980     
EPOCH 59, T Loss: 0.002, V Loss: 0.089, T Acc: 0.999, V Acc: 0.976     
EPOCH 60, T Loss: 0.002, V Loss: 0.074, T Acc: 1.000, V Acc: 0.981     
EPOCH 61, T Loss: 0.001, V Loss: 0.105, T Acc: 1.000, V Acc: 0.976     
EPOCH 62, T Loss: 0.002, V Loss: 0.125, T Acc: 0.999, V Acc: 0.973     
EPOCH 63, T Loss: 0.002, V Loss: 0.129, T Acc: 0.999, V Acc: 0.972     
EPOCH 64, T Loss: 0.002, V Loss: 0.084, T Acc: 1.000, V Acc: 0.980     
EPOCH 65, T Loss: 0.001, V Loss: 0.084, T Acc: 1.000, V Acc: 0.979     
EPOCH 66, T Loss: 0.002, V Loss: 0.156, T Acc: 1.000, V Acc: 0.975     
EPOCH 67, T Loss: 0.002, V Loss: 0.105, T Acc: 0.999, V Acc: 0.976     
EPOCH 68, T Loss: 0.001, V Loss: 0.125, T Acc: 1.000, V Acc: 0.975     
EPOCH 69, T Loss: 0.001, V Loss: 0.098, T Acc: 1.000, V Acc: 0.979     
EPOCH 70, T Loss: 0.001, V Loss: 0.148, T Acc: 1.000, V Acc: 0.975     
EPOCH 71, T Loss: 0.002, V Loss: 0.142, T Acc: 1.000, V Acc: 0.974     
EPOCH 72, T Loss: 0.002, V Loss: 0.102, T Acc: 0.999, V Acc: 0.980     
EPOCH 73, T Loss: 0.001, V Loss: 0.083, T Acc: 1.000, V Acc: 0.982     
EPOCH 74, T Loss: 0.003, V Loss: 0.067, T Acc: 0.999, V Acc: 0.983     
EPOCH 75, T Loss: 0.002, V Loss: 0.084, T Acc: 1.000, V Acc: 0.978     
EPOCH 76, T Loss: 0.001, V Loss: 0.080, T Acc: 1.000, V Acc: 0.983     
EPOCH 77, T Loss: 0.001, V Loss: 0.091, T Acc: 1.000, V Acc: 0.978     
EPOCH 78, T Loss: 0.001, V Loss: 0.079, T Acc: 1.000, V Acc: 0.981     
EPOCH 79, T Loss: 0.001, V Loss: 0.108, T Acc: 1.000, V Acc: 0.978     
EPOCH 80, T Loss: 0.001, V Loss: 0.116, T Acc: 1.000, V Acc: 0.978     
EPOCH 81, T Loss: 0.002, V Loss: 0.156, T Acc: 1.000, V Acc: 0.970     
EPOCH 82, T Loss: 0.001, V Loss: 0.115, T Acc: 1.000, V Acc: 0.977     
EPOCH 83, T Loss: 0.001, V Loss: 0.122, T Acc: 1.000, V Acc: 0.975     
EPOCH 84, T Loss: 0.001, V Loss: 0.129, T Acc: 1.000, V Acc: 0.975     
EPOCH 85, T Loss: 0.001, V Loss: 0.121, T Acc: 1.000, V Acc: 0.976     
EPOCH 86, T Loss: 0.002, V Loss: 0.179, T Acc: 0.999, V Acc: 0.975     
EPOCH 87, T Loss: 0.001, V Loss: 0.092, T Acc: 1.000, V Acc: 0.983     
EPOCH 88, T Loss: 0.001, V Loss: 0.102, T Acc: 1.000, V Acc: 0.978     
EPOCH 89, T Loss: 0.001, V Loss: 0.099, T Acc: 1.000, V Acc: 0.980     
EPOCH 90, T Loss: 0.002, V Loss: 0.076, T Acc: 0.999, V Acc: 0.983     


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Prediction Percentage -> 1, 2, 3 predictions
__________________________
01 (100 000 000)  ->  (1 2 5)        
02 (100 000 000)  ->  (24 27 28)     
03 (100 000 000)  ->  (38 17 13)     
04 (100 000 000)  ->  (12 40  2)     
05 (100 000 000)  ->  (31 21 25)     
06 (100 000 000)  ->  (23 19 20)     
07 (097 002 000)  ->  (5 2 3)        
08 (100 000 000)  ->  (22 25 29)     
09 (100 000 000)  ->  (15 12 35)     
10 (100 000 000)  ->  (14  1  2)     
11 (100 000 000)  ->  (40 12  7)     
12 (100 000 000)  ->  (14  1  2)     
13 (100 000 000)  ->  (33 35 37)     
14 (100 000 000)  ->  (13  3  2)     
15 (100 000 000)  ->  (13  0  1)     
16 (100 000 000)  ->  (12 40  0)     
17 (061 027 011)  ->  (20 26 18)     
18 (100 000 000)  ->  (35 33  3)     
19 (100 000 000)  ->  (18 26 27)     
20 (100 000 000)  ->  (18 26 27)     
21 (100 000 000)  ->  (34 38 35)     

The model was able to correctly guess 20 out of the 21 traffic signs, which gives an accuracy of 95%.

wrongly identifid images:
Image wrongly identified as:  20 while actual image is:  33    
![Wrongly identified image][image5]


| Image                   | Prediction                       | Result  |
|-------------------------|----------------------------------|---------|
| SpeedLimit30            | SpeedLimit30                     | True    |
| NarrowRoadOnRight       | NarrowRoadOnRight                | True    |
| KeepRight               | KeepRight                        | True    |
| PriorityRoad            | PriorityRoad                     | True    |
| WildAnimalsCrossing     | WildAnimalsCrossing              | True    |
| SlipperyRoad            | SlipperyRoad                     | True    |
| SpeedLimit80            | SpeedLimit80                     | True    |
| BumpyRoad               | BumpyRoad                        | True    |
| NoVehicles              | NoVehicles                       | True    |
| Stop                    | Stop                             | True    |
| RoundAboutMandatory     | RoundAboutMandatory              | True    |
| Stop1                   | Stop1                            | True    |
| TurnRightAhead          | TurnRightAhead                   | True    |
| Yield1                  | Yield1                           | True    |
| Yield                   | Yield                            | True    |
| PriorityRoad1           | PriorityRoad1                    | True    |
| TurnRightAhead          | Dangerous curve to the right     | False   |
| AheadOnly               | AheadOnly                        | True    |
| GeneralCaution1         | GeneralCaution1                  | True    |
| GeneralCaution          | GeneralCaution                   | True    |
| TurnLeftAhead           | TurnLeftAhead                    | True    |

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Almost all images had above 90% accuracy
