
# Image classification on CFAR10 dataset using Normalization

### Three types of normalization are implemeted  here
#### Batch Normalization
#### Layer Normalization
#### Group Normalization

### Files

#### Model.py 
##### 1. file contains the model class which has the structure of the model
##### 2. File contains train and test functions

#### Utils.py
##### 1. File contains transfroms applied on train and test data


#### Model Summary

Requirement already satisfied: torchsummary in /usr/local/lib/python3.10/dist-packages (1.5.1)
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
         GroupNorm-2           [-1, 16, 32, 32]              32
           Dropout-3           [-1, 16, 32, 32]               0
            Conv2d-4           [-1, 16, 32, 32]           2,304
         GroupNorm-5           [-1, 16, 32, 32]              32
           Dropout-6           [-1, 16, 32, 32]               0
            Conv2d-7            [-1, 8, 32, 32]             128
         MaxPool2d-8            [-1, 8, 16, 16]               0
            Conv2d-9           [-1, 24, 16, 16]           1,728
        GroupNorm-10           [-1, 24, 16, 16]              48
          Dropout-11           [-1, 24, 16, 16]               0
           Conv2d-12           [-1, 24, 16, 16]           5,184
        GroupNorm-13           [-1, 24, 16, 16]              48
          Dropout-14           [-1, 24, 16, 16]               0
           Conv2d-15           [-1, 24, 16, 16]           5,184
        GroupNorm-16           [-1, 24, 16, 16]              48
          Dropout-17           [-1, 24, 16, 16]               0
           Conv2d-18            [-1, 8, 16, 16]             192
        MaxPool2d-19              [-1, 8, 8, 8]               0
           Conv2d-20             [-1, 32, 8, 8]           2,304
        GroupNorm-21             [-1, 32, 8, 8]              64
          Dropout-22             [-1, 32, 8, 8]               0
           Conv2d-23             [-1, 32, 8, 8]           9,216
        GroupNorm-24             [-1, 32, 8, 8]              64
          Dropout-25             [-1, 32, 8, 8]               0
           Conv2d-26             [-1, 32, 8, 8]           9,216
        GroupNorm-27             [-1, 32, 8, 8]              64
          Dropout-28             [-1, 32, 8, 8]               0
        AvgPool2d-29             [-1, 32, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             320
================================================================
Total params: 36,608
Trainable params: 36,608
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.41
Params size (MB): 0.14
Estimated Total Size (MB): 1.56
----------------------------------------------------------------
