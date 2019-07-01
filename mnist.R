###Use CPU
library("keras"); library("sessioninfo")
use_python("/Users/lee/anaconda3/bin/python")

batch_size <- 128
num_classes <- 10
epochs <- 5

img_rows <- 28
img_cols <- 28

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() %>%
layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
input_shape = input_shape) %>%
layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_dropout(rate = 0.25) %>%
layer_flatten() %>%
layer_dense(units = 128, activation = 'relu') %>%
layer_dropout(rate = 0.5) %>%
layer_dense(units = num_classes, activation = 'softmax')

summary(model)

model %>% compile(
loss = loss_categorical_crossentropy,
optimizer = optimizer_adadelta(),
metrics = c('accuracy')
)

system.time({
    model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2
    )
})

#iMac Pro self-compile
#user   system  elapsed
#1537.536 8788.473  335.959

#iMac Pro lakshayg
#user   system  elapsed
#1276.076 5051.262  397.757

#iMac Pro defaults
#user   system  elapsed
#1584.637 5461.211  436.779

###Macbook Pro
#user   system  elapsed
#1751.089  457.561  487.355

#MacBokPro 2018 CPU
#user   system  elapsed
#1244.427  180.420  341.415

###Hal
#user  system  elapsed
#2119.7754  179.570  90.270

###Use GPU
library("keras"); library("sessioninfo")
#use_python("/Users/lee/anaconda3/bin/python")
#use_backend(backend = "plaidml")

batch_size <- 128
num_classes <- 10
epochs <- 5

img_rows <- 28
img_cols <- 28

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)
#k_cast(x_train, dtype="int8")
#k_cast(x_test, dtype="int8")


x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
#k_cast(y_train, dtype="int8")
#k_cast(y_test, dtype="int8")

model <- keras_model_sequential() %>%
layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
input_shape = input_shape) %>%
layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_dropout(rate = 0.25) %>%
layer_flatten() %>%
layer_dense(units = 128, activation = 'relu') %>%
layer_dropout(rate = 0.5) %>%
layer_dense(units = num_classes, activation = 'softmax')

summary(model)

model %>% compile(
loss = loss_categorical_crossentropy,
optimizer = optimizer_adadelta(),
metrics = c('accuracy')
)

system.time({
    model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = 5,
    validation_split = 0.2
    )
})

#iMac Pro Metal int8


#iMac Pro OpenCL
#user  system elapsed
#76.331  88.456  81.397

#iMac Pro Metal
#user  system elapsed
#11.466  12.410  43.466

#MacBook Pro OpenCl IntelHD accuarcy 0
#user  system elapsed
#726.089  32.368 727.070

#MacBook Pro OpenCl
#user  system elapsed
#204.040  18.417 207.960

#MacBook Pro Metal
#user  system elapsed
#121.773  22.589 186.115

#MacBook Pro Metal plaidml .3.5
#user  system elapsed
#19.144  15.798  74.404

#MacBokPro 2018 Metal
#user  system elapsed
#16.467  14.347  62.911

###Hal
#user  system  elapsed
#1401.799  45.319  57.963


###Use Multiple GPUs
library("keras"); library("sessioninfo")
library(reticulate)
#use_python("/Users/lee/anaconda3/bin/python")
#use_backend(backend = "plaidml")
use_condaenv(condaenv='tf_gpu')

batch_size <- 128
num_classes <- 10
epochs <- 5

img_rows <- 28
img_cols <- 28

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)
#k_cast(x_train, dtype="int8")
#k_cast(x_test, dtype="int8")


x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
#k_cast(y_train, dtype="int8")
#k_cast(y_test, dtype="int8")

model <- keras_model_sequential() %>%
layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
input_shape = input_shape) %>%
layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_dropout(rate = 0.25) %>%
layer_flatten() %>%
layer_dense(units = 128, activation = 'relu') %>%
layer_dropout(rate = 0.5) %>%
layer_dense(units = num_classes, activation = 'softmax')

summary(model)

parallel_model <- multi_gpu_model(model, gpus=4)

parallel_model %>% compile(
loss = loss_categorical_crossentropy,
optimizer = optimizer_adadelta(),
metrics = c('accuracy')
)

system.time({
    parallel_model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = 5,
    validation_split = 0.2
    )
})

###MultiGPU Hal
#user  #system  #elapsed
#86.103  13.663  37.164
