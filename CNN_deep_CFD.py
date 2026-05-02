#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 18:41:23 2025

@author: himalayagaur
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


## Define a function to create a dataset of simulated fluid velocity 
## along with corresponding parameters

def generate_simulation_data(num_samples):
    
    data = []             #initialize empty lists to store velocity fields ('data') and their associated parameters ('params')
    params = []
    
    
    for _ in range(num_samples):
        inflow_speed = np.random.uniform(0.1, 1.0)
        viscosity = np.random.uniform(0.01, 0.1)
        
        ## Loop to generate 'num_samples' simulation example
        ##for each sample,randomly chooses an inflow speed between 0.1 and 1.0 and a viscosity value between 0.01 and 0.1.
        
        u = inflow_speed * np.ones((64,64))
        v = np.zeros((64, 64))
        
        ## Creates two 64x64 grids:
        ## `u` is filled with the inflow speed, simulating horizontal velocity.
        ## `v` is filled with zeros, so vertical velocity is zero for this simple example.
        
        
        velocity_field = np.stack([u, v], axis=-1)
        
        ## Combines `u` and `v` to form a 2D velocity field with two channels (horizontal and vertical velocity).
        
        data.append(velocity_field)
        params.append([inflow_speed, viscosity])
        
        #stores the velocity field and its parameters in their respective lists.
        
    return np.array(data), np.array(params)
    
    ## Returns two NumPy arrays: one with velocity fields and another with parameters.
    
    ## CNN Generator MODEL
    
    ## Defines a function to create the generator model; `param_dim` is the number of input parameters.
def build_generator(param_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(param_dim,)),
        tf.keras.layers.Dense(64*64*32, activation='relu'),
        tf.keras.layers.Reshape((64, 64, 32)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),  # no upsampling
        tf.keras.layers.Conv2D(2, 3, padding='same')                       # output (64,64,2)
    ])
    return model
    
## Initializes a sequential model where layers are added one after another.
## Fully connected (dense) layer with 256 neurons and ReLU activation, taking a vector of length `param_dim` as input.
## Another fully connected layer expanding to 131,072 neurons (646432), preparing data for reshaping into a spatial feature map.
## Reshapes the flat vector into a 4D tensor representing a 64x64 grid with 32 channels.
## Transposed convolutional layer (also called deconvolution) upsamples spatial dimensions, reducing channels to 16 with a 3x3 filter.
## Final 2D convolution layer producing 2 output channels (velocity components) with a 3x3 kernel; no activation to allow negative or positive values.

##  DIVERGENCE LOSS FUNCTION

def divergence_loss(velocity_field):
    u = velocity_field[..., 0]
    v = velocity_field[..., 1]
    ## Extracts horizontal (`u`) and vertical (`v`) velocity components from the last dimension.
    ## Computes approximate partial derivatives of velocity components in x and y directions using finite differences.
    
    dudx = u[:, :, 1:] - u[:, :, :-1]
    dvdy = v[:, 1:, :] -v[:, :-1, :]
    
    #Crop to common region
    dudx = dudx[:, :-1, :]
    dvdy = dvdy[:, :, :-1]
    
    div = dudx + dvdy
    ## Returns the average squared divergence as the loss value; smaller values signify closer to divergence-free.
    return tf.reduce_mean(tf.square(div))


## TRAINING STEP FUNCTION

param_dim = 2  # inflow speed, viscosity  Set the number of input parameters (here 2: inflow speed and viscosity)

model = build_generator(param_dim)   ## Builds the neural network generator model.

optimizer = tf.keras.optimizers.Adam() ##Creates an Adam optimizer to update model weights during training.

@tf.function()
def train_step(params, true_velocity):
    with tf.GradientTape() as tape:
        pred_velocity = model(params, training=True)

        mse_loss = tf.reduce_mean(tf.square(pred_velocity - true_velocity))
        div_loss = divergence_loss(pred_velocity)
        total_loss = mse_loss + 0.1 * div_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss

## TRAINING LOOP
data, params = generate_simulation_data(1000)
split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]
train_params, test_params = params[:split_idx], params[split_idx:]

batch_size = 32
epochs = 50
for epoch in range(epochs):
    for i in range(0, len(train_data), batch_size):
        x_batch = tf.convert_to_tensor(train_params[i:i+batch_size], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(train_data[i:i+batch_size].astype(np.float32), dtype=tf.float32)
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

## VISUALIZATION OF VELOCITY FIELD

def plot_velocity_field(velocity):
    u = velocity[..., 0]
    v = velocity[..., 1]
    
    Y, X = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    plt.quiver(X, Y, u, v, scale=10)
    plt.show()
                              



































    



















        