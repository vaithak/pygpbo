import pygpbo
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

params_dict = {
    'l1_f': np.arange(4,33),
    'l1_k': np.arange(2,6),
    'l2_f': np.arange(4,33),
    'l2_k': np.arange(2,6),
    'actn': ['elu','relu','selu','sigmoid','tanh'],
    'Drop': (0, 0.5),
}
discrete_vars = ['l1_f', 'l1_k', 'l2_f', 'l2_k', 'actn']
discrete_to_num = {}
for var in discrete_vars:
    options = params_dict[var]
    n = len(options)
    val = np.linspace(1/(2*n),(2*n-1) / (2*n),n) 
    options = {options[x]: val[x] for x in range(n)}
    discrete_to_num[var] = options

print(discrete_to_num)

def objective_NN(l1_f=8, l1_k=3, l2_f=8, l2_k=3, actn='relu', Drop=0.5, batch_size=128, epochs=15):
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(l1_f, kernel_size=(l1_k, l1_k), activation=actn),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(l2_f, kernel_size=(l2_k, l2_k), activation=actn),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(Drop),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]

def get_actual_params(l1_f, l1_k, l2_f, l2_k, actn, Drop):
    l1_f = list(discrete_to_num['l1_f'].keys())[Closest(list(discrete_to_num['l1_f'].values()), l1_f)]
    l1_k = list(discrete_to_num['l1_k'].keys())[Closest(list(discrete_to_num['l1_k'].values()), l1_k)]
    l2_f = list(discrete_to_num['l2_f'].keys())[Closest(list(discrete_to_num['l2_f'].values()), l2_f)]
    l2_k = list(discrete_to_num['l2_k'].keys())[Closest(list(discrete_to_num['l2_k'].values()), l2_k)]
    actn = list(discrete_to_num['actn'].keys())[Closest(list(discrete_to_num['actn'].values()), actn)]

    return l1_f, l1_k, l2_f, l2_k, actn, Drop

# converts params from (0, 1) -> Discrete space for discrete variables
def F(l1_f, l1_k, l2_f, l2_k, actn, Drop):
    l1_f, l1_k, l2_f, l2_k, actn, Drop = get_actual_params(l1_f, l2_f, l1_k, l2_k, actn, Drop)
    return objective_NN(l1_f, l1_k, l2_f, l2_k, actn, Drop)

def Closest(l, k):
    for i in range(len(l)-1):
        if abs(l[i+1]-k) >=  abs(l[i]-k):
            return i
    return -1

# Discrete params will be mapped to (0, 1)
bounds = {'l1_f': (0, 1), 'l1_k': (0, 1), 'l2_f': (0, 1), 'l2_k': (0, 1), 'actn': (0, 1), 'Drop': (0, 0.5)}
optimizer = pygpbo.BayesOpt(F, bounds)
optimizer.add_custom_points([
                             {'l1_f': 0.1, 'l1_k': 0.1, 'l2_f': 0.1, 'l2_k': 0.1, 'actn': 0.1, 'Drop': 0.2},
                             {'l1_f': 0.4, 'l1_k': 0.4, 'l2_f': 0.4, 'l2_k': 0.4, 'actn': 0.8, 'Drop': 0.2},
                            ])

optimizer.maximize(n_iter=30)
print("Optimum point and value: ", optimizer.max_sample)
print(get_actual_params(**optimizer.max_sample[0]))

# Output seen: 
"""
Optimum point and value:  ({'Drop': 0.0, 'actn': 'elu, 'l1_f': 27, 'l1_k': 5, 'l2_f': 32, 'l2_k': 3}, 
                        array([0.90060002]))
"""


