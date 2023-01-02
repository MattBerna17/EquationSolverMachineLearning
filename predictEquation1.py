import tensorflow as tf
import numpy as np
from tensorflow import keras


print("-------------------------------------------- EQUATION SOLVER by MattBerna17 on GitHub --------------------------------------------")

c = True            # We assume the user wants to add a couple of values (x, y)
xs = np.array([], dtype=float)     # Create an array of xs and ys
ys = np.array([], dtype=float)

# Take the input from keyboard as long as the user wants
while c:
    x = input("Insert x value: ")
    y = input("Insert y value: ")

    xs = np.append(xs, float(x))
    ys = np.append(ys, float(y))

    response = input("Do you want to continue? (Y or N): ")
    if (response == "Y" or response == "y" or response == "1"):
        c = True
    else:
        c = False
    print()


# Now we create the neural network to elaborate the data from the input
model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

# We define the compilation method of the neural network by using the stochastic gradient descent as the optimization method and the mean squared error as the error method
model.compile(optimizer="sgd", loss="mean_squared_error")

# Now we train the model:
print("Evaluating the input...")
model.fit(xs, ys, epochs=10000)

print("\n\nModel defined:\nInsert other x values to predict y values:\n")
c = True
while c:
    inp1 = input("x: ")
    inp = []
    inp.append(float(inp1))
    print("y: ", end="")
    print(model.predict(inp, verbose=0))
    
    response = input("Do you want to continue? (Y or N): ")
    if (response == "Y" or response == "y" or response == "1"):
        c = True
    else:
        c = False
    print()

print("\n\nGoodbye!")