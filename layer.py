import numpy as np

# Sigmoid
def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_sigmoid_der(x):
    return f_sigmoid(x)*(1 - f_sigmoid(x))

# Relu
def f_relu(x):
    return np.maximun(x)

def f_relu_der(x):
    return (x > 0).astype(float)

# Gelu
# It was a whim of mine to implement this function (Nicolas)
def f_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

def f_gelu_der(x):
    tanh_term = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * (x**3)))
    return 0.5 * (1 + tanh_term + x * (1 - tanh_term**2) * np.sqrt(2/np.pi) * (1 + 3*0.044715*(x**2)))

# Tanh
def f_tanh(x):
    return np.tanh(x)

def f_tanh_der(x):
    return 1 - (np.tanh(x))**2

class Layer():
    def __init__(self, in_nodes, out_nodes, activation="sigmoid"):
        if activation == "sigmoid":
            self.activation = f_sigmoid
            self.activation_derivative = f_sigmoid_der
        elif activation == "relu":
            self.activation = f_relu
            self.activation_derivative = f_relu_der
        elif activation == "gelu":
            self.activation = f_gelu
            self.activation = f_gelu_der
        elif activation == "tanh":
            self.activation = f_tanh
            self.activation_derivative = f_tanh_der
        else:
            print(f"Activation function {activation} not supported.")
            return

        self.w = np.random.randn(in_nodes, out_nodes)
        self.b = np.zeros((1, out_nodes))

        self.z = None
        self.a = None

        self.activation_name = activation
    
    def forward(self, x):
        self.z = x @ self.w + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, delta, previous_a):
        dz = delta * self.activation(self.z)
        dw = previous_a @ dz / previous_a.shape[0]
        db = np.mean(dz, axis=0, keepdims=True)
        n_delta = dz @ self.w.T
        return dw, db, n_delta