
def mse(ws, model, X, Y):
    N = len(X)
    err = 0.0
    for i in range(N):
        xi = X[i]
        yi = model(ws, xi)
        err += (Y[i] - yi) ** 2
    return err / N


def grad_mse(ws, model, gradients, X, Y):
    """
    Manually compute the gradient of MSE loss w.r.t. weights ws.

    Args:
        ws (np.ndarray): weight vector (shape: Mx1 or flat)
        model (callable): model(ws, x) → scalar prediction
        gradients (list of callables): gradients[j](ws, x) → ∂model/∂ws[j]
        X (np.ndarray): input data, shape (N, D)
        Y (np.ndarray): true labels, shape (N,) or (N,1)

    Returns:
        np.ndarray: gradient vector of shape (M, 1)
    """
    N = len(X)
    M = len(ws)
    grad_ws = np.zeros((M, 1))

    for i in range(N):
        xi = X[i]
        yi = Y[i]
        pred = model(ws, xi)
        error = yi - pred

        for j in range(M):
            grad_ws[j] += error * gradients[j](ws, xi)

    grad_ws = -2 / N * grad_ws
    return grad_ws

def grad_desc_mse(K, ws, learning_eps, loss_fn, grad_loss_fn, verbose=False):
    """
    Gradient descent for minimizing MSE loss.

    Args:
        K (int): Number of iterations.
        ws (np.ndarray): Initial weight vector.
        learning_eps (float): Learning rate.
        loss_fn (callable): loss_fn(ws) → scalar loss.
        grad_loss_fn (callable): grad_loss_fn(ws) → gradient array same shape as ws.
        verbose (bool): If True, plots parameter updates in 2D.

    Returns:
        ws (np.ndarray): Final weights.
        history (list of float): Loss value at each iteration (length K+1).
    """
   
    history = [loss_fn(ws)]
    for k in range(K):
        grad_ws = grad_loss_fn(ws)
        old_ws = ws.copy()
        ws = old_ws - learning_eps * grad_ws

        if verbose:
            # Draw a line from old_ws to new ws (assumes ws has at least 2 elements)
            plt.plot(
                [old_ws[0], ws[0]],
                [old_ws[1], ws[1]],
                '-k'
            )

        history.append(loss_fn(ws))
    return ws, history


def plot3d(f, A, B, show_colorbar=True):
    """
    Plots a 3D surface of function f(x1, x2) over meshgrid A, B with MATLAB-like orientation.
    
    Parameters:
    - f: function of (x1, x2)
    - A, B: meshgrid arrays
    - show_colorbar: whether to display the color bar
    """
    Z = f(A, B)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='k')

    # Set axes labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')

    # Set viewing angle: azim=-135, elev=30 gives MATLAB-style view with (0,0) front left
    ax.view_init(elev=30, azim=-135)

    # Add tighter colorbar
    if show_colorbar:
        # Create colorbar next to the plot (adjust size and position)
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(Z)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)

    # Clean layout
    plt.title('3D Surface Plot')
    plt.tight_layout()
    plt.show()

def plot3d_(f, A, B, show_colorbar=True):
    """
    Plots a 3D surface of function f(x1, x2) over meshgrid A, B with MATLAB-like orientation.
    
    Parameters:
    - f: function of (x1, x2)
    - A, B: meshgrid arrays
    - show_colorbar: whether to display the color bar
    """
    # Create a vectorized version of the function
    if not hasattr(f, 'vectorized'):
        f_vec = np.vectorize(f)
    else:
        f_vec = f
    
    # Apply function to each element in the meshgrid
    Z = f_vec(A, B)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='k')

    # Set axes labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')

    # Set viewing angle: azim=-135, elev=30 gives MATLAB-style view with (0,0) front left
    ax.view_init(elev=30, azim=-135)

    # Add tighter colorbar
    if show_colorbar:
        # Create colorbar next to the plot (adjust size and position)
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(Z)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)

    # Clean layout
    plt.title('3D Surface Plot')
    plt.tight_layout()
    
    return fig


def init_ws(input_size=2, hidden_size=32, output_size=1, seed=None):
    """
    Xavier initialization for a 2-layer network:
    - Hidden layer: ReLU activation
    - Output layer: Sigmoid activation

    Returns:
        ws: 1D NumPy array with all initial weights and biases concatenated
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Hidden layer ---
    # Each hidden neuron has 2 weights + 1 bias
    limit_hidden = np.sqrt(6 / (input_size + hidden_size))
    hidden_weights = []
    for _ in range(hidden_size):
        w1 = np.random.uniform(-limit_hidden, limit_hidden)
        w2 = np.random.uniform(-limit_hidden, limit_hidden)
        b  = 0.0
        hidden_weights.extend([w1, w2, b])  # 3 values per neuron

    # --- Output layer ---
    # One weight per hidden neuron + 1 bias
    limit_output = np.sqrt(6 / (hidden_size + output_size))
    output_weights = list(np.random.uniform(-limit_output, limit_output, hidden_size))
    output_bias = 0.0
    output_weights.append(output_bias)

    # Combine all weights into a single vector
    ws = np.array(hidden_weights + output_weights).reshape(-1, 1)
    return ws

def accuracy(X, Y, f):
    """
    Compute accuracy of function f on XOR inputs X and labels Y.
    Prints mismatches during evaluation.

    Args:
        X (np.ndarray): shape (N, 2), input data
        Y (np.ndarray): shape (N,), true labels (0 or 1)
        f (callable): function f(x1, x2) → 0 or 1

    Returns:
        float: accuracy as a fraction between 0 and 1
    """
    correct = 0
    for i in range(len(X)):
        x1, x2 = X[i]
        y_pred = f(x1, x2)
        if y_pred == Y[i]:
            correct += 1
        else:
            print(f"f({x1},{x2}) = {y_pred}, but should be {Y[i]}")
    return correct / len(X)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)


def init_single_neuron_ws(input_size=2, seed=None):
    """
    Xavier initialization for a single neuron:
    - input_size: number of input features (e.g., 2 for XOR)
    - returns: weights and bias in shape (input_size + 1, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    output_size = 1  # single neuron output
    limit = np.sqrt(6 / (input_size + output_size))

    weights = [np.random.rand() * 2 * limit - limit for _ in range(input_size)]
    bias = 0.0
    ws0 = np.array(weights + [bias]).reshape(-1, 1)  # shape: (input_size + 1, 1)

    return ws0

def plot_history(history, title="Training History"):
    """
    Plot the training history of a model.

    Args:
        history (list): List of loss values at each iteration.
        title (str): Title of the plot.
    """
    plt.plot(history)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()


def hidden_layer(ws, x, n):
    """
    Compute the outputs of the hidden layer (pre-activation) for a given input x.
    
    Args:
        ws: Flat weight vector (length = 3 * n)
        x: Input array (e.g., [x1, x2])
        n: Number of hidden neurons
        
    Returns:
        y: Output vector of shape (n,)
    """
    y = np.zeros(n)
    for i in range(n):
        i1 = 3 * i      # index for w1
        i2 = 3 * i + 1  # index for w2
        i3 = 3 * i + 2  # index for bias
        y[i] = ws[i1] * x[0] + ws[i2] * x[1] + ws[i3]
    return y

