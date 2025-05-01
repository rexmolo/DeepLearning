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
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('f(x1, x2)')

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
