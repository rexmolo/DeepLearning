import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    grad_ws = np.zeros(M)

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
        # Store parameters at each step
        if verbose:

            # Draw a line from old_ws to new ws (assumes ws has at least 2 elements)
            plt.plot(
                [old_ws[0], ws[0]],
                [old_ws[1], ws[1]],
                '-k'
            )

        print(f"Iteration {k}, MSE: {loss_fn(ws)}, Weights: {ws}")

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

def plot2d_contour(f, A, B, show_colorbar=True, mark_points=None, levels=15, 
                   show_path=False, a_history=None, b_history=None, 
                   learning_rate=None, flip_path=False, fig=None, ax=None):
    """
    Plots a 2D contour plot of f(x1, x2) over meshgrid A, B.

    Parameters:
      - f: function of (x1, x2) - assumed f(a, b) in this context
      - A, B: meshgrid arrays (representing 'a' and 'b' values)
      - show_colorbar: whether to display the color bar
      - mark_points: list of tuples [(a1,b1), (a2,b2),...] points to mark on the plot
      - levels: Number of contour levels
      - show_path: Whether to show gradient descent path
      - a_history, b_history: Lists of a and b values from gradient descent
      - learning_rate: The learning rate used for gradient descent (for title)
      - flip_path: If True, flip the path vertically for better readability
      - fig: Existing figure (optional, to support overlay with plot3d)
      - ax: Existing axes (optional, to support overlay with plot3d)
    
    Returns:
      - fig, ax: The figure and axes objects
    """
    # Vectorize f if needed
    if not hasattr(f, 'vectorized'):
        f_vec = np.vectorize(f)
    else:
        f_vec = f

    # Evaluate Z
    Z = f_vec(A, B)

    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Draw contour lines
    cmap = plt.get_cmap('viridis') 
    cset = ax.contour(
        A, B, Z,
        levels=levels,
        cmap=cmap,
        linewidths=1,
    )

    # Mark specified points if provided
    if mark_points is not None:
        if isinstance(mark_points, (list, tuple)):
            # If a single point is provided as a tuple
            if not isinstance(mark_points[0], (list, tuple)):
                mark_points = [mark_points]
            
            # Plot each point with different markers
            markers = ['+', '*', 'x', 'o', 'd', 's']
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
            
            for i, point in enumerate(mark_points):
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                ax.plot(point[0], point[1], marker=marker, markersize=10, 
                        color=color, linestyle='None', 
                        label=f'Point {i+1} ({point[0]:.2f}, {point[1]:.2f})')
        else:
            print("Warning: mark_points should be a tuple/list of points (a, b).")

    # Plot the gradient descent path if requested
    if show_path and a_history is not None and b_history is not None:
        plot_path_on_contour(ax, a_history, b_history, flip_path)

    if show_colorbar:
        # Add a colorbar for the contour levels
        cbar = fig.colorbar(cset, ax=ax)
        cbar.set_label('Loss value')

    # Set labels and title
    ax.set_xlabel('Parameter a')
    ax.set_ylabel('Parameter b')
    
    # Set title based on whether we're showing a path
    if show_path and learning_rate is not None:
        ax.set_title(f'Gradient Descent Path (lr={learning_rate})')
    else:
        ax.set_title('2D Contour Plot of Loss Function')
        
    ax.grid(True)
    plt.tight_layout()

    # Return the figure and axes objects
    return fig, ax

def plot_path_on_contour(ax, a_history, b_history, flip_path=False):
    """Helper function to plot gradient descent path on a contour plot"""
    if flip_path:
        # Calculate the vertical center of the path
        b_min, b_max = min(b_history), max(b_history)
        b_center = (b_min + b_max) / 2
        
        # Create a flipped version of the path
        b_history_flipped = [2 * b_center - b for b in b_history]
        
        # Plot the flipped path
        ax.plot(a_history, b_history_flipped, 'b-', linewidth=1.5, 
                label='Gradient Descent Path')
        
        # Mark start and end points
        if len(a_history) > 0:
            ax.plot(a_history[0], b_history_flipped[0], 'g*', markersize=12, 
                   markeredgecolor='black', markeredgewidth=1.5, label='Start')
            ax.plot(a_history[-1], b_history_flipped[-1], 'rx', markersize=10, 
                   markeredgewidth=2, label='End')
    else:
        # Plot the original path
        ax.plot(a_history, b_history, 'b-', linewidth=1.5, 
                label='Gradient Descent Path')
        
        # Mark start and end points
        if len(a_history) > 0:
            ax.plot(a_history[0], b_history[0], 'g*', markersize=12, 
                   markeredgecolor='black', markeredgewidth=1.5, label='Start')
            ax.plot(a_history[-1], b_history[-1], 'rx', markersize=10, 
                   markeredgewidth=2, label='End')
    
    # Add legend if we're showing the path
    ax.legend(loc='best')