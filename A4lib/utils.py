

def rss2(a, b, X, Y):
    # exactly sum of squared residuals
    return np.sum((Y - (a*X + b))**2)

def grad_rss2(a, b, X, Y):
    """
    Calculate the gradient of RSS with respect to a and b.
    
    Parameters:
    - a: slope parameter
    - b: intercept parameter
    - X: input features (1D array)
    - Y: target values (1D array)
    
    Returns:
    - numpy array [grad_a, grad_b] with gradients for both parameters
    """
    n = len(X)
    grad_a = -2 * np.sum((Y - (a*X + b)) * X)
    grad_b = -2 * np.sum(Y - (a*X + b))
    return np.array([grad_a, grad_b])

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

def grad_desc_rss2(K, a0, b0, learning_eps, f_cost, f_grad, verbose=False):
    """
    Perform gradient descent for minimizing the Residual Sum of Squares (RSS).
    
    Parameters:
    - K: Number of iterations
    - a0: Initial value for parameter a
    - b0: Initial value for parameter b
    - learning_eps: Learning rate (step size)
    - f_cost: Cost function that calculates RSS given a and b
    - f_grad: Function that calculates the gradient of RSS with respect to a and b
    - verbose: If True, print progress and create visualization
    
    Returns:
    - a_history: Array of a values over iterations
    - b_history: Array of b values over iterations
    """
    
    # Initialize arrays to store parameters
    a_history = np.zeros(K + 1)
    b_history = np.zeros(K + 1)
    
    # Set initial values
    a_history[0] = a0
    b_history[0] = b0
    
    # Gradient descent loop
    for k in range(K):
        # Calculate gradient
        grad = f_grad(a_history[k], b_history[k])
        grad_a = grad[0]
        grad_b = grad[1]
        
        # Update parameters
        a_history[k + 1] = a_history[k] - learning_eps * grad_a
        b_history[k + 1] = b_history[k] - learning_eps * grad_b
        
        # Print progress if verbose
        if verbose and (k % 100 == 0 or k == K-1):
            cost = f_cost(a_history[k+1], b_history[k+1])
            print(f"Iteration {k+1}/{K}: a = {a_history[k+1]:.6f}, b = {b_history[k+1]:.6f}, cost = {cost:.6f}")

        
    return a_history, b_history

def plot2d_contour(f, A, B, show_colorbar=True, mark_point=None, levels=15, show_path=False, a_history=None, b_history=None, learning_rate=None, flip_path=False):
    """
    Plots a 2D contour plot of f(x1, x2) over meshgrid A, B.

    Parameters:
      - f: function of (x1, x2) - assumed f(a, b) in this context
      - A, B: meshgrid arrays (representing 'a' and 'b' values)
      - show_colorbar: whether to display the color bar
      - mark_point: tuple (a, b), optional point to mark on the plot with a '+'.
      - levels: Number of contour levels.
      - show_path: Whether to show gradient descent path
      - a_history, b_history: Lists of a and b values from gradient descent
      - learning_rate: The learning rate used for gradient descent (for title)
      - flip_path: If True, flip the path vertically for better readability
    """
    # Vectorize f if needed
    if not hasattr(f, 'vectorized'):
        f_vec = np.vectorize(f)
    else:
        f_vec = f

    # Evaluate Z
    Z = f_vec(A, B)

    fig, ax = plt.subplots(figsize=(8, 6)) # Create a standard 2D plot with specified size

    # --- draw contour lines ---
    cmap = plt.get_cmap('viridis') # Or 'jet' if you prefer
    cset = ax.contour(
        A, B, Z,
        levels=levels,        # Number of contour levels
        cmap=cmap,
        linewidths=1,
    )

    # Mark the specified point if provided
    if mark_point is not None:
        # Ensure mark_point is a tuple/list of length 2
        if isinstance(mark_point, (list, tuple)) and len(mark_point) == 2:
            # Plot the point (a0, b0)
            ax.plot(mark_point[0], mark_point[1], marker='+', markersize=10, color='blue', linestyle='None', label=f'Point ({mark_point[0]}, {mark_point[1]})')
        else:
            # Print a warning if mark_point is not in the expected format
            print("Warning: mark_point should be a tuple or list of length 2 (a, b).")

    # Plot the gradient descent path if requested
    if show_path and a_history is not None and b_history is not None:
        if flip_path:
            # Calculate the vertical center of the path
            b_min = min(b_history)
            b_max = max(b_history)
            b_center = (b_min + b_max) / 2
            
            # Create a flipped version of the path
            b_history_flipped = [2 * b_center - b for b in b_history]
            
            # Plot the flipped path
            ax.plot(a_history, b_history_flipped, 'b-', linewidth=1, label='Gradient Descent Path')
            
            # Optionally mark start and end points - with enhanced visibility
            if len(a_history) > 0:
                # Draw the start point with larger size and different color to make it more visible
                ax.plot(a_history[0], b_history_flipped[0], 'g*', markersize=12, markeredgecolor='black', 
                       markeredgewidth=1.5, label='Start')
                # Draw the end point as before
                ax.plot(a_history[-1], b_history_flipped[-1], 'rx', markersize=10, markeredgewidth=2, label='End')
        else:
            # Plot the original path
            ax.plot(a_history, b_history, 'b-', linewidth=1, label='Gradient Descent Path')
            
            # Optionally mark start and end points - with enhanced visibility
            if len(a_history) > 0:
                # Draw the start point with larger size and different color to make it more visible
                ax.plot(a_history[0], b_history[0], 'g*', markersize=12, markeredgecolor='black', 
                       markeredgewidth=1.5, label='Start')
                # Draw the end point as before
                ax.plot(a_history[-1], b_history[-1], 'rx', markersize=10, markeredgewidth=2, label='End')
        
        # Add legend if we're showing the path
        ax.legend(loc='best')

    if show_colorbar:
        # Add a colorbar for the contour levels
        fig.colorbar(cset, ax=ax).set_label('RSS')

    # Set labels and title
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    
    # Set title based on whether we're showing a path
    if show_path and learning_rate is not None:
        ax.set_title(f'Gradient Descent Path (lr={learning_rate})')
    else:
        ax.set_title('2D Contour Plot')
        
    ax.grid(True)
    plt.tight_layout() # Adjust layout

    # Return the figure and axes objects
    return fig, ax