import numpy as np

# Non-negativity projection to ensure the tensor remains valid
def non_negativity_projection(tensor, beta):
    """
    Project the tensor onto the non-negative orthant, ensuring all values are non-negative.
    
    Parameters:
        tensor: The tensor to be projected
        epsilon: Small value to replace zeros with (default: 1e-10)
    
    Returns:
        projected_tensor: The tensor after applying the non-negativity projection
    """
    return np.maximum(tensor, -1/(beta-1))

def update_lagrange_multipliers(mu, target_marginal, theta_star, beta, axis):
    """
    Update the Lagrange multipliers for a given marginal (generalized for multiple dimensions).
    
    Parameters:
        mu: Current Lagrange multipliers (a vector for the current marginal)
        target_marginal: The target marginal distribution (from input)
        theta_star: The transport plan after projection (θ^*)
        beta: The divergence parameter
        axis: The axis corresponding to the marginal being projected
        
    Returns:
        mu: Updated Lagrange multipliers for the current marginal (vector)
    """
    
    # Identify which axes we need to sum over (all but the current axis)
    sum_axes = tuple(i for i in range(theta_star.ndim) if i != axis)
    
    # Compute the numerator: sum over the other dimensions
    # Create a copy of theta_star to avoid modifying the original tensor
    theta_star_updated = np.copy(theta_star)

    # Loop over all values in the tensor along the given axis
    # For each value in the axis (we get a slice along the current axis)
    for idx in range(theta_star.shape[axis]):
        # Create a slice of theta_star for the current axis index
        slice_idx = [slice(None)] * theta_star.ndim  # Create a list of slices
        slice_idx[axis] = idx  # Set the slice for the current axis
        
        # Subtract the corresponding Lagrange multiplier (mu[idx]) from the slice
        theta_star_updated[tuple(slice_idx)] -= mu[idx]
    
    numerator = np.sum(
        np.power((beta - 1) * theta_star_updated + 1, 1 / (beta - 1)), axis=sum_axes
    ) - target_marginal
    
    # Compute the denominator: sum over the other dimensions
    denominator = np.sum(
        np.power((beta - 1) * theta_star_updated + 1, 1 / (beta - 1) - 1), axis=sum_axes
    )
    
    # Update the Lagrange multipliers
    mu_new = mu + numerator / denominator
    return mu_new

def iterate_lagrange_updates(mu, target_marginal, theta_star, beta, num_iters, axis):
    """
    Iterate Lagrange multiplier updates for a given axis.
    
    Parameters:
        mu: Current Lagrange multipliers (a vector for the current marginal)
        target_marginals: List of target marginals for each axis
        theta_star: The transport plan after projection (θ^*)
        beta: The divergence parameter
        num_iters: Number of iterations for the Lagrange multiplier update
        axis: The axis corresponding to the marginal being projected
        
    Returns:
        mu_final: Final Lagrange multipliers after N iterations
    """
    # Iterate N times
    for _ in range(num_iters):
        mu = update_lagrange_multipliers(mu, target_marginal, theta_star, beta, axis)
    
    # Return the final Lagrange multiplier after N iterations
    return mu

def marginal_projection(mu, theta_tilde, axis):
    """
    Update dual variables for a given marginal (generalized for multiple dimensions).
    
    Parameters:
        mu: Current Lagrange multipliers (a vector for the current marginal)
        theta_tilde: The transport plan after projection (θ^*)
        axis: The axis corresponding to the marginal being projected
        
    Returns:
        theta_tilde_updated: Updated dual variables
    """
    # Compute the numerator: sum over the other dimensions
    # Create a copy of theta_star to avoid modifying the original tensor
    theta_tilde_updated = np.copy(theta_tilde)

    # Loop over all values in the tensor along the given axis
    # For each value in the axis (we get a slice along the current axis)
    for idx in range(theta_tilde.shape[axis]):
        # Create a slice of theta_star for the current axis index
        slice_idx = [slice(None)] * theta_tilde.ndim  # Create a list of slices
        slice_idx[axis] = idx  # Set the slice for the current axis
        
        # Subtract the corresponding Lagrange multiplier (mu[idx]) from the slice
        theta_tilde_updated[tuple(slice_idx)] -= mu[idx]
        
    return theta_tilde_updated

def alternating_beta_projection_log_nd(pi_hat, marginals, beta, num_iters=50, full_cycles=100, replace_zeros=True, epsilon=1e-10):
    """
    Alternating Beta projection in log-domain for N-dimensional tensor.
    
    Parameters:
        pi_hat: Initial reference tensor, shape (d1, d2, ..., dn)
        marginals: List of target marginals, one for each axis (each is a 1D array)
        beta: The Beta divergence parameter
        num_iters: Number of iterations for updating the Lagrange multipliers
        full_cycles: Number of full cycles (iterations over all marginals)
        replace_zeros: If True, replaces zeros in pi_hat with epsilon
        epsilon: Small value to replace zeros with (default: 1e-10)

    Returns:
        pi: Final transport plan, same shape as pi_hat
    """

    # Step 1: Replace zeros with epsilon (if replace_zeros is True)
    if replace_zeros:
        pi_hat = np.where(pi_hat == 0, epsilon, pi_hat)

    # Step 2: Rescale pi_hat to ensure it is a valid probability distribution (sums to 1)
    pi_hat /= np.sum(pi_hat)

    ndim = pi_hat.ndim

    # Initialize the Lagrange multipliers (one for each dimension)
    mu = [np.zeros_like(marginals[i]) for i in range(ndim)]
    
    # Initialize tilde_theta and theta_star
    tilde_theta = 1/(beta-1) * (np.power(pi_hat, beta-1) - 1)
    theta_star = non_negativity_projection(tilde_theta, beta)
    

    # Begin the full cycles
    for cycle in range(full_cycles):
        for axis in range(ndim):
            # Step 1: Update Lagrange multipliers for the current marginal (e.g., axis)
            mu_init = mu[axis]
            mu_limit = iterate_lagrange_updates(mu_init, marginals[axis], theta_star, beta, num_iters, axis)
                        
            # Step 2: Update tilde_theta by projecting onto the marginal constraints
            tilde_theta = marginal_projection(mu_limit, tilde_theta, axis)
            
            # Step 3: Project onto non-negativity (as in the original algorithm)
            theta_star = non_negativity_projection(tilde_theta, beta)

    # Final transport plan (exponentiate the log_theta to get the transport plan)
    pi = np.power((beta-1)*theta_star + 1, 1/(beta-1))
    
    return pi