import numpy as np

def non_negativity_projection(tensor, beta):
    """
    Projection onto the non-negative orthant, ensuring all values are non-negative
    
    Parameters:
        tensor: The current projected plan, tensor shape (I1, I2, ..., IL)
        beta: The divergence parameter, scalar
    
    Returns:
        projected_tensor: Projected plan, same shape as tensor
    """
    return np.maximum(tensor, -1/(beta-1))

def update_lagrange_multipliers(mu, target_marginal, theta_star, beta, axis):
    """
    Update the Lagrange multipliers for a given marginal distribution
    
    Parameters:
        mu: Current Lagrange multipliers, vector of length dl
        target_marginal: The target marginal distribution, vector of length dl
        theta_star: The current projected plan, tensor
        beta: The divergence parameter, scalar
        axis: The axis corresponding to the marginal being projected (l), scalar
        
    Returns:
        mu: Updated Lagrange multipliers for the current marginal, vector of length dl
    """
    
    # Identify which axes is needed to sum over (all but the current axis)
    sum_axes = tuple(i for i in range(theta_star.ndim) if i != axis)
    
    # Create a copy of theta_star to avoid modifying the original tensor
    theta_star_updated = np.copy(theta_star)

    # Loop over all values in the tensor along the given axis
    for idx in range(theta_star.shape[axis]):
        # Create a slice of theta_star for the current axis index
        slice_idx = [slice(None)] * theta_star.ndim  # Create a list of slices
        slice_idx[axis] = idx  # Set the slice for the current axis
        
        # Subtract the corresponding Lagrange multiplier (mu[idx]) from the slice
        theta_star_updated[tuple(slice_idx)] -= mu[idx]
        
    # Newton step
    # Compute the numerator: sum over the other dimensions 
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
    Iterate updates of Lagrange multipliers for a given marginal distribution
    
    Parameters:
        mu: Current Lagrange multipliers, vector of length dl
        target_marginals:The target marginal distribution, vector of length dl
        theta_star: The current projected plan, tensor
        beta: The divergence parameter, scalar
        num_iters: Number of iterations for the Lagrange multipliers update
        axis: The axis corresponding to the marginal being projected (l), scalar
        
    Returns:
        mu_final: Updated Lagrange multipliers for the current marginal after N iterations, vector of length dl 
    """
    # Iterate N times
    for _ in range(num_iters):
        mu = update_lagrange_multipliers(mu, target_marginal, theta_star, beta, axis)
    
    # Return the final Lagrange multiplier after N iterations
    return mu

def marginal_projection(mu, theta_tilde, axis):
    """
    Projection onto the marginal consr
    
    Parameters:
        mu: Convergent Lagrange multipliers values, vector of length dl
        theta_tilde: The current projected plan, tensor
        axis: The axis corresponding to the marginal being projected (l), scalar
        
    Returns:
        theta_tilde_updated: Projected plan, same shape as theta_tilde
    """

    # Create a copy of theta_star to avoid modifying the original tensor
    theta_tilde_updated = np.copy(theta_tilde)

    # Loop over all values in the tensor along the given axis
    for idx in range(theta_tilde.shape[axis]):
        # Create a slice of theta_star for the current axis index
        slice_idx = [slice(None)] * theta_tilde.ndim  # Create a list of slices
        slice_idx[axis] = idx  # Set the slice for the current axis
        
        # Subtract the corresponding Lagrange multiplier (mu[idx]) from the slice
        theta_tilde_updated[tuple(slice_idx)] -= mu[idx]
        
    return theta_tilde_updated

def alternating_beta_projection(pi_hat, marginals, beta, num_iters=50, full_cycles=100, replace_zeros=True, epsilon=1e-10):
    """
    Alternating KL projection in dual domain for L-dimensional plan
    
    Parameters:
        pi_hat: Initial plan, tensor shape (I1, I2, ..., IL)
        marginals: Marginal distributions, list of one-dimensional arrays
        beta: The divergence parameter, scalar
        num_iters: Number of iterations for the Lagrange multipliers update
        full_cycles: Number of full cycles of alternating projections
        replace_zeros: If True, replaces zeros in pi_hat with epsilon
        epsilon: Small value to replace zeros with (default: 1e-10), scalar

    Returns:
        pi: Projection, same shape as pi_hat
    """

    # Step 1: Replace zeros with epsilon (if replace_zeros is True)
    if replace_zeros:
        pi_hat = np.where(pi_hat == 0, epsilon, pi_hat)

    # Step 2: Rescale pi_hat to ensure it is a valid probability distribution (sums to 1)
    pi_hat /= np.sum(pi_hat)
    # Get number of dimensions
    ndim = pi_hat.ndim

    # Step 3: Initialization
    # Initialize the Lagrange multipliers (one for each dimension)
    mu = [np.zeros_like(marginals[i]) for i in range(ndim)]
    # Initialize tilde_theta and theta_star
    tilde_theta = 1/(beta-1) * (np.power(pi_hat, beta-1) - 1)
    theta_star = non_negativity_projection(tilde_theta, beta)
    

    # Step 4: Perform alternating projections 
    for cycle in range(full_cycles):
        for axis in range(ndim):
            #Update Lagrange multipliers for the current marginal (e.g., axis)
            mu_init = mu[axis]
            mu_limit = iterate_lagrange_updates(mu_init, marginals[axis], theta_star, beta, num_iters, axis)
                        
            #Update tilde_theta by projecting onto the marginal constraints
            tilde_theta = marginal_projection(mu_limit, tilde_theta, axis)
            
            #Update theta_star by projecting onto non-negativity constrains
            theta_star = non_negativity_projection(tilde_theta, beta)

    # Step 5: Convert to primal domain
    pi = np.power((beta-1)*theta_star + 1, 1/(beta-1))
    
    return pi