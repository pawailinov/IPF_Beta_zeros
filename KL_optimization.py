import numpy as np

def alternating_kl_projection(pi_hat, marginals, num_iters=100, replace_zeros=True, epsilon=1e-10):
    """
    Alternating KL projection in dual domain for L-dimensional plan

    Parameters:
        pi_hat: Initial plan, tensor shape (I1, I2, ..., IL)
        marginals: Marginal distributions, list of one-dimensional arrays
        num_iters: Number of alternating projections
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

    # Step 3: Convert to dual domain
    log_theta = np.log(pi_hat)
    ndim = log_theta.ndim

    # Step 4: Perform alternating projections over the marginals
    for k in range(num_iters):
        axis = k % ndim  # Which marginal to project onto this step

        # Sum over all other axes to get marginal vector along current axis
        sum_axes = tuple(i for i in range(ndim) if i != axis)
        sum_given_axis = np.sum(np.exp(log_theta), axis=sum_axes)

        # Reshape target and current marginal for broadcasting
        shape = [1] * ndim
        shape[axis] = -1  # Only allow broadcasting along current axis

        log_theta += (
            np.log(marginals[axis]).reshape(shape) - np.log(sum_given_axis).reshape(shape)
        )

    # Step 5: Convert to primal domain
    pi = np.exp(log_theta)

    return pi
