
import numpy as np


def get_1d_gauss_kernel(sigma, extent=3):
    """Build a 1-dimensional Gaussian kernel.

    Parameters
    ----------
    sigma : int or float
        The standard deviation of the Gaussian function.
    extent : int, optional
        How many times sigma to consider on each side of the mean. 3 x sigma
        would cover >99% of the values in a Gaussian. This parameter defines the
        length of the returned kernel, i.e. = (2 * extent) + 1.

    Returns
    -------
    out : 1d array
        The Gaussian kernel.
    """
    n = np.ceil(sigma * extent)
    kernel = np.exp(-(np.arange(-n, n + 1) ** 2) / (2 * (sigma ** 2)))
    kernel = kernel / np.sum(kernel)  # Normalize

    return kernel


def get_1d_LoG_kernel(sigma, extent=3, return_threshold_factor=False):
    """Build a 1-dimensional Laplacian of Gaussian (LoG) kernel.

    Parameters
    ----------
    sigma : int or float
        The standard deviation of the Gaussian function.
    extent : int, optional
        How many times sigma to consider on each side of the mean. 3 x sigma
        would cover >99% of the values in a Gaussian. This parameter defines the
        length of the returned kernel, i.e. = (2 * extent) + 1.
    return_threshold_factor: bool, optional
        Whether or not to return the threshold factor.

    Returns
    -------
    out : 1d array
        The Laplacian of Gaussian kernel.
    """
    kernel = get_1d_gauss_kernel(sigma, extent)
    kernel_len = np.int32(len(kernel))
    kernel_half_len = kernel_len // 2

    # Compute the Laplacian of the above Gaussian.
    # The first (sigma ** 2) below is the normalising factor to render the
    # outputs scale-invariant. The rest is the 2nd derivative of the Gaussian.
    kernel = (sigma ** 2) * \
             (-1 / (sigma ** 4)) * kernel * \
             ((np.arange(-kernel_half_len, kernel_half_len + 1) ** 2) -
              (sigma ** 2))

    # Sum of the points within one-sigma of mean
    threshold_factor = np.sum(kernel) - (
        2 * np.sum(kernel[0:np.ceil(2 * sigma).astype(np.int)]))
    # Note: Doing this before removal of DC (below) because it undesirably
    # lowers the threshold for larger sigma.

    # Normalize, in order to set the convolution outputs to be closer to
    # putative blobs' original SNRs.
    kernel /= threshold_factor

    # Remove DC
    # kernel -= np.mean(kernel)  # Not really necessary

    if return_threshold_factor:
        return kernel, threshold_factor
    else:
        return kernel


def first_true(bool_mask, invalid_val=-1):
    """Get the index of the first True value in the numpy 1D array 'bool_mask'.
    Returns 'invalid_val' if a True value can not be found.
    Based on - https://stackoverflow.com/a/47269413"""
    if len(bool_mask) == 0:
        return invalid_val
    return np.where(bool_mask.any(), bool_mask.argmax(), invalid_val)


def last_true(bool_mask, invalid_val=-1):
    """Get the index of the last True value in the numpy 1D array 'bool_mask'.
    Returns 'invalid_val' if a True value can not be found.
    Based on - https://stackoverflow.com/a/47269413"""
    if len(bool_mask) == 0:
        return invalid_val
    val = bool_mask.shape[0] - np.flip(bool_mask, axis=0).argmax() - 1
    return np.where(bool_mask.any(), val, invalid_val)
