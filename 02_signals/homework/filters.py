import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for n in range(Hi):
        for m in range(Wi):
            for k in range(-(Hk // 2), Hk // 2 + 1):
                for l in range(-(Wk // 2), Wk // 2 + 1):
                    if n + k > -1 and m + l > -1 and n + k < Hi and m + l < Wi:
                        out[n, m] += (
                            image[n + k, m + l]
                            * kernel[(Hk - 1) // 2 - k, (Wk - 1) // 2 - l]
                        )
    return out


def zero_pad(image, pad_height, pad_width):
    """Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """
    H, W = image.shape

    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width), dtype=image.dtype)

    out[pad_height : pad_height + H, pad_width : pad_width + W] = image

    return out


def conv_fast(image, kernel):
    """An efficient implementation of convolution filter.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    padded_image = zero_pad(image, Hk // 2, Wk // 2)

    kernel = np.flip(kernel)

    for i in range(Hi):
        for j in range(Wi):
            region = padded_image[i : i + Hk, j : j + Wk]
            out[i, j] = np.sum(region * kernel)

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape

    image_padded = np.zeros([max(Hi, Wi), max(Hi, Wi)], dtype=image.dtype)
    image_padded[:Hi, :Wi] = image
    kernel_padded = quarter_roll(kernel, max(Hi, Wi), max(Hi, Wi))

    f_image = np.fft.fft2(image_padded)
    f_kernel = np.fft.fft2(kernel_padded)

    f_out = f_image * f_kernel

    out = np.real(np.fft.ifft2(f_out))[:Hi, :Wi]

    return out


def quarter_roll(img, new_h, num_w):
    """Pads the image and rolls the images quarters so that they get into corners

    Args:
        img: numpy array of shape (Hi, Wi).
        new_h: height of result
        new_w: width if result

    Returns:
        img_rolled: numpy array of shape (new_h, new_w).
    """
    Hi, Wi = img.shape
    roll_height = Hi // 2
    roll_width = Wi // 2

    img_rolled = np.zeros((new_h, num_w), dtype=img.dtype)
    img_rolled[:Hi, :Wi] = img
    img_rolled = np.roll(img_rolled, shift=(-roll_height, -roll_width), axis=(0, 1))
    return img_rolled


def cross_correlation(f, g):
    """Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = conv_faster(f, np.flip(g))

    return out


def zero_mean_cross_correlation(f, g):
    """Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    g = g - np.mean(g)

    out = conv_faster(f, np.flip(g))

    return out


def normalized_cross_correlation(f, g):
    """Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hi, Wi = f.shape
    Hk, Wk = g.shape

    out = np.zeros((Hi, Wi))
    pad_f = zero_pad(f, Hk // 2, Wk // 2)

    g = (g - np.mean(g)) / np.std(g)

    for i in range(Hi):
        for j in range(Wi):
            sub_f = pad_f[i : i + Hk, j : j + Wk]
            sub_f = (sub_f - np.mean(sub_f)) / np.std(sub_f)
            out[i, j] = np.sum(sub_f * g)

    return out
