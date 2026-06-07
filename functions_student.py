import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def get_affine_transformation(points_in, points_out):
    """
    TODO for students:
    Estimate the affine transformation matrix mapped from points_in to points_out.
    Transform the input points to homogenous coordinates and solve the least-squares problem.
    """
    # transform to homogenous coordinates
    points_in_hom = np.hstack([points_in, np.ones((len(points_in), 1))])

    # solve the least-squares problem A.T@Ax = A.Tb
    lstsq_problem = np.linalg.lstsq(points_in_hom, points_out, rcond=None)

    return lstsq_problem[0]

def transform_points(points, matrix):
    """
    TODO for students:
    Given a set of 2D points, apply the transformation matrix.
    Return the new (x, y) coordinates.
    """
    # homogneous coordinates
    points_hom = np.hstack([points, np.ones((len(points), 1))])
    # transform the points
    return points_hom @ matrix

def _affine_to_homogeneous(matrix):
    """Convert the 3x2 row-vector affine matrix to a 3x3 homogeneous matrix."""
    matrix = np.asarray(matrix)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape != (3, 2):
        raise ValueError("Affine matrix must have shape (3, 2) or (3, 3)")

    homogeneous = np.eye(3, dtype=matrix.dtype)
    homogeneous[:, :2] = matrix
    return homogeneous

def _invert_affine(matrix):
    """Convert the 3x2 row-vector affine matrix to a 3x3 homogeneous matrix and invert it."""
    homogeneous = _affine_to_homogeneous(matrix)
    return np.linalg.inv(homogeneous)[:, :2]

def backwards_mapping(image, output_shape, transformation, background=0):
    """
    TODO for students:
    Apply a backward mapping transformation to the input image.
    """
    # create the points of the new image
    height, width = np.arange(output_shape[0]), np.arange(output_shape[1])
    y_axis, x_axis = np.meshgrid(height, width, indexing='ij')

    # transform the points into the original image
    stack_grid = np.column_stack((y_axis.flatten(), x_axis.flatten()))  # (row, col)
    points_transformed = transform_points(stack_grid, transformation)

    # remove all points, that land outside the original image
    col = points_transformed[:, 1]
    row = points_transformed[:, 0]
    mask = (col >= 0) & (col < image.shape[1]) & (row >= 0) & (row < image.shape[0])
    col_new = col[mask]
    row_new = row[mask]
    output_points = stack_grid[mask]

    # write the pixel values at the correct location
    output_image = np.full(output_shape, background, dtype=image.dtype)
    output_image[output_points[:, 0].astype(int), output_points[:, 1].astype(int)] = image[row_new.astype(int), col_new.astype(int)]

    return output_image

def downsample_bilinear(img, factor):
    """
    TODO for students:
    Downsample a grayscale image by the given factor using bilinear interpolation.
    """
    height, width = img.shape

    #find new width and height by dividing by the provided downsampling factor
    new_height = int(height / factor)
    new_width = int(width / factor)

    #initialize an array for our downsampled output
    downsampled = np.zeros((new_height, new_width))


    #what does the loop do: we iterate through every single coordinate of our blank image that (downsampled/smaller output) row by row, col by col and:
    for row_new in range(new_height):
        for col_new in range(new_width):

            #find where the center of our new pixel would be placed if we overlaid it to the original image
            row_orig = (row_new + 0.5) * factor - 0.5
            col_orig = (col_new + 0.5) * factor - 0.5

            #we find the neighbouring pixels by rounding down (above/ to the left) and adding one (below, to the right)
            row_low = int(np.floor(row_orig))
            row_high = row_low + 1
            col_low = int(np.floor(col_orig))
            col_high = col_low + 1

            #we find distances
            delta_r = row_orig - row_low #from the row above to our target
            delta_c = col_orig - col_low #from col to the left to our target

            #apply clipping to stay within the borders - row_low, row_high etc. act as borders,
            #if we have a negative value, we zero it out instead, too large values are clipped to the border values
            row_low = max(0, min(row_low, height - 1))
            row_high = max(0, min(row_high, height - 1))
            col_low = max(0, min(col_low, width - 1))
            col_high = max(0, min(col_high, width - 1))

            #retrieve color intensity values from the surrounding pixels (from the original image)
            #we convert them to float to prevent warnings during subtraction math
            p1 = float(img[row_low, col_low])  # top-left
            p2 = float(img[row_low, col_high])  # top-right
            p3 = float(img[row_high, col_low])  # bottom-left
            p4 = float(img[row_high, col_high])  # bottom-right

            #bilinear interpolation
            top_interp = p1 + delta_c * (p2 - p1) #horizontal interpolation, blend between two top pixels.
            bottom_interp = p3 + delta_c * (p4 - p3) #horizontal interpolation, blend between two bottom pixels.
            final_interp = top_interp + delta_r * (bottom_interp - top_interp) #vertical interpolation, blends two above interpolations together

            #assign to our output array prepared before
            downsampled[row_new, col_new] = final_interp

    return downsampled


def histogram_equalization(img):
    """
    TODO for students:
    Compute histogram equalization mapping from an image.
    Return the equalized image.
    """
    #histogram equalization - we use the image's histogram for contrast adjustment

    #first we find the histogram of an image
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256]) #.flatten turns the 2D image into 1D array, while bins=256 covers our range of 0-255

    #cumulative distribution function (CDF) calculation
    cdf = hist.cumsum() # .cumsum is a cumulative sum of each row

    #normalize the CDF to map to the 0-255 range, just in case we have many 0 values, this way we avoid math calculation problems
    cdf_norm = np.ma.masked_equal(cdf, 0)

    #linear scaling transformation : y = ( ((x - cdf_min(x)) * (target_max - target_min))/ cdf_max(x) - cdf_min(x) ) + target_min
    cdf_linscaled = ((cdf_norm - cdf_norm.min()) * 255 ) / (cdf_norm.max() - cdf_norm.min()) # target_min = 0, target_max = 255

    #fill masked values as 0, conver the look up table to uint8
    cdf_filled = np.ma.filled(cdf_linscaled, 0).astype('uint8')

    #now we have a look up table to map the original image pixels
    return cdf_filled[img]



def convolve2d(img, kernel):
    """
    TODO: Apply a 2D convolution (without padding, assumes odd kernel).
    
    Parameters:
        img (np.ndarray): Grayscale image.
        kernel (np.ndarray): 2D filter kernel.
    
    Returns:
        np.ndarray: Convolved image (same size as input, zero-padded).
    """
    height, width = img.shape
    #prepare our empty output
    output_img = np.zeros_like(img, dtype=np.float32)

    #find kernel center
    kernel_height, kernel_width = kernel.shape

    center_width = kernel_width // 2
    center_height = kernel_height // 2

    #apply padding
    img_padded = np.pad(img, ((center_height, center_height), (center_width, center_width)), mode='edge')
    img_padded = img_padded.astype(np.float32)

    for ky in range(kernel_height):
        for kx in range(kernel_width):
            output_img += (kernel[ky, kx] *
                           img_padded[ky: ky + height, kx: kx + width])

    return output_img


def _median_filter(img, kernel_size):
    """
    TODO: Apply a median filter.

    Parameters:
        img (np.ndarray): Grayscale image.

    Returns:
        filtered image
    """
    height, width = img.shape
    output_img = np.zeros_like(img, dtype=np.float32)
    center = kernel_size // 2

    img_padded = np.pad(img, ((center, center), (center, center)), mode='edge')
    img_padded = img_padded.astype(np.float32)

    for x in range(height):
        for y in range(width):
            window = img_padded[x: x + kernel_size, y: y + kernel_size]
            output_img[x, y] = np.median(window)

    return output_img

def _gaussian_filter(img, kernel_size, sigma):

    #find the center
    center = kernel_size // 2

    #get our coordinate range for the kernel
    coords = np.arange(-center, center + 1)
    #use the coords to make a full kernel
    cols, rows = np.meshgrid(coords, coords)

    #compute squared distance
    sqrd_dist = cols ** 2 + rows ** 2

    #apply Gaussian formula
    kernel = np.exp(-sqrd_dist / (2 * sigma ** 2))

    #normalize kernel
    kernel_norm = kernel / np.sum(kernel)

    return convolve2d(img, kernel_norm)

def sobel_filter(img):
    """
    TODO: Apply Sobel edge detection filter (magnitude of gradients).
    
    Parameters:
        img (np.ndarray): Grayscale image.
    
    Returns:
        np.ndarray: Sobel gradient magnitude image.
    """
    #define our sobel kernel matrices
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    #apply convolution using the convolve2d function written above
    out_x = convolve2d(img, Gx)
    out_y = convolve2d(img, Gy)

    #compute gradient magnitude
    output = np.sqrt(out_x ** 2 + out_y ** 2)

    #normalize
    if output.max() > 0:
        output = output / output.max() * 255

    return output

def filter_wrapper_fn(img, mode, kernel_size=3, sigma=1.0):
    """
    TODO Apply one of the following filters to the image: mean, median, gaussian, sobel.

    Parameters:
        img (np.ndarray): Grayscale image.
        mode (str): One of "mean", "median", "gaussian", "sobel"
        kernel_size (int): Kernel size (must be odd)
        sigma (float): Gaussian std dev (only used for gaussian)
    
    Returns:
        np.ndarray: Filtered image.
    """

    if mode == "mean":
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return convolve2d(img, kernel)

    elif mode == "median":
        return _median_filter(img, kernel_size)

    elif mode == "gaussian":
        return _gaussian_filter(img, kernel_size, sigma)

    elif mode == "sobel":
        return sobel_filter(img)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: mean, median, gaussian, sobel")


def get_keypoints(image, filtering=True, sigma=3):
    """
    Extracts keypoints from the image. 
    You might want to optionally smooth the image with a gaussian filter first.
    """
    if filtering:
        image = gaussian_filter(image, sigma=sigma)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image.astype(np.uint8), None)

    return keypoints, descriptors

def intersect2d(array1, array2):
    """ Helper to get intersection of row matches """
    test = array1[:, None] == array2
    return array2[np.all(test.mean(0) > 0, axis=1)]

def matching(descriptors_1, descriptors_2, max_ratio=0.7, cross_checking=True):
    """
    TODO:
    Matches the descriptors against each other.
    Returns the best match for each descriptor, if it is significant.
    The significance is defined by the max_ratio: distance_1 / distance_2 < max_ratio.
    Optional cross-checking of matches.
    """
    #descriptor = list of 128 numbers that describes what a given patch of an image looks like.
    #we want to find which keypoints in image 1 correspond to the same real-world points in image 2

    #find pairwise L2 distances (straight-line distance between descriptors)
    dists = np.linalg.norm(descriptors_1[:, None] - descriptors_2[None, :], axis=2) # (N, M)

    #forward matching: img1 -> img2
    sorted_idx1 = np.argsort(dists, axis=1) #sort to get the closest keypoints first
    matches1 = sorted_idx1[:, :2]  # shape (N, 2) select best and second best
    dist1 = dists[np.arange(len(descriptors_1)), matches1[:, 0]] #closest distance
    dist2 = dists[np.arange(len(descriptors_1)), matches1[:, 1]] #second closest distance
    mask1 = dist1 / dist2 < max_ratio # keep match only if best is significantly closer than second best
    final_matches = np.stack((np.arange(descriptors_1.shape[0])[mask1], matches1[mask1, 0])).T # save the "surviving" matches as pairs

    #backward matching: img2 -> img1 (for the cross-check) #do the same as previously but on the other axis,
    #this way for each img2 descriptor, we find the closest img1 descriptor
    sorted_idx2 = np.argsort(dists, axis=0)
    matches2 = sorted_idx2[:2, :].T  # shape (M, 2) — best and second best per img2 descriptor
    dist1_r = dists[matches2[:, 0], np.arange(len(descriptors_2))]
    dist2_r = dists[matches2[:, 1], np.arange(len(descriptors_2))]
    mask2 = dist1_r / dist2_r < max_ratio

    # cross_checking
    if cross_checking:
        final_matches2 = np.stack(
            (np.arange(descriptors_2.shape[0])[mask2], matches2[mask2, 0]),
        ).T
    if cross_checking:
        # return the intersection of the two matches arrays (invert final_matches2 to point in the same direction)
        final_matches = intersect2d(final_matches, final_matches2[:, ::-1])
    return final_matches

def ransac(points_in, points_out, matches, percentage_outliers=0.5, probability=0.99, cutoff=20, k=3):
    """
    TODO:
    Implement Random Sample Consensus to predict a robust affine model on a set with outliers.
    """
    #RANSAC is an algorithm used to fine the best match between two sets of points, even when there is a lot of noise or outliers.

    #instead of taking infinitely many guesses, we use a formula for number of iterations:
    # "If 50% of my data is bad (percentage_outliers=0.5), and I want to be 99% sure (probability=0.99) that I pick a purely good sample at least once,
    # how many attempts do I need?"
    num_iterations = int(np.log(1 - probability) / np.log(1 - (1 - percentage_outliers) ** k))

    support_best = 0
    inlier_best = None

    for _ in range(num_iterations):
        #randomly choose k pieces
        sample_idx = np.random.choice(matches.shape[0], k, replace=False)
        sample_matches = matches[sample_idx]

        #make a hypothetical model based on the randomly selected k(=3) points
        pts_in = points_in[sample_matches[:, 0]]
        pts_out = points_out[sample_matches[:, 1]]

        model = get_affine_transformation(pts_in, pts_out)

        #apply the model to all points, calculate the errors and use those errors as "votes". If there is too many errors, the model is not accepted.
        transformed = transform_points(points_in[matches[:, 0]], model)

        errors = np.linalg.norm(transformed - points_out[matches[:, 1]], axis=1)

        inliers = errors < cutoff
        support = np.sum(inliers)

        #we check if the model is better than the current best model, and if so - we replace it
        if support > support_best:
            support_best = support
            inlier_best = inliers

    # return the inliers
    return matches[inlier_best], support_best

def helper_plot_fn(img1, img2, transformed_img2):
    """ Plotting helper """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title("Source Image")
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title("Destination Image")
    axes[2].imshow(transformed_img2, cmap='gray')
    axes[2].set_title("Transformed Image")
    plt.tight_layout()
    plt.show()
