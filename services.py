from math import sqrt
from typing import Tuple
from custom_types import LaplaceEdgeDetectorType, EdgeDetectorDirectionType, ListImage, FilterKernel, ImageFunction, ListImageRaw
from utils import convertToProperImage

def getMirroredImageFunction(image: ListImage, filter_kernel_sizes: Tuple[int, int]) -> ImageFunction:
    N = len(image)
    M = len(image[0])
    
    extension_sizes = tuple(size // 2 for size in filter_kernel_sizes)

    def mirroredImageFunction(i: int, j: int) -> int:
        if i >= N + extension_sizes[0]:
            raise KeyError
        elif i < 0:
            ii = -i
        elif i >= N:
            ii = N - 1 - (i - N + 1)
        else:
            ii = i

        if j >= M + extension_sizes[1]:
            raise KeyError
        elif j < 0:
            jj = -j
        elif j >= M:
            jj = M - 1 - (j - M + 1)
        else:
            jj = j

        return image[ii][jj]

    return mirroredImageFunction

def getMirroredImage(image: ListImage, filter_kernel_sizes: Tuple[int, int]) -> ListImage:
    N = len(image)
    M = len(image[0])
    
    extension_sizes = tuple(size // 2 for size in filter_kernel_sizes)

    mirrored_image = []

    mirroredImageFunction = getMirroredImageFunction(image, filter_kernel_sizes)

    for i in range(-extension_sizes[0], N + extension_sizes[0]):
        mirrored_image.append([])

        for j in range(-extension_sizes[1], M + extension_sizes[1]):
            mirrored_image[-1].append(mirroredImageFunction(i, j))

    return mirrored_image

def linearSpatialFilteringRaw(image: ListImage, filter_kernel: FilterKernel) -> ListImageRaw:
    N = len(image)
    M = len(image[0])

    filter_kernel_sizes = (len(filter_kernel), len(filter_kernel[0]))

    extended_image_function = getMirroredImageFunction(image, filter_kernel_sizes)

    a = filter_kernel_sizes[0] // 2 # equal to (filterKernelSizes[0] - 1) / 2 in formula
    b = filter_kernel_sizes[1] // 2

    filtered_image = [[sum([sum([filter_kernel[a + s][b + t] * extended_image_function(i + s, j + t) for t in range(-b, b + 1)]) for s in range(-a, a + 1)]) for j in range(M)] for i in range(N)]

    return filtered_image

def linearSpatialFiltering(image: ListImage, filter_kernel: FilterKernel) -> ListImage:
    filtered_image = linearSpatialFilteringRaw(image, filter_kernel)

    return convertToProperImage(filtered_image)

# only for 3x3 window
def laplacian2EdgeDetection(image: ListImage, type: EdgeDetectorDirectionType) -> ListImage:
    operator_mask_kernel = [[(1 if type=='downward' else -1) * (1 if (i != 1 or j != 1) else -8.) for j in range(3)] for i in range(3)]

    return linearSpatialFiltering(image, operator_mask_kernel)

# only for 3x3 window
def laplaceEdgeDetection(image: ListImage, type: LaplaceEdgeDetectorType, direction_type: EdgeDetectorDirectionType) -> ListImage:
    def maskKernelMainWeightsCondition(i: int, j: int, type: LaplaceEdgeDetectorType = type):
        if type == 'horizontal':
            return i == 1
        elif type == 'main_diagonal':
            return i == j
        elif type == 'vertical':
            return j == 1
        else:
            return i == 2 - j
        
    multiplier = 1 if direction_type == 'upward' else -1

    operator_mask_kernel = [[multiplier * (2. if maskKernelMainWeightsCondition(i, j) else -1) for j in range(3)] for i in range(3)]

    return linearSpatialFiltering(image, operator_mask_kernel)

# only for 3x3 window
def sobelEdgeDetection(image: ListImage) -> ListImage:
    horizontal_mask_kernel = [[-1, -2, -1], [0, 0, 0.], [1, 2, 1]]

    vertical_mask_kernel = [[-1, 0, 1], [-2, 0 , 2], [-1, 0., 1]]

    horizontal_edges = linearSpatialFilteringRaw(image, horizontal_mask_kernel)

    vertical_edges = linearSpatialFilteringRaw(image, vertical_mask_kernel)

    result_image = [[sqrt(horizontal_edges[i][j] ** 2 + vertical_edges[i][j] ** 2) for j in range(len(image[0]))] for i in range(len(image))]

    return convertToProperImage(result_image)

def unsharpMasking(image: ListImage, k: float = 1, filter_kernel_size: int = 3) -> ListImage:
    center_kernel_element_condition = lambda i, j: i == j and filter_kernel_size == j + i + 1

    filter_kernel = [[((1 + k) * (filter_kernel_size ** 2) - k)/ (filter_kernel_size ** 2) if center_kernel_element_condition(i, j) else -k / (filter_kernel_size ** 2) for j in range(filter_kernel_size)] for i in range(filter_kernel_size)]

    return linearSpatialFiltering(image, filter_kernel)