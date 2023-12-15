from typing import List, Callable, Union, Literal

ListImage = List[List[int]]

ListImageRaw = List[List[float]]

FilterKernel = List[List[float]]

FloatOrNone = Union[float, None]

ImageFunction = Callable[[int, int], int]

Laplacian2EdgeDetectorType = Literal['downward', 'upward']

LaplaceEdgeDetectorType = Literal['horizontal', 'main_diagonal', 'vertical', 'antidiagonal']
