from traceback import print_exc
from os.path import basename, splitext
from typing import List
from PIL import Image
from constants import COMPUTED_DIRECTORY_NAME
from custom_types import LaplaceEdgeDetectorType, Laplacian2EdgeDetectorType
from utils import convertToListImage, convertToPillowImage, saveImage
from services import laplaceEdgeDetection, laplacian2EdgeDetection, sobelEdgeDetection

def labTask(image_path: str) -> None:
    computed_directory = f'./{COMPUTED_DIRECTORY_NAME}/'

    file_name_with_extension = basename(image_path)
    file_name, file_extension = splitext(file_name_with_extension)

    with Image.open(image_path) as im:
                image = convertToListImage(im)
                
                laplacian_2_edge_detection_types: List[Laplacian2EdgeDetectorType] = ['downward', 'upward']

                laplacian_2_results = [laplacian2EdgeDetection(image, type) for type in laplacian_2_edge_detection_types]

                for i in range(len(laplacian_2_results)):
                    saveImage(convertToPillowImage(laplacian_2_results[i]), f'{computed_directory}{file_name}_laplacian_2_{laplacian_2_edge_detection_types[i]}{file_extension}')

                laplace_edge_detection_types: List[LaplaceEdgeDetectorType] = ['horizontal', 'main_diagonal', 'vertical', 'antidiagonal']

                laplace_results = [laplaceEdgeDetection(image, type) for type in laplace_edge_detection_types]

                for i in range(len(laplace_results)):
                    saveImage(convertToPillowImage(laplace_results[i]), f'{computed_directory}{file_name}_laplace_{laplace_edge_detection_types[i]}{file_extension}')

                sobel_result = sobelEdgeDetection(image)

                saveImage(convertToPillowImage(sobel_result), f'{computed_directory}{file_name}_sobel_{file_extension}')

                

if __name__ == "__main__":
    try:
         labTask('./assets/cameraman.tif') 

         labTask('./assets/lena_gray_256.tif')         
    except Exception as e:
        print('Error occured:')
        print_exc()