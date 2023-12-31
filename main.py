from traceback import print_exc
from os.path import basename, splitext
from typing import List
from PIL import Image
from constants import COMPUTED_DIRECTORY_NAME
from custom_types import LaplaceEdgeDetectorType, EdgeDetectorDirectionType
from utils import convertToListImage, convertToPillowImage, saveImage
from services import laplaceEdgeDetection, laplacian2EdgeDetection, sobelEdgeDetection, unsharpMasking

def labTask(image_path: str) -> None:
    computed_directory = f'./{COMPUTED_DIRECTORY_NAME}/'

    file_name_with_extension = basename(image_path)
    file_name, file_extension = splitext(file_name_with_extension)

    with Image.open(image_path) as im:
                image = convertToListImage(im)
                
                edge_detection_direction_types: List[EdgeDetectorDirectionType] = ['downward', 'upward']

                laplacian_2_results = [laplacian2EdgeDetection(image, type) for type in edge_detection_direction_types]

                for i in range(len(laplacian_2_results)):
                    saveImage(convertToPillowImage(laplacian_2_results[i]), f'{computed_directory}{file_name}_laplacian_2_{edge_detection_direction_types[i]}{file_extension}')

                laplace_edge_detection_types: List[LaplaceEdgeDetectorType] = ['horizontal', 'main_diagonal', 'vertical', 'antidiagonal']

                laplace_results = [[laplaceEdgeDetection(image, type, direction_type) for type in laplace_edge_detection_types] for direction_type in edge_detection_direction_types]

                for i in range(len(laplace_results)):
                    for j in range(len(laplace_results[i])):    
                        saveImage(convertToPillowImage(laplace_results[i][j]), f'{computed_directory}{file_name}_laplace_{laplace_edge_detection_types[j]}_{edge_detection_direction_types[i]}_{file_extension}')

                sobel_result = sobelEdgeDetection(image)

                saveImage(convertToPillowImage(sobel_result), f'{computed_directory}{file_name}_sobel_{file_extension}')

                unsharp_masking_ks = [0.5, 1, 1.5]

                unsharp_masking_results = [unsharpMasking(image, k) for k in unsharp_masking_ks]

                for i in range(len(unsharp_masking_results)):
                    saveImage(convertToPillowImage(unsharp_masking_results[i]), f'{computed_directory}{file_name}_unsharp_masking_{unsharp_masking_ks[i]}{file_extension}')

                

if __name__ == "__main__":
    try:
         labTask('./assets/cameraman.tif') 

         labTask('./assets/lena_gray_256.tif')         
    except Exception as e:
        print('Error occured:')
        print_exc()