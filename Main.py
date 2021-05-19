import LoadDataSetImages
import LoadDataSetTruth
import Visualize_GroundTruth
if __name__ == '__main__':
    images,filepathofimages=LoadDataSetImages.get_dataset_images("images/test")
    matrix=LoadDataSetTruth.get_dataset_truth("groundTruth/test")
    Visualize_GroundTruth.visualize(matrix)



