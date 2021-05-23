import LoadDataSetImages
import LoadDataSetTruth
import Visualize_GroundTruth
if __name__ == '__main__':
    images,filepathofimages=LoadDataSetImages.get_dataset_images("images/test")
    matrix=LoadDataSetTruth.get_dataset_truth("groundTruth/test")
    print(images[0].shape)
    Visualize_GroundTruth.visualize(filepathofimages[30],matrix[30])
    # fff.visualize2(filepathofimages[25],matrix[25])
    # aaaa.get_dataset_truth_temp("groundTruth/test")







