# ImageNet Data

To download and processing the ImageNet dataset, please follow these steps:

1. **Compute CLIP embeddings of ImageNet dataset**: Execute the `data_generator.py` script located in the 
`scaling/data/imagenet/` directory. This script will automatically download the ImageNet dataset, compute the
CLIP embeddings for each image, and save it in the appropriate format.
2. **Generate test and validation set**: Since the ImageNet dataset does not come with a labeled test set, we 
create a test set by randomly selecting 50 images from each class in the training set. The remaining images in the training set
are used as the training set. This is done in the `test_val_split.py` script.