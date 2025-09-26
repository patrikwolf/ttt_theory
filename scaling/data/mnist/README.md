# MNIST Data

To download and processing the MNIST dataset, please follow these steps:

1. **Download the MNIST Dataset**: Execute the `data_generator.py` script located in the `scaling/data/mnist/` directory. 
This script will automatically download the MNIST dataset and save it in the appropriate format.
2. **Generate Embeddings**: For the TTT procedures, we will use cached embeddings. To generate these embeddings based on a pre-trained
model, run the `generate_embeddings.py` script in the directory `scaling/mnist/a_global_classifier`.