from scaling.data.imagenet.data_generator import process_split

if __name__ == "__main__":
    # Define batch size
    batch_size = 64

    # Run both splits
    process_split("val", batch_size=batch_size)
    process_split("train", batch_size=batch_size)
    print('Done!')