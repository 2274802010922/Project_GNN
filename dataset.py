import kagglehub


def download_dataset():

    path = kagglehub.dataset_download(
        "anhduy091100/vaipe-minimal-dataset"
    )

    print("Dataset downloaded to:", path)

    return path
