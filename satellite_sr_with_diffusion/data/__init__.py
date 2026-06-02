"""Create dataset and dataloader."""

import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    """
    Creates a dataloader for a given dataset.

    Args:
        dataset: The dataset object.
        dataset_opt (dict): Options for the dataset.
        phase (str): 'train' or 'val'.
    """
    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=dataset_opt["use_shuffle"],
            num_workers=dataset_opt["num_workers"],
            pin_memory=True,
        )
    if phase == "val":
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
    raise NotImplementedError(f"Dataloader [{phase}] is not found.")
