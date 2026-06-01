from typing import Any, Dict

from vlm4vla.data.base_action_prediction_dataset import ActionPredictionDataset
from vlm4vla.data.libero_hdf5_dataset import LiberoHDF5Dataset


class LiberoActionPredictionDataset(ActionPredictionDataset, LiberoHDF5Dataset):
    """TF-free LIBERO equivalent of OpenVLADataset.

    Reads HDF5 via LiberoHDF5Dataset and emits items through the unchanged
    ActionPredictionDataset.batch_transform (5-field contract).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        ActionPredictionDataset.__init__(self, **kwargs)
        if self.organize_type == "interleave":
            kwargs["window_sample"] = "sliding"
            kwargs["left_pad"] = False
        elif self.organize_type == "segment":
            kwargs["window_sample"] = "range"
            kwargs["left_pad"] = True
        else:
            raise ValueError("organize type must be interleave or segment")
        LiberoHDF5Dataset.__init__(self, **kwargs)

    def __iter__(self) -> Dict[str, Any]:
        for item in LiberoHDF5Dataset.__iter__(self):
            yield self.batch_transform(
                task_description=item["task_description"],
                action=item["action"],
                episode_mask=item["episode_mask"],
                images=item["images"],
                gripper_images=item["gripper_images"],
            )
