import vlm4vla.data as vdata


def test_registered_in_namespace():
    assert hasattr(vdata, "LiberoActionPredictionDataset")
    assert "LiberoActionPredictionDataset" in vdata.__all__


def test_inherits_both_parents():
    from vlm4vla.data.libero_action_prediction_dataset import LiberoActionPredictionDataset
    from vlm4vla.data.base_action_prediction_dataset import ActionPredictionDataset
    from vlm4vla.data.libero_hdf5_dataset import LiberoHDF5Dataset
    assert issubclass(LiberoActionPredictionDataset, ActionPredictionDataset)
    assert issubclass(LiberoActionPredictionDataset, LiberoHDF5Dataset)
