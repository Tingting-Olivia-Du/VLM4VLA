import vlm4vla.data as vdata


def test_registered_in_namespace():
    assert hasattr(vdata, "LiberoActionPredictionDataset")
    assert "LiberoActionPredictionDataset" in vdata.__all__
