import random
from ream.config import DegradationConfig
from ream.degradation import ControlledDegrader

EXAMPLE = {
    "id": "x", "label": 1,
    "left": {"name":"Alpha", "address":"12 Main St", "postal_code":"12345", "category":"cafe", "lat":40.0, "lon":-80.0, "neighbors":[{"name":"n1"},{"name":"n2"}]},
    "right": {"name":"Alpha", "address":"12 Main Street", "postal_code":"12345", "category":"cafe", "lat":40.0001, "lon":-80.0001, "neighbors":[{"name":"m1"},{"name":"m2"}]},
}

def test_coordinate_shift_preserves_neighbors_and_label():
    d=ControlledDegrader(DegradationConfig())
    out=d.coordinate_shift(EXAMPLE, random.Random(1), 100.0)
    assert out["label"] == EXAMPLE["label"]
    assert out["left"]["neighbors"] == EXAMPLE["left"]["neighbors"]
    assert out["right"]["neighbors"] == EXAMPLE["right"]["neighbors"]

def test_neighborhood_deletion_never_removes_all():
    d=ControlledDegrader(DegradationConfig())
    out=d.neighborhood_delete(EXAMPLE, random.Random(2), 1.0)
    assert len(out["left"]["neighbors"]) >= 1
    assert len(out["right"]["neighbors"]) >= 1

def test_text_mask_keeps_branch_available():
    d=ControlledDegrader(DegradationConfig())
    out=d.text_mask(EXAMPLE, random.Random(3), 1.0)
    for side in ["left","right"]:
        vals=[out[side].get(k) for k in ("name","address","postal_code","category")]
        if out[side] != EXAMPLE[side]:
            assert any(v not in (None, "") for v in vals)

def test_neighborhood_deletion_does_not_change_spatial_density_state():
    from ream.data import spatial_features
    d=ControlledDegrader(DegradationConfig())
    before,_=spatial_features(EXAMPLE)
    out=d.neighborhood_delete(EXAMPLE, random.Random(7), 1.0)
    after,_=spatial_features(out)
    assert (before == after).all()


def test_positive_attribute_conflict_uses_partition_donor():
    from ream.degradation import PartitionDonorIndex
    donor = {
        "id":"d", "city":"x", "label":0,
        "left":{"name":"Other","address":"99 Elsewhere","postal_code":"99999","category":"cafe","lat":41.0,"lon":-81.0,"neighbors":[]},
        "right":{"name":"Other2","address":"100 Elsewhere","postal_code":"88888","category":"cafe","lat":41.1,"lon":-81.1,"neighbors":[]},
    }
    ex=dict(EXAMPLE); ex["city"]="x"
    idx=PartitionDonorIndex([ex, donor])
    d=ControlledDegrader(DegradationConfig(), idx.lookup)
    out=d.attribute_conflict(ex, random.Random(4))
    assert out["left"].get("postal_code") == "99999" or out["right"].get("postal_code") == "88888"
