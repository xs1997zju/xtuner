import copy
import json
import pickle
from dataclasses import replace

import pytest

from xtuner.v1.datasets.media import (
    MediaDeliveryProof,
    MediaEntry,
    MediaPipelineConfig,
    MediaPlan,
    MediaSource,
    TensorSpec,
)
from xtuner.v1.datasets.media.provider import create_provider, register_provider


def make_plan():
    source = MediaSource("fixture://one", "manifest-sha", 32, 48)
    spec = TensorSpec("fixture", "float32", (6, 4))
    entry = MediaEntry("record-1:image-0", (source,), (spec,), 4096)
    return MediaPlan("fixture", "1", "{}", "processor-sha", "manifest-sha", (entry,), (spec,), ("record-1",))


def test_plan_serialization_has_no_runtime_payload():
    plan = make_plan()
    assert MediaPlan.from_json(plan.to_json()) == plan
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert copy.deepcopy(plan).fingerprint == plan.fingerprint
    assert plan.output_bytes == 96
    assert json.loads(plan.to_json())["entries"][0]["sources"][0]["uri"] == "fixture://one"


def test_source_revision_is_part_of_identity():
    plan = make_plan()
    changed = plan.to_dict()
    changed["entries"][0]["sources"][0]["revision"] = "different-version"
    assert MediaPlan.from_dict(changed).fingerprint != plan.fingerprint


def test_missing_source_identity_is_rejected():
    with pytest.raises(ValueError, match="identity"):
        MediaSource("fixture://one", "", 32, 48)


@pytest.mark.parametrize("shape", [(True, 3), (-1, 3), (2**62, 8)])
def test_untrusted_shapes_rejected_before_allocation(shape):
    with pytest.raises(ValueError):
        TensorSpec("fixture", "float32", shape)


def test_plan_rejects_payload_metadata():
    with pytest.raises((TypeError, ValueError)):
        MediaSource("fixture://one", "revision", 32, 48, metadata_json={"object": object()})


def test_disabled_is_native_and_explicit_requests_not_ignored():
    assert not MediaPipelineConfig().enabled
    with pytest.raises(ValueError, match="enabled"):
        MediaPipelineConfig(per_image=True)


@pytest.mark.parametrize("options", [
    {},
    {"max_inflight_bytes": 1000, "max_pending_tasks": 1, "decode_threads": 2},
    {"max_inflight_bytes": 1000, "max_pending_tasks": 1, "prefetch_batches": 1},
    {"max_inflight_bytes": 1000, "max_pending_tasks": 1, "materialization": "lazy"},
])
def test_invalid_enabled_combinations_rejected(options):
    with pytest.raises(ValueError):
        MediaPipelineConfig(enabled=True, **options)


def test_mock_provider_registration_needs_no_model_dispatch():
    class MockProvider:
        def __init__(self, settings):
            self.settings = settings

    register_provider("schema-fixture-v1", MockProvider)
    assert create_provider("schema-fixture-v1", '{"variant":1}').settings == '{"variant":1}'
    with pytest.raises(ValueError, match="duplicate"):
        register_provider("schema-fixture-v1", MockProvider)


def test_delivery_proof_rejects_dropped_media_field():
    proof = MediaDeliveryProof("run:step:micro", make_plan().fingerprint, make_plan().output_specs)
    with pytest.raises(ValueError, match="missing"):
        proof.validate_context(type("Context", (), {"fixture": None})())


@pytest.mark.parametrize("field,value", [
    ("provider", "another"),
    ("provider_version", "2"),
    ("processor_config_json", '{"variant":1}'),
    ("processor_fingerprint", "different"),
    ("source_manifest_fingerprint", "different"),
])
def test_packing_rejects_mixed_processing_identities(field, value):
    plan = make_plan()
    with pytest.raises(ValueError, match="identities"):
        MediaPlan.merge([plan, replace(plan, **{field: value})])


def test_packing_preserves_repetition_order_and_nested_decisions():
    first = make_plan()
    second = replace(first, sample_ids=("record-2",), decisions_json='{"crop":2}')
    merged = MediaPlan.merge([first, second, first])
    assert merged.entries == first.entries + second.entries + first.entries
    assert merged.sample_ids == ("record-1", "record-2", "record-1")
    assert merged.output_specs == (TensorSpec("fixture", "float32", (18, 4)),)
    assert json.loads(merged.decisions_json) == {"merged_decisions": [{}, {"crop": 2}, {}]}
    nested = MediaPlan.merge([merged])
    assert json.loads(nested.decisions_json) == {"merged_decisions": [{"merged_decisions": [{}, {"crop": 2}, {}]}]}
    assert MediaPlan.from_json(merged.to_json()) == merged
    assert pickle.loads(pickle.dumps(merged)) == merged


def test_packing_rejects_schema_lie_before_allocating():
    plan = make_plan()
    changed = replace(plan, output_specs=(TensorSpec("fixture", "float32", (12, 4)),))
    with pytest.raises(ValueError, match="ordered semantic units"):
        MediaPlan.merge([changed])
    with pytest.raises(ValueError, match="empty"):
        MediaPlan.merge([])


@pytest.mark.parametrize("spec", [
    TensorSpec("renamed", "float32", (6, 4)),
    TensorSpec("fixture", "bfloat16", (6, 4)),
    TensorSpec("fixture", "float32", (6, 5)),
    TensorSpec("fixture", "float32", (6, 4), concat_dim=1),
    TensorSpec("fixture", "float32", (6, 4), concat_dim=None),
])
def test_packing_rejects_incompatible_tensor_layouts(spec):
    plan = make_plan()
    changed = replace(plan, entries=(replace(plan.entries[0], output_specs=(spec,)),), output_specs=(spec,))
    with pytest.raises(ValueError, match="schemas"):
        MediaPlan.merge([plan, changed])
