import copy
import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from xtuner.v1.datasets.media.config import MediaPipelineConfig
from xtuner.v1.datasets.media.provider import create_provider
from xtuner.v1.datasets.media.providers import shared
from xtuner.v1.datasets.media.schema import MaterializedMedia, MediaPlan, canonical_json
from xtuner.v1.datasets.mllm_tokenize_fn import glm53_vl_tokenize_fn as glm_tokenize
from xtuner.v1.datasets.mllm_tokenize_fn import qwen3_vl_tokenize_fn as qwen_tokenize
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import load_image
from xtuner.v1.datasets.utils import apply_exif_orientation


def _pipeline(manifest=None, **overrides):
    return MediaPipelineConfig(
        enabled=True,
        max_inflight_bytes=256 * 1024**2,
        max_pending_tasks=4,
        source_manifest=None if manifest is None else str(manifest),
        **overrides,
    )


@pytest.fixture(params=("glm", "qwen3", "qwen3.5"))
def image_fixture(request, tmp_path):
    family = request.param
    if family == "glm":
        name = "glm_image"
        processor = Glm5NextImageProcessor(min_image_tokens=4, max_image_tokens=32)
    else:
        name = "qwen_image"
        processor = Qwen2VLImageProcessor(patch_size=16, size={"shortest_edge": 4096, "longest_edge": 16384})
    identity = shared.image_processor_identity(name, processor)
    profile_key = shared.image_profile_fingerprint(identity)
    certificates = []
    native_outputs = []
    for image_index, (width, height, color) in enumerate(((64, 96, (120, 30, 200)), (96, 64, (50, 170, 20)))):
        path = tmp_path / f"image-{image_index}.png"
        image = Image.new("RGB", (width, height), color)
        exif = Image.Exif()
        exif[274] = 6 if image_index == 0 else 1
        image.save(path, exif=exif)
        image.close()
        decoded = apply_exif_orientation(load_image(str(path)))
        processed = processor(images=[decoded], return_tensors="pt")
        source_bytes = path.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        certificates.append(
            {
                "uri": str(path),
                "revision": f"sha256:{digest}",
                "content_sha256": digest,
                "source_bytes": len(source_bytes),
                "width": decoded.width,
                "height": decoded.height,
                "orientation": exif[274],
                "decode_verified": True,
                "eligible": True,
                "profiles": {
                    profile_key: {
                        "image_grid_thw": processed["image_grid_thw"][0].tolist(),
                        "pixel_values": {
                            "dtype": str(processed["pixel_values"].dtype).removeprefix("torch."),
                            "shape": list(processed["pixel_values"].shape),
                        },
                        "estimated_peak_bytes": 32 * 1024**2,
                    }
                },
            }
        )
        native_outputs.append(processed)
        decoded.close()
    manifest = tmp_path / "certificates.json"
    manifest.write_text(canonical_json({"schema_version": 1, "sources": certificates}))
    config_json = shared.make_image_provider_config(name, processor, str(manifest))
    provider = create_provider(name, config_json)
    sources = provider.sources_for_uris([certificate["uri"] for certificate in certificates])
    return SimpleNamespace(
        family=family,
        name=name,
        processor=processor,
        provider=provider,
        sources=sources,
        manifest=manifest,
        certificates=certificates,
        native_outputs=native_outputs,
        profile_key=profile_key,
        config_json=config_json,
    )


def _plan(fixture, per_image=True, sources=None, occurrence_id="record"):
    return fixture.provider.describe(
        fixture.sources if sources is None else sources,
        occurrence_id=occurrence_id,
        per_image=per_image,
        source_manifest_fingerprint=fixture.provider.manifest_fingerprint,
    )


def test_describe_is_decode_free_and_keeps_exact_effective_grid(image_fixture):
    fixture = image_fixture
    with patch.object(Image, "open", side_effect=AssertionError("describe must not decode")):
        plan = _plan(fixture)
    assert fixture.provider._processor is None
    assert fixture.sources[0].orientation == 6
    assert (fixture.sources[0].width, fixture.sources[0].height) == (96, 64)
    assert [json.loads(entry.metadata_json)["image_grid_thw"][0] for entry in plan.entries] == [
        output["image_grid_thw"][0].tolist() for output in fixture.native_outputs
    ]
    assert MediaPlan.from_json(plan.to_json()) == plan
    assert plan.processor_fingerprint == hashlib.sha256(plan.processor_config_json.encode()).hexdigest()
    assert plan.source_manifest_fingerprint == hashlib.sha256(fixture.manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("per_image", (False, True))
def test_real_native_processor_matches_provider_grouped_and_per_image(image_fixture, per_image):
    fixture = image_fixture
    plan = _plan(fixture, per_image)
    outputs = [fixture.provider.process(entry) for entry in plan.entries]
    actual = fixture.provider.assemble(plan, outputs)
    if fixture.name == "glm_image":
        expected = {
            field: torch.cat([output[field] for output in fixture.native_outputs])
            for field in ("pixel_values", "image_grid_thw")
        }
        assert len(plan.entries) == 2
    else:
        images = [apply_exif_orientation(load_image(source.uri)) for source in fixture.sources]
        try:
            expected = fixture.processor.preprocess(images, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        assert len(plan.entries) == (2 if per_image else 1)
    for field in ("pixel_values", "image_grid_thw"):
        assert actual.tensors[field].dtype == expected[field].dtype
        torch.testing.assert_close(actual.tensors[field], expected[field], rtol=0, atol=0)
    with pytest.raises(ValueError, match="occurrence order"):
        fixture.provider.assemble(
            _plan(fixture), list(reversed([fixture.provider.process(entry) for entry in _plan(fixture).entries]))
        )


def test_merge_keeps_occurrences_and_rejects_identity_change(image_fixture):
    fixture = image_fixture
    plans = [
        _plan(fixture, sources=[source], occurrence_id=f"record-{index}")
        for index, source in enumerate(fixture.sources)
    ]
    merged = MediaPlan.merge(plans)
    assert merged.entries == tuple(entry for plan in plans for entry in plan.entries)
    assert merged.sample_ids == ("record-0", "record-1")
    assert merged.output_specs == _plan(fixture).output_specs
    with pytest.raises(ValueError, match="identities"):
        MediaPlan.merge([plans[0], replace(plans[1], source_manifest_fingerprint="different")])
    repeated = MediaPlan.merge([plans[0], plans[0]])
    assert len(repeated.entries) == 2
    assert repeated.output_specs[0].shape[0] == 2 * plans[0].output_specs[0].shape[0]


def test_source_content_is_checked_before_decode(image_fixture):
    fixture = image_fixture
    plan = _plan(fixture)
    source_path = fixture.manifest.parent / "image-0.png"
    source_bytes = source_path.read_bytes()
    source_path.write_bytes(bytes([source_bytes[0] ^ 1]) + source_bytes[1:])
    with patch.object(Image, "open", side_effect=AssertionError("changed source must fail before decode")):
        with pytest.raises(ValueError, match="bytes differ"):
            fixture.provider.process(plan.entries[0])


def test_changed_manifest_and_weak_descriptors_are_rejected(image_fixture):
    fixture = image_fixture
    with pytest.raises(ValueError, match="descriptor differs"):
        fixture.provider.describe(
            [replace(fixture.sources[0], metadata_json='{"forged_certificate":true}')],
            occurrence_id="bad",
            per_image=True,
            source_manifest_fingerprint=fixture.provider.manifest_fingerprint,
        )
    fixture.manifest.write_text(fixture.manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        create_provider(fixture.name, fixture.config_json)


@pytest.mark.parametrize("mutation", ("decode", "profile", "exif", "spatial_grid", "revision"))
def test_certificate_counterexamples_are_not_silently_accepted(image_fixture, mutation):
    fixture = image_fixture
    certificates = copy.deepcopy(fixture.certificates)
    certificate = certificates[0]
    if mutation == "decode":
        certificate["decode_verified"] = False
    elif mutation == "profile":
        certificate["profiles"] = {}
    elif mutation == "exif":
        certificate["orientation"] = 1
    elif mutation == "revision":
        certificate["revision"] = certificate["uri"]
    else:
        grid = certificate["profiles"][fixture.profile_key]["image_grid_thw"]
        assert grid[1] != grid[2]
        grid[1], grid[2] = grid[2], grid[1]
    fixture.manifest.write_text(canonical_json({"schema_version": 1, "sources": certificates}))
    provider = create_provider(
        fixture.name, shared.make_image_provider_config(fixture.name, fixture.processor, str(fixture.manifest))
    )
    if mutation in ("decode", "profile", "revision"):
        with pytest.raises(ValueError):
            provider.sources_for_uris([certificate["uri"]])
    else:
        sources = provider.sources_for_uris([certificate["uri"]])
        plan = provider.describe(
            sources, occurrence_id="bad", per_image=True, source_manifest_fingerprint=provider.manifest_fingerprint
        )
        with pytest.raises(ValueError, match="orientation|grid"):
            provider.process(plan.entries[0])


def _tokenize_instance(fixture):
    if fixture.family == "glm":
        tokenize_fn = object.__new__(glm_tokenize.Glm53VLTokenizeFunction)
    else:
        tokenize_fn = object.__new__(qwen_tokenize.Qwen3VLTokenizeFunction)
    tokenize_fn.data_name = "synthetic"
    tokenize_fn._image_path = [source.uri for source in fixture.sources]
    tokenize_fn.image_processor = fixture.processor
    tokenize_fn._media_oss_loader_config = None
    tokenize_fn.oss_loader = None
    tokenize_fn.media_pipeline = None
    tokenize_fn.max_length = None
    return tokenize_fn


def test_eager_uses_common_materializer_and_lazy_never_reads_pixels(image_fixture):
    fixture = image_fixture
    tokenize_fn = _tokenize_instance(fixture)
    record = {"id": "record", "messages": []}
    assert shared.prepare_image_inputs(tokenize_fn, record, "", fixture.name) is None
    tokenize_fn.media_pipeline = _pipeline(fixture.manifest, materialization="lazy", per_image=True)
    with patch.object(shared.CertifiedImageProvider, "_read_source", side_effect=AssertionError("lazy decode")):
        pixels, grids, plan = shared.prepare_image_inputs(tokenize_fn, record, "", fixture.name)
    assert pixels is None
    assert plan is not None
    assert grids.tolist() == [output["image_grid_thw"][0].tolist() for output in fixture.native_outputs]
    tokenize_fn.media_pipeline = _pipeline(fixture.manifest, materialization="eager", per_image=True)
    materialized = fixture.provider.assemble(plan, [fixture.provider.process(entry) for entry in plan.entries])
    with patch.object(shared, "_materialize_plans", return_value=[materialized]) as common:
        pixels, eager_grid, eager_plan = shared.prepare_image_inputs(tokenize_fn, record, "", fixture.name)
    assert common.call_count == 1
    assert pixels is materialized.tensors["pixel_values"]
    assert torch.equal(eager_grid, grids)
    assert eager_plan is None


def test_missing_certificate_and_video_choose_policy_before_collation(image_fixture):
    fixture = image_fixture
    tokenize_fn = _tokenize_instance(fixture)
    tokenize_fn.media_pipeline = _pipeline()
    with pytest.raises(shared.UnsupportedMediaSource, match="source_manifest"):
        shared.prepare_image_inputs(tokenize_fn, {}, "", fixture.name)
    with pytest.raises(shared.UnsupportedMediaSource, match="video"):
        shared.check_video_policy(tokenize_fn)
    tokenize_fn.media_pipeline = _pipeline(unsupported_policy="eager")
    assert shared.prepare_image_inputs(tokenize_fn, {}, "", fixture.name) is None
    shared.check_video_policy(tokenize_fn)


def test_mixed_pack_resolves_only_before_collation_without_mutating_items(image_fixture):
    fixture = image_fixture
    plan = _plan(fixture)
    planned = {"input_ids": [1], "media_plan": plan, "image_grid_thw": torch.tensor([[1, 4, 4]])}
    native = {"input_ids": [2], "pixel_values_videos": torch.zeros(1, 1)}
    with pytest.raises(shared.UnsupportedMediaSource, match="before collation"):
        shared.resolve_mixed_media_items([planned, native], _pipeline(fixture.manifest))
    materialized = MaterializedMedia({"pixel_values": torch.zeros(1, 1), "image_grid_thw": torch.ones(1, 3)})
    with patch.object(shared, "_materialize_plans", return_value=[materialized]) as common:
        resolved = shared.resolve_mixed_media_items(
            [planned, native], _pipeline(fixture.manifest, unsupported_policy="eager")
        )
    common.assert_called_once()
    assert "media_plan" in planned and "pixel_values" not in planned
    assert "media_plan" not in resolved[0]
    assert resolved[0]["pixel_values"] is materialized.tensors["pixel_values"]
    assert resolved[1]["pixel_values_videos"] is native["pixel_values_videos"]


def test_discarded_video_does_not_trigger_fallback(image_fixture):
    fixture = image_fixture
    planned = {"input_ids": [1], "media_plan": _plan(fixture)}
    text = {"input_ids": [2], "labels": [-100], "num_imgs": [0]}
    discarded_video = {"input_ids": [3], "pixel_values_videos": torch.zeros(1, 1)}
    candidates = [planned, text, discarded_video]
    retained = candidates[:2]
    with patch.object(shared, "_materialize_plans", side_effect=AssertionError("discarded video fallback")):
        resolved = shared.resolve_mixed_media_items(retained, _pipeline(fixture.manifest, unsupported_policy="eager"))
    assert resolved == retained
    assert resolved[0]["media_plan"] is planned["media_plan"]


def test_family_native_and_lazy_token_metadata_match(image_fixture):
    fixture = image_fixture
    tokenize_fn = _tokenize_instance(fixture)
    counts = [
        output["image_grid_thw"].prod().item() // fixture.processor.merge_size**2 for output in fixture.native_outputs
    ]
    record = {"id": "record", "messages": [{"role": "user", "content": "synthetic"}]}
    if fixture.family == "glm":
        tokenize_fn.merge_unit = fixture.processor.merge_size**2
        tokenize_fn.processor = SimpleNamespace(image_token_id=7, replace_image_token=Mock(return_value="visual"))
        tokenize_fn._render_base = Mock(return_value=("<|image|>" * 2, [False] * (len("<|image|>") * 2)))
        input_ids = [10] + [7] * sum(counts) + [11]
        tokenize_fn._finalize = Mock(return_value=(input_ids, [-100] * len(input_ids), [0] * len(input_ids)))
        native = tokenize_fn.multi_modal_get_item(record)
        tokenize_fn.media_pipeline = _pipeline(fixture.manifest, materialization="lazy", per_image=True)
        lazy = tokenize_fn.multi_modal_get_item(record)
        assert torch.equal(native["mm_token_type_ids"], lazy["mm_token_type_ids"])
    else:
        is_qwen35 = fixture.family == "qwen3.5"
        image_id, start_id, end_id = (248056, 248053, 248054) if is_qwen35 else (151655, 151652, 151653)
        tokenize_fn.img_context_token_id = image_id
        tokenize_fn.video_context_token_id = 248057 if is_qwen35 else 151656
        tokenize_fn.img_start_token_id = start_id
        tokenize_fn.img_end_token_id = end_id
        tokenize_fn.merge_length = fixture.processor.merge_size**2
        tokenize_fn.chat_template_name = "qwen3.5-vl" if is_qwen35 else "qwen3-vl"
        tokenize_fn.chat_template = Mock()
        tokenize_fn.tokenizer = Mock()
        tokenize_fn.enable_3d_rope = True
        tokenize_fn.add_vision_id = False
        tokenize_fn.add_generation_prompt = False
        tokenize_fn.enable_thinking = None

        def tokenize(*args, **kwargs):
            input_ids = [10]
            for count in counts:
                input_ids.extend([start_id] + [image_id] * (1 if is_qwen35 else count) + [end_id])
            input_ids.append(11)
            return {"input_ids": input_ids, "labels": [-100] * len(input_ids)}

        messages = Mock(messages=[SimpleNamespace(role="user")])
        messages.tokenize.side_effect = tokenize
        with patch.multiple(
            qwen_tokenize,
            ChatMessages=Mock(return_value=messages),
            Qwen35ChatMessages=Mock(return_value=messages),
            replace_image_token=Mock(),
        ):
            native = tokenize_fn.multi_modal_get_item(record)
            tokenize_fn.media_pipeline = _pipeline(fixture.manifest, materialization="lazy", per_image=True)
            lazy = tokenize_fn.multi_modal_get_item(record)
        assert torch.equal(native["position_ids"], lazy["position_ids"])
    assert "pixel_values" in native and "media_plan" not in native
    assert "pixel_values" not in lazy and lazy["media_plan"] is not None
    assert torch.equal(native["image_grid_thw"], lazy["image_grid_thw"])
    for field in ("input_ids", "labels", "num_tokens", "num_img_tokens", "num_imgs"):
        assert native[field] == lazy[field]


@pytest.mark.parametrize(
    "family_module,config_class",
    (
        (glm_tokenize, glm_tokenize.Glm53VLTokenizeFnConfig),
        (qwen_tokenize, qwen_tokenize.Qwen3VLTokenizeFnConfig),
    ),
)
def test_family_build_forwards_validated_pipeline(family_module, config_class):
    pipeline = _pipeline(unsupported_policy="eager")
    config = config_class(processor_path="unused", media_pipeline=pipeline)
    constructor = "Glm53VLTokenizeFunction" if family_module is glm_tokenize else "Qwen3VLTokenizeFunction"
    assert config.supports_media_pipeline is True
    with patch.object(family_module, constructor) as build:
        config.build(Mock())
    assert build.call_args.kwargs["media_pipeline"] is pipeline


@pytest.mark.parametrize("fail", (False, True))
def test_common_materializer_is_closed_even_on_failure(monkeypatch, fail):
    instance = Mock()
    factory = Mock(return_value=instance)
    monkeypatch.setitem(
        sys.modules, "xtuner.v1.datasets.media.materializer", SimpleNamespace(MediaMaterializer=factory)
    )
    plan = object()
    config = object()
    if fail:
        instance.materialize.side_effect = ValueError("native processing failure")
        with pytest.raises(ValueError, match="native processing failure"):
            shared._materialize_plans([plan], config)
    else:
        assert shared._materialize_plans([plan], config) == [instance.materialize.return_value]
    factory.assert_called_once_with(config)
    instance.materialize.assert_called_once_with(plan)
    instance.close.assert_called_once_with()


def _with_pixel_dtype(fixture, pixel_dtype):
    configured = copy.copy(fixture)
    configured.config_json = shared.make_image_provider_config(
        fixture.name, fixture.processor, str(fixture.manifest), pixel_dtype=pixel_dtype
    )
    configured.provider = create_provider(fixture.name, configured.config_json)
    return configured


def test_pixel_dtype_config_is_opt_in_and_keeps_native_serialization():
    native = MediaPipelineConfig()
    assert native.pixel_dtype == "native"
    assert "pixel_dtype" not in native.model_dump()
    assert "pixel_dtype" not in json.loads(native.model_dump_json())
    assert native.model_dump() == MediaPipelineConfig(pixel_dtype="native").model_dump()
    enabled_native = _pipeline()
    bf16 = _pipeline(pixel_dtype="bfloat16")
    assert bf16.model_dump() == {**enabled_native.model_dump(), "pixel_dtype": "bfloat16"}
    assert MediaPipelineConfig.model_validate_json(bf16.model_dump_json()) == bf16
    with pytest.raises(ValueError, match="features require"):
        MediaPipelineConfig(pixel_dtype="bfloat16")
    with pytest.raises(ValueError):
        _pipeline(pixel_dtype="float16")


def test_native_pixel_dtype_preserves_config_and_fingerprints(image_fixture):
    fixture = image_fixture
    legacy_config = {
        "identity": shared.image_processor_identity(fixture.name, fixture.processor),
        "source_manifest": str(fixture.manifest.resolve()),
        "source_manifest_fingerprint": hashlib.sha256(fixture.manifest.read_bytes()).hexdigest(),
        "oss_loader_config": None,
    }
    explicit_native = _with_pixel_dtype(fixture, "native")
    assert fixture.config_json == explicit_native.config_json == canonical_json(legacy_config)
    assert explicit_native.provider.profile_fingerprint == fixture.provider.profile_fingerprint
    assert explicit_native.provider.processor_fingerprint == fixture.provider.processor_fingerprint
    assert _plan(explicit_native).fingerprint == _plan(fixture).fingerprint


def test_bf16_pixel_dtype_changes_only_transport_identity_and_specs(image_fixture):
    native = image_fixture
    bf16 = _with_pixel_dtype(native, "bfloat16")
    native_config = json.loads(native.config_json)
    assert json.loads(bf16.config_json) == {**native_config, "pixel_dtype": "bfloat16"}
    assert bf16.provider.identity == native.provider.identity
    assert bf16.provider.profile_fingerprint == native.provider.profile_fingerprint
    assert bf16.provider.manifest_fingerprint == native.provider.manifest_fingerprint
    assert bf16.provider.processor_fingerprint != native.provider.processor_fingerprint
    native_plan, bf16_plan = _plan(native), _plan(bf16)
    assert native_plan.fingerprint != bf16_plan.fingerprint
    assert bf16_plan.source_manifest_fingerprint == native_plan.source_manifest_fingerprint
    assert bf16_plan.output_specs[0] == replace(native_plan.output_specs[0], dtype="bfloat16")
    assert bf16_plan.output_specs[1] == native_plan.output_specs[1]
    for entry in bf16_plan.entries:
        certified_peak = sum(bf16.provider._profile(source)["estimated_peak_bytes"] for source in entry.sources)
        assert entry.estimated_peak_bytes == certified_peak + sum(spec.nbytes for spec in entry.output_specs)
    with pytest.raises(ValueError, match="identities"):
        MediaPlan.merge([bf16_plan, native_plan])


@pytest.mark.parametrize("per_image", (False, True))
def test_bf16_pixel_dtype_casts_after_native_processor_before_assembly(image_fixture, per_image):
    native = image_fixture
    bf16 = _with_pixel_dtype(native, "bfloat16")
    certificate_bytes = native.manifest.read_bytes()
    native_plan, bf16_plan = _plan(native, per_image), _plan(bf16, per_image)
    native_outputs = [native.provider.process(entry) for entry in native_plan.entries]
    bf16_outputs = [bf16.provider.process(entry) for entry in bf16_plan.entries]
    assert len(native_outputs) == len(bf16_outputs)
    for native_output, bf16_output in zip(native_outputs, bf16_outputs):
        native_pixels = native_output.tensors["pixel_values"]
        bf16_pixels = bf16_output.tensors["pixel_values"]
        assert native_pixels.dtype == torch.float32
        assert bf16_pixels.dtype == torch.bfloat16
        assert bf16_pixels.device.type == "cpu"
        assert torch.equal(bf16_pixels.view(torch.int16), native_pixels.to(torch.bfloat16).view(torch.int16))
        assert torch.equal(bf16_output.tensors["image_grid_thw"], native_output.tensors["image_grid_thw"])
        assert bf16_output.tensors["image_grid_thw"].dtype == torch.int64
        assert bf16_output.metadata == native_output.metadata
    assembled = bf16.provider.assemble(bf16_plan, bf16_outputs)
    expected_pixels = torch.cat([output.tensors["pixel_values"].to(torch.bfloat16) for output in native_outputs])
    assert torch.equal(assembled.tensors["pixel_values"].view(torch.int16), expected_pixels.view(torch.int16))
    assert assembled.nbytes == assembled.storage_bytes == bf16_plan.output_bytes
    assert native.manifest.read_bytes() == certificate_bytes


@pytest.mark.parametrize("mutation", ("dtype", "shape", "grid"))
def test_bf16_pixel_dtype_rejects_invalid_native_output_before_cast(image_fixture, mutation):
    fixture = _with_pixel_dtype(image_fixture, "bfloat16")
    processed = {
        name: image_fixture.native_outputs[0][name].clone() for name in ("pixel_values", "image_grid_thw")
    }
    if mutation == "dtype":
        processed["pixel_values"] = processed["pixel_values"].to(torch.bfloat16)
    elif mutation == "shape":
        processed["pixel_values"] = processed["pixel_values"][:-1]
    else:
        processed["image_grid_thw"][:, 1:] = processed["image_grid_thw"][:, 1:].flip(-1)
    plan = _plan(fixture)
    with patch.object(fixture.provider, "_preprocess", return_value=processed):
        with pytest.raises(ValueError, match="shape/dtype|native grid"):
            fixture.provider.process(plan.entries[0])


def test_bf16_pixel_dtype_keeps_native_certificate_peak_floor(image_fixture):
    fixture = image_fixture
    certificates = copy.deepcopy(fixture.certificates)
    certificate = certificates[0]
    native_output = fixture.native_outputs[0]
    minimum_peak = sum(
        native_output[name].numel() * native_output[name].element_size() for name in ("pixel_values", "image_grid_thw")
    )
    minimum_peak += certificate["source_bytes"] + certificate["width"] * certificate["height"] * 3
    certificate["profiles"][fixture.profile_key]["estimated_peak_bytes"] = minimum_peak - 1
    manifest = fixture.manifest.with_name("underreserved-certificates.json")
    manifest.write_text(canonical_json({"schema_version": 1, "sources": certificates}))
    provider = create_provider(
        fixture.name,
        shared.make_image_provider_config(fixture.name, fixture.processor, str(manifest), pixel_dtype="bfloat16"),
    )
    with patch.object(Image, "open", side_effect=AssertionError("invalid peak must fail before decode")):
        with pytest.raises(ValueError, match="peak reservation is smaller"):
            sources = provider.sources_for_uris([certificate["uri"]])
            provider.describe(
                sources, occurrence_id="bad-peak", per_image=True,
                source_manifest_fingerprint=provider.manifest_fingerprint,
            )


def test_bf16_pixel_dtype_materializer_transport_schema_and_proof_agree(image_fixture):
    from xtuner.v1.datasets.media.runtime import MediaRuntime
    from xtuner.v1.datasets.media.schema import MediaDeliveryProof

    fixture = _with_pixel_dtype(image_fixture, "bfloat16")
    plan = _plan(fixture)
    pipeline = _pipeline(fixture.manifest, pixel_dtype="bfloat16", per_image=True, decode_threads=2)
    with MediaRuntime(pipeline) as runtime:
        with runtime.consume(plan, key="bf16-parity", device="cpu") as media:
            media.validate(plan.output_specs)
            assert media.tensors["pixel_values"].dtype == torch.bfloat16
            assert media.tensors["image_grid_thw"].dtype == torch.int64
            assert media.nbytes == plan.output_bytes
            document = json.loads(runtime._transport._schema(plan, media.metadata))
            assert {spec["name"]: spec["dtype"] for spec in document["specs"]} == {
                "pixel_values": "bfloat16", "image_grid_thw": "int64",
            }
            proof = MediaDeliveryProof("bf16-parity", plan.fingerprint, plan.output_specs)
            proof.validate_context(SimpleNamespace(**media.tensors))
            mismatched = {**media.tensors, "pixel_values": media.tensors["pixel_values"].float()}
            with pytest.raises(ValueError, match="dtype changed"):
                proof.validate_context(SimpleNamespace(**mismatched))
    assert runtime.metrics["host_reserved_bytes"] == 0


def test_bf16_pixel_dtype_is_forwarded_to_provider(image_fixture):
    tokenize_fn = _tokenize_instance(image_fixture)
    tokenize_fn.media_pipeline = _pipeline(image_fixture.manifest, pixel_dtype="bfloat16")
    provider = shared.initialize_image_provider(tokenize_fn, image_fixture.name)
    assert provider.pixel_dtype == "bfloat16"
    assert json.loads(provider.config_json)["pixel_dtype"] == "bfloat16"
    assert provider.profile_fingerprint == image_fixture.profile_key


def test_pixel_dtype_rejects_unknown_provider_values(image_fixture):
    fixture = image_fixture
    with pytest.raises(ValueError, match="pixel_dtype"):
        shared.make_image_provider_config(
            fixture.name, fixture.processor, str(fixture.manifest), pixel_dtype="float16"
        )
    config = json.loads(fixture.config_json)
    config["pixel_dtype"] = "float16"
    with pytest.raises(ValueError, match="pixel_dtype"):
        create_provider(fixture.name, canonical_json(config))


def _embedded_certificate_plan(fixture, plan):
    """Reconstruct the V1 wire representation for the exact same occurrences."""
    return replace(
        plan,
        entries=tuple(
            replace(entry, sources=tuple(
                replace(source, metadata_json=canonical_json(fixture.provider.certificates[source.uri]))
                for source in entry.sources
            ))
            for entry in plan.entries
        ),
    )


def test_compact_sources_keep_semantics_and_read_old_plans(image_fixture):
    import pickle
    from contextlib import closing

    from xtuner.v1.datasets.media.materializer import MediaMaterializer

    fixture = _with_pixel_dtype(image_fixture, "bfloat16")
    compact = _plan(fixture)
    embedded = _embedded_certificate_plan(fixture, compact)
    assert all(source.metadata_json == "{}" for entry in compact.entries for source in entry.sources)
    assert compact.output_specs == embedded.output_specs
    assert compact.processor_fingerprint == embedded.processor_fingerprint
    assert compact.source_manifest_fingerprint == embedded.source_manifest_fingerprint
    assert compact.fingerprint != embedded.fingerprint  # Representation changed; never reuse an old plan identity.
    assert len(compact.to_json().encode()) < len(embedded.to_json().encode())
    assert MediaPlan.from_json(compact.to_json()) == compact
    assert pickle.loads(pickle.dumps(compact)) == compact
    assert MediaPlan.from_json(embedded.to_json()) == embedded
    assert tuple(entry.estimated_peak_bytes for entry in compact.entries) == tuple(
        entry.estimated_peak_bytes for entry in embedded.entries
    )
    with closing(MediaMaterializer(_pipeline(fixture.manifest, pixel_dtype="bfloat16"))) as materializer:
        before = materializer.materialize(embedded)
        after = materializer.materialize(compact)
        # Only descriptor/control bytes shrink; native image/pixel reservations do not.
        difference = len(embedded.to_json().encode()) - len(compact.to_json().encode())
        assert materializer.reservation_bytes(embedded) - materializer.reservation_bytes(compact) == 4 * difference
    for field in compact.output_specs:
        assert torch.equal(before.tensors[field.name], after.tensors[field.name])
    assert before.tensors["pixel_values"].dtype == torch.bfloat16


@pytest.mark.parametrize("embedded", (False, True))
@pytest.mark.parametrize("field,value", [
    ("revision", "object-version:forged"),
    ("width", 128),
    ("height", 128),
    ("orientation", 8),
    ("source_bytes", 99999),
    ("content_sha256", "f" * 64),
])
def test_compact_and_old_descriptors_reject_changed_source_facts_before_decode(image_fixture, embedded, field, value):
    fixture = image_fixture
    source = fixture.sources[0]
    if embedded:
        source = replace(source, metadata_json=canonical_json(fixture.certificates[0]))
    source = replace(source, **{field: value})
    with patch.object(Image, "open", side_effect=AssertionError("forged descriptor reached decode")):
        with pytest.raises(ValueError, match="identity/geometry differs"):
            _plan(fixture, sources=[source])


def test_old_embedded_certificate_cannot_override_frozen_manifest(image_fixture):
    fixture = image_fixture
    forged = copy.deepcopy(fixture.certificates[0])
    forged["profiles"][fixture.profile_key]["image_grid_thw"][1] += 2
    source = replace(fixture.sources[0], metadata_json=canonical_json(forged))
    with pytest.raises(ValueError, match="descriptor differs"):
        _plan(fixture, sources=[source])


def test_compact_descriptors_do_not_replicate_unselected_certificate_data(image_fixture):
    fixture = image_fixture
    original = _plan(fixture)
    certificates = copy.deepcopy(fixture.certificates)
    for certificate in certificates:
        certificate["profiles"]["another-processor-profile"] = {"unrelated": "x" * 16384}
        certificate["audit_note"] = "offline verification notes" * 100
    manifest = fixture.manifest.with_name("many-profiles.json")
    manifest.write_text(canonical_json({"schema_version": 1, "sources": certificates}))
    provider = create_provider(
        fixture.name, shared.make_image_provider_config(fixture.name, fixture.processor, str(manifest))
    )
    sources = provider.sources_for_uris([source.uri for source in fixture.sources])
    assert sources == fixture.sources
    plan = provider.describe(sources, occurrence_id="record", per_image=True,
                             source_manifest_fingerprint=provider.manifest_fingerprint)
    assert plan.entries == original.entries
    assert plan.source_manifest_fingerprint != original.source_manifest_fingerprint
    assert plan.fingerprint != original.fingerprint  # Unselected data is still bound by the manifest hash.
    assert "unrelated" not in plan.to_json()
    # No repeated certificate serialization on the compact lookup path.
    with patch.object(shared, "canonical_json", side_effect=AssertionError("lookup serialized a certificate")):
        assert provider._profile(sources[0]) == certificates[0]["profiles"][fixture.profile_key]
