"""Real processor parity for the native, image-only Qwen lazy adapter."""
import copy
import os

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from xtuner.v1.datasets.mllm_tokenize_fn import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_tokenize_fn import LAZY_CHAT_TEMPLATES
from xtuner.v1.train.lazy_media_prefetch import materialize_image_plan
from xtuner.v1.datasets.collator import qwen3_vl_sft_collator
from xtuner.v1.data_proto.sequence_context import SequenceContext

@pytest.fixture(scope="module", params=LAZY_CHAT_TEMPLATES)
def tokenize(request):
    template = request.param
    variable = "QWEN3_VL_PATH" if template == "qwen3-vl" else "QWEN3_5_MOE_PATH"
    model = os.environ.get(variable)
    if not model:
        pytest.skip(f"{variable} is required for real-processor parity")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return Qwen3VLTokenizeFnConfig(
        processor_path=model, chat_template=template, min_pixels=4096,
        max_pixels=16384, max_length=8192,
    ).build(tokenizer, anno_name="native-lazy-parity")


def compare(reference, actual):
    for key in reference:
        if key == "pixel_values":
            continue
        if isinstance(reference[key], torch.Tensor):
            assert torch.equal(reference[key], actual[key]), key
        else:
            assert reference[key] == actual[key], key


def sample(tmp_path, template, *, exif=False):
    entries = []
    for index, size in enumerate(((96, 64), (64, 128), (96, 64))):
        path = tmp_path / f"image-{index}.png"
        image = Image.new("RGB", size, (10 + index, 80, 130))
        if exif:
            tags = Image.Exif()
            tags[274] = 6
            image.save(path, exif=tags)
            size = size[::-1]  # annotation geometry after native EXIF orientation
        else:
            image.save(path)
        entries.append({"type": "image_url", "image_url": {"url": str(path), "image_wh": list(size)}})
    entries[-1] = copy.deepcopy(entries[0])  # repeat the same source image
    prompt = ("<IMG_CONTEXT>" * len(entries) if template == "qwen3-vl" else "") + "Describe these images."
    return {"messages": [
        {"role": "user", "content": entries + [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": "Three colored images."},
    ]}


@pytest.mark.parametrize("threads", ["0", "2"])
@pytest.mark.parametrize("exif", [False, True])
def test_native_eager_lazy_parity(tokenize, tmp_path, monkeypatch, threads, exif):
    data = sample(tmp_path, tokenize.chat_template_name, exif=exif)
    monkeypatch.setenv("XTUNER_LAZY_MEDIA", "0")
    eager = tokenize(copy.deepcopy(data))
    monkeypatch.setenv("XTUNER_LAZY_MEDIA", "1")
    monkeypatch.setenv("XTUNER_LAZY_MEDIA_THREADS", threads)
    lazy = tokenize(copy.deepcopy(data))
    assert "media_plan" in lazy and "pixel_values" not in lazy
    compare(eager, lazy)
    pixels, grid = materialize_image_plan(lazy["media_plan"])
    assert pixels.dtype == torch.float32
    assert torch.equal(pixels, eager["pixel_values"])
    assert torch.equal(grid, eager["image_grid_thw"])
    from types import SimpleNamespace
    from xtuner.v1.train.lazy_media_prefetch import kickoff_lazy_media, take_media_future
    future_plan = copy.deepcopy(lazy["media_plan"])
    kickoff_lazy_media([{"seq_ctx": SimpleNamespace(media_plan=future_plan)}], decode_here=True)
    assert torch.equal(take_media_future(future_plan).result()[0], pixels)
    kwargs = dict(pack_max_length=8192, padding_token_idx=0)
    reference = qwen3_vl_sft_collator([[copy.deepcopy(eager), copy.deepcopy(eager)]], **kwargs)[0]
    packed = qwen3_vl_sft_collator([[copy.deepcopy(lazy), copy.deepcopy(lazy)]], **kwargs)[0]
    ctx, ref = packed["seq_ctx"], reference["seq_ctx"]
    assert torch.equal(packed["shifted_labels"], reference["shifted_labels"])
    for key in ("input_ids", "position_ids", "image_grid_thw"):
        a, b = getattr(ctx, key), getattr(ref, key)
        assert (a is None and b is None) or torch.equal(a, b), key
    assert torch.equal(materialize_image_plan(ctx.media_plan)[0], ref.pixel_values)
    # Mixed eager fallback keeps original order, and never mutates caller records.
    mixed = qwen3_vl_sft_collator([[copy.deepcopy(eager), copy.deepcopy(lazy)]], **kwargs)[0]
    assert mixed["seq_ctx"].media_plan is None
    assert torch.equal(mixed["seq_ctx"].pixel_values, ref.pixel_values)
    assert "media_plan" in lazy
    merged = SequenceContext.cat([ctx, ctx])
    assert merged.media_plan["processor_family"] == "qwen"
    assert torch.equal(materialize_image_plan(merged.media_plan)[0], torch.cat([ref.pixel_values] * 2))
    broken = copy.deepcopy(lazy["media_plan"])
    broken["images"][0]["grid"][1] += 2
    with pytest.raises(RuntimeError, match="processed grid"):
        materialize_image_plan(broken)


def test_pure_text_unchanged(tokenize, monkeypatch):
    data = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}
    monkeypatch.setenv("XTUNER_LAZY_MEDIA", "0")
    eager = tokenize(copy.deepcopy(data))
    monkeypatch.setenv("XTUNER_LAZY_MEDIA", "1")
    lazy = tokenize(copy.deepcopy(data))
    compare(eager, lazy)
    assert "media_plan" not in lazy


def test_missing_geometry_eager_fallback(tokenize, tmp_path, monkeypatch):
    data = sample(tmp_path, tokenize.chat_template_name)
    for entry in data["messages"][0]["content"]:
        if "image_url" in entry:
            entry["image_url"].pop("image_wh")
    monkeypatch.setenv("XTUNER_LAZY_MEDIA", "0")
    eager = tokenize(copy.deepcopy(data))
    monkeypatch.setenv("XTUNER_LAZY_MEDIA", "1")
    fallback = tokenize(copy.deepcopy(data))
    compare(eager, fallback)
    assert "media_plan" not in fallback
    assert torch.equal(eager["pixel_values"], fallback["pixel_values"])
