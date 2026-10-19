"""Stdlib contracts for MTP construction; no framework import or tensor execution.

Run standalone to avoid repository pytest conftest framework imports:
    python tests/model/test_qwen35_mtp_attention.py
"""

import ast
import copy
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MOE_SOURCE = ROOT / "xtuner/v1/model/moe/moe.py"
QWEN_SOURCE = ROOT / "xtuner/v1/model/moe/qwen3_5_text.py"


def extract_class(source, class_name, method_names, bases=(), namespace=None):
    tree = ast.parse(source.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = []
    for name in method_names:
        method = copy.deepcopy(
            next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
        )
        method.decorator_list = []
        methods.append(method)
    extracted = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id=base.__name__, ctx=ast.Load()) for base in bases],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), extracted],
        type_ignores=[],
    )
    scope = {base.__name__: base for base in bases}
    scope.update(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[class_name]


BaseMoE = extract_class(MOE_SOURCE, "MoE", ["_get_mtp_attention_spec", "build_mtp_block"])
QwenMoE = extract_class(
    QWEN_SOURCE,
    "Qwen3_5_VLTextMoE",
    ["_get_mtp_attention_spec", "to_hf_key_list"],
    bases=(BaseMoE,),
    namespace={"re": re},
)
QwenConfig = extract_class(QWEN_SOURCE, "Qwen3_5_VLTextMoEConfig", ["layers_type"])


def make_config(depth=5, shared=True):
    config = SimpleNamespace(
        num_hidden_layers=depth,
        attention=object(),
        linear_attention=object(),
        mtp_config=SimpleNamespace(num_layers=4, share_weights=shared, loss_scaling_factor=1.0, loss_type="ce"),
    )
    config.layers_type = QwenConfig.layers_type(config)
    for name in (
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "mlp_bias",
        "gate_bias",
        "moe_bias",
        "hidden_act",
        "rms_norm_eps",
        "rms_norm_type",
        "num_experts_per_tok",
        "n_routed_experts",
        "n_shared_experts",
        "with_shared_expert_gate",
        "hidden_factor",
        "rope_scaling_cfg",
        "generate_config",
        "router",
        "router_compute_dtype",
        "moe_act_fn_cfg",
        "float8_cfg",
        "dispatcher",
    ):
        setattr(config, name, object())
    config.n_routed_experts = 2
    return config


def make_model(model_class):
    model = model_class()
    model.ep_mesh = object()
    model.expert_tp_mesh = object()
    model.ep_tp_mesh = object()
    model.moe_decoder_layer_cls = lambda **kwargs: SimpleNamespace(**kwargs)
    model.mtp_layer_cls = lambda **kwargs: SimpleNamespace(**kwargs)
    model.mtp_block_cls = lambda **kwargs: SimpleNamespace(**kwargs)
    return model


class TestMTPAttentionSpec(unittest.TestCase):
    def test_default_full_sliding_and_linear_selection_unchanged(self):
        model = BaseMoE()
        for layer_type in ("full_attention", "sliding_attention", "linear_attention"):
            with self.subTest(layer_type=layer_type):
                config = make_config()
                config.layers_type[-1] = layer_type
                selected_type, selected_config = model._get_mtp_attention_spec(config)
                self.assertEqual(selected_type, layer_type)
                self.assertIs(
                    selected_config, config.linear_attention if layer_type == "linear_attention" else config.attention
                )

    def test_default_missing_linear_config_rejected(self):
        config = make_config()
        config.linear_attention = None
        with self.assertRaisesRegex(AssertionError, "linear_attention config must be provided"):
            BaseMoE()._get_mtp_attention_spec(config)

    def test_default_unknown_layer_rejected(self):
        config = make_config()
        config.layers_type[-1] = "unsupported"
        with self.assertRaisesRegex(ValueError, "Unsupported layer type unsupported"):
            BaseMoE()._get_mtp_attention_spec(config)

    def test_qwen_five_and_forty_ordinary_patterns_are_not_mutated(self):
        for depth in (5, 40):
            with self.subTest(depth=depth):
                config = make_config(depth)
                before = vars(config).copy()
                layers_before = config.layers_type.copy()
                selected_type, selected_config = QwenMoE()._get_mtp_attention_spec(config)
                self.assertEqual(selected_type, "full_attention")
                self.assertIs(selected_config, config.attention)
                self.assertEqual(vars(config), before)
                self.assertEqual(config.layers_type, layers_before)
                self.assertEqual(
                    config.layers_type[:5], ["linear_attention"] * 3 + ["full_attention", "linear_attention"]
                )
                self.assertEqual(config.layers_type[-1], "linear_attention" if depth == 5 else "full_attention")

    def test_shared_builder_retains_one_physical_layer_and_all_passthroughs(self):
        for depth in (5, 40):
            with self.subTest(depth=depth):
                config = make_config(depth)
                before = vars(config).copy()
                layers_before = config.layers_type.copy()
                model = make_model(QwenMoE)
                block = model.build_mtp_block(config)
                self.assertIs(block.mtp_config, config.mtp_config)
                self.assertEqual(block.mtp_config.num_layers, 4)
                self.assertTrue(block.mtp_config.share_weights)
                self.assertEqual(block.mtp_config.loss_type, "ce")
                self.assertEqual(block.mtp_config.loss_scaling_factor, 1.0)
                self.assertEqual(len(block.mtp_layers), 1)
                layer = block.mtp_layers[0]
                decoder = layer.decoder_layer
                self.assertEqual(decoder.layer_type, "full_attention")
                self.assertIs(decoder.attention_config, config.attention)
                self.assertEqual(decoder.layer_idx, depth)
                for name, value in vars(decoder).items():
                    if name in ("layer_type", "attention_config", "layer_idx"):
                        continue
                    expected = (
                        getattr(model, name)
                        if name in ("ep_mesh", "expert_tp_mesh", "ep_tp_mesh")
                        else getattr(config, "router" if name == "router_config" else name)
                    )
                    self.assertIs(value, expected, name)
                for name in ("hidden_size", "rms_norm_eps", "rms_norm_type", "float8_cfg"):
                    self.assertIs(getattr(layer, name), getattr(config, name), name)
                self.assertEqual(vars(config), before)
                self.assertEqual(config.layers_type, layers_before)

    def test_nonshared_builder_keeps_physical_count_and_indices(self):
        config = make_config(shared=False)
        block = make_model(QwenMoE).build_mtp_block(config)
        self.assertEqual(len(block.mtp_layers), 4)
        self.assertEqual([layer.decoder_layer.layer_idx for layer in block.mtp_layers], [5, 6, 7, 8])
        self.assertTrue(all(layer.decoder_layer.layer_type == "full_attention" for layer in block.mtp_layers))

    def test_generic_builder_still_uses_last_ordinary_attention(self):
        for layer_type in ("full_attention", "sliding_attention", "linear_attention"):
            with self.subTest(layer_type=layer_type):
                config = make_config()
                config.layers_type[-1] = layer_type
                block = make_model(BaseMoE).build_mtp_block(config)
                decoder = block.mtp_layers[0].decoder_layer
                self.assertEqual(decoder.layer_type, layer_type)
                self.assertIs(
                    decoder.attention_config,
                    config.linear_attention if layer_type == "linear_attention" else config.attention,
                )

    def test_missing_mtp_config_still_rejected(self):
        config = make_config()
        config.mtp_config = None
        with self.assertRaisesRegex(AssertionError, "mtp_config must be provided"):
            make_model(QwenMoE).build_mtp_block(config)

    def test_original_full_attention_mtp_hf_mapping_and_ordinary_linear_mapping(self):
        model = make_model(QwenMoE)
        model.config = make_config()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            with self.subTest(name=name):
                self.assertEqual(
                    model.to_hf_key_list(f"mtp_block.layers.0.decoder_layer.self_attn.{name}.weight"),
                    [f"mtp.layers.0.self_attn.{name}.weight"],
                )
        self.assertEqual(
            model.to_hf_key_list("layers.4.self_attn.in_proj_qkv.weight"),
            ["model.language_model.layers.4.linear_attn.in_proj_qkv.weight"],
        )
        self.assertEqual(model.to_hf_key_list("mtp_block.layers.0.eh_proj.weight"), ["mtp.fc.weight"])
        self.assertEqual(model.to_hf_key_list("mtp_block.layers.0.enorm.weight"), ["mtp.pre_fc_norm_embedding.weight"])
        self.assertEqual(
            model.to_hf_key_list("mtp_block.layers.0.decoder_layer.experts.fused_w1w3.weight"),
            [
                f"mtp.layers.0.mlp.experts.{index}.{projection}.weight"
                for index in range(2)
                for projection in ("gate_proj", "up_proj")
            ],
        )


if __name__ == "__main__":
    unittest.main()
