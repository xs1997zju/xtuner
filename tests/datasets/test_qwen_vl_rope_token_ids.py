from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from xtuner.v1.datasets.mllm_tokenize_fn import qwen3_vl_tokenize_fn as qwen_tokenize


class TestQwenVLRopeTokenIds(TestCase):
    families = (
        ("qwen3-vl", 151655, 151656, 151652, 151653),
        ("qwen3.5-vl", 248056, 248057, 248053, 248054),
    )

    def _check_path(self, family, modality, enable_3d_rope):
        template, image_token_id, video_token_id, vision_start_id, vision_end_id = family
        tokenize_fn = object.__new__(qwen_tokenize.Qwen3VLTokenizeFunction)
        tokenize_fn.tokenizer = Mock()
        tokenize_fn.chat_template = Mock()
        tokenize_fn.chat_template_name = template
        tokenize_fn.img_context_token_id = image_token_id
        tokenize_fn.video_context_token_id = video_token_id
        tokenize_fn.img_start_token_id = vision_start_id
        tokenize_fn.img_end_token_id = vision_end_id
        tokenize_fn.enable_3d_rope = enable_3d_rope
        tokenize_fn.add_vision_id = False
        tokenize_fn.add_generation_prompt = False
        tokenize_fn.enable_thinking = None
        tokenize_fn.max_length = None
        tokenize_fn.data_name = "synthetic"
        tokenize_fn.oss_loader = None
        tokenize_fn._image_path = ["image.png"]
        tokenize_fn._video_path = ["video.mp4"]
        tokenize_fn._video_wh_list = [[64, 64]]
        tokenize_fn._video_extra_info_list = []
        tokenize_fn.calc_frame_info = Mock(return_value=([2], [], []))
        pixels = torch.zeros(16, 1536)
        grid = torch.tensor([[1, 4, 4]])
        tokenize_fn.image_processor = Mock(merge_size=2)
        tokenize_fn.merge_length = tokenize_fn.image_processor.merge_size**2
        tokenize_fn.image_processor.preprocess.return_value = {"pixel_values": pixels, "image_grid_thw": grid}
        tokenize_fn.video_processor = SimpleNamespace(
            merge_size=2,
            patch_size=16,
            temporal_patch_size=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            size={"shortest_edge": 4096, "longest_edge": 65536},
            preprocess=Mock(return_value={"pixel_values_videos": pixels, "video_grid_thw": grid}),
        )
        if modality == "text":
            input_ids = [10, 11]
            tokenized_ids = input_ids.copy()
            expected_positions = torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]])
        else:
            media_token_id = image_token_id if modality == "image" else video_token_id
            input_ids = [10, vision_start_id, *([media_token_id] * 4), vision_end_id, 11]
            tokenized_ids = (
                [10, vision_start_id, media_token_id, vision_end_id, 11]
                if template == "qwen3.5-vl"
                else input_ids.copy()
            )
            expected_positions = torch.tensor(
                [[[0, 1, 2, 2, 2, 2, 4, 5]], [[0, 1, 2, 2, 3, 3, 4, 5]], [[0, 1, 2, 3, 2, 3, 4, 5]]]
            )
        messages = Mock(messages=[SimpleNamespace(role="user")])
        messages.tokenize.return_value = {"input_ids": tokenized_ids, "labels": [-100] * len(tokenized_ids)}
        record = {"messages": [{"role": "user", "content": "synthetic"}]}
        methods = {
            "text": tokenize_fn.pure_text_get_item,
            "image": tokenize_fn.multi_modal_get_item,
            "video": tokenize_fn.video_get_item,
        }
        with (
            patch.multiple(
                qwen_tokenize,
                ChatMessages=Mock(return_value=messages),
                Qwen35ChatMessages=Mock(return_value=messages),
                load_image=Mock(return_value=object()),
                apply_exif_orientation=Mock(side_effect=lambda image: image),
                replace_image_token=Mock(),
                replace_video_token=Mock(),
                replace_video_timestamps_and_num_frame=Mock(),
                read_qwen3_vl_video=Mock(return_value=([torch.zeros(3, 64, 64)] * 2, [0, 1], None)),
            ),
            patch.object(qwen_tokenize, "get_rope_index_3", wraps=qwen_tokenize.get_rope_index_3) as rope,
        ):
            result = methods[modality](record)
        self.assertEqual(result["input_ids"], input_ids)
        self.assertEqual(result["labels"], [-100] * len(input_ids))
        self.assertEqual(result["num_tokens"], len(input_ids))
        self.assertEqual(result["num_img_tokens"], [0] if modality == "text" else [16])
        self.assertEqual(result["num_imgs"], [{"text": 0, "image": 1, "video": 2}[modality]])
        if modality == "text":
            self.assertNotIn("pixel_values", result)
            self.assertNotIn("image_grid_thw", result)
        else:
            self.assertTrue(torch.equal(result["pixel_values"], pixels))
            self.assertEqual(result["pixel_values"].dtype, pixels.dtype)
            self.assertTrue(torch.equal(result["image_grid_thw"], grid))
            self.assertEqual(result["image_grid_thw"].dtype, grid.dtype)
            self.assertNotIn("pixel_values_videos", result)
            self.assertNotIn("video_grid_thw", result)
        if enable_3d_rope:
            rope.assert_called_once()
            self.assertEqual(rope.call_args.kwargs["image_token_id"], image_token_id)
            self.assertEqual(rope.call_args.kwargs["video_token_id"], video_token_id)
            self.assertEqual(rope.call_args.kwargs["vision_start_token_id"], vision_start_id)
            self.assertTrue(torch.equal(result["position_ids"], expected_positions))
            self.assertEqual(result["position_ids"].dtype, torch.long)
        else:
            rope.assert_not_called()
            self.assertIsNone(result["position_ids"])

    def test_family_token_ids_are_used_for_all_rope_paths(self):
        for family in self.families:
            for modality in ("text", "image", "video"):
                with self.subTest(family=family[0], modality=modality):
                    self._check_path(family, modality, enable_3d_rope=True)

    def test_disabled_rope_preserves_native_inputs(self):
        for family in self.families:
            for modality in ("text", "image", "video"):
                with self.subTest(family=family[0], modality=modality):
                    self._check_path(family, modality, enable_3d_rope=False)
