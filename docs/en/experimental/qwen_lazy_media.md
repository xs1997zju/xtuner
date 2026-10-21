# Experimental Qwen lazy images

This change extends the image-only lazy path from InternLM/xtuner#2201 to Qwen3-VL and Qwen3.5. The native tokenizer builds text, labels, positions and predicted grids as before; decoding and image preprocessing can run at consumption time or through the existing one-batch lookahead. Processed grids are checked against the prediction.

The review base is PR #2201 at `b62fdfbdcd1cd647c853038fada8b7618035c6d9`. It includes that PR's GLM support and lifecycle behavior. This incremental change does not use the unified media runtime.

## Switches

All switches are inherited from #2201. With `XTUNER_LAZY_MEDIA` unset, Qwen keeps its native eager path.

| Mode | `XTUNER_LAZY_MEDIA` | `XTUNER_LAZY_MEDIA_THREADS` | `XTUNER_LAZY_MEDIA_SHARED` | `XTUNER_LAZY_MEDIA_PREFETCH` |
| --- | --- | --- | --- | --- |
| Native eager | 0 | — | 0 | 0 |
| Local lazy with lookahead | 1 | 2 | 0 | 1 |
| SP leader lazy with lookahead | 1 | 2 | 1 | 1 |

Use the existing native Qwen model configuration and Hugging Face checkpoint loader. No model depth, attention, MTP, DeepStack, vision SP partition or parameter-loading changes are included.

## Data and transport

The serial setting (`THREADS=0` or `1`) preserves native per-source-sample processor calls. Larger values use ordered per-image processing, with parity checked by real-processor tests. Pixels stay FP32 and grids stay integer. Missing or invalid image geometry uses the eager path; mixed eager/lazy packs materialize in the collator while preserving order. Videos keep their native path.

Shared mode decodes once on the SP leader and broadcasts **complete GPU pixels** to each rank before the native vision code performs its SP partition. It avoids a full follower GPU-to-CPU round trip, but may increase GPU memory use. This is not pixel scatter. Background materialization stays in CPU threads; collectives execute on the training thread.

## Validation and limits

Real tokenizer/processor CPU tests cover both templates, serial and two-thread materialization, repeated images, EXIF orientation, pixels/grids/tokens/labels/positions, collator ordering and mixed fallback, context concatenation, and grid-mismatch rejection. Run these tests with `QWEN3_VL_PATH` and `QWEN3_5_MOE_PATH` set to the respective local checkpoint directories.

GPU/NCCL training, parameter updates, memory cost, and performance remain unqualified. CPU parity with the native path does not establish that every native model behavior matches Hugging Face. Existing #2201 resume, epoch-transition, early-exit cleanup and rank-plan-agreement limitations also apply. A fresh work directory and a non-resumed, image-only window are required for the initial diagnostic comparison. No speedup or root-cause claim is made.
