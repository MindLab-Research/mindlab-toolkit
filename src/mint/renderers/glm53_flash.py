"""GLM-5.3-Flash renderer: GLM-5.3 text plus official Glm5Next image expansion.

Template placeholders come from ``zai-org/GLM-5.3-Flash`` ``emit_image``.
Image token counts come from Transformers ``Glm5NextImageProcessor``
(``get_number_of_image_patches`` / ``merge_size``), matching
``Glm5NextProcessor.replace_image_token``. This module does not call
Cookbook ``image_to_chunk``.
"""

from __future__ import annotations

import base64
import io
import os
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import (
    Message,
    RenderContext,
    TrainOnWhat,
    register_renderer,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

from mint.renderers.glm52 import GLM53Renderer

GLM53_FLASH_RENDERER = "MindLab/glm53_flash"
# Same kind of dependency as the GLM tokenizer: a Hugging Face repo id, not a
# path checked into this package. ``from_pretrained`` reads the Hub cache.
GLM53_FLASH_MODEL_ID = "zai-org/GLM-5.3-Flash"

_BEGIN_IMAGE = "<|begin_of_image|>"
_IMAGE = "<|image|>"
_END_IMAGE = "<|end_of_image|>"
_VIDEO_PLACEHOLDER = "<|begin_of_video|><|video|><|end_of_video|>"
_AUDIO_PLACEHOLDER = "<|begin_of_audio|><|end_of_audio|>"
# Set by render_message so each image record names the message that held it.
_MESSAGE_INDEX: ContextVar[int | None] = ContextVar(
    "glm53_flash_message_index", default=None
)


def load_glm53_flash_image_processor(model_name: str | None = None) -> Any:
    """Load ``Glm5NextImageProcessor`` the same way the GLM tokenizer is loaded.

    ``model_name`` is a Hugging Face repo id or a local snapshot directory.
    The default is :data:`GLM53_FLASH_MODEL_ID`. Set ``GLM53_FLASH_PROCESSOR``
    to override that default without passing an argument.
    """
    from transformers.models.glm5_next.image_processing_glm5_next import (
        Glm5NextImageProcessor,
    )

    source = model_name or os.environ.get("GLM53_FLASH_PROCESSOR") or GLM53_FLASH_MODEL_ID
    return Glm5NextImageProcessor.from_pretrained(source)


def _resolve_image_processor(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        return load_glm53_flash_image_processor(raw)
    image_processor = getattr(raw, "image_processor", raw)
    if not hasattr(image_processor, "get_number_of_image_patches"):
        raise RendererError(
            "GLM-5.3-Flash requires official Glm5NextImageProcessor "
            "(get_number_of_image_patches); do not pass a Qwen/Cookbook processor"
        )
    return image_processor


def _image_source(part: Mapping[str, Any]) -> Any:
    part_type = part.get("type")
    if part_type == "image":
        if "image" in part:
            return part["image"]
        raw = part.get("image_url", part.get("url"))
        if isinstance(raw, Mapping):
            raw = raw.get("url")
        if raw is not None:
            return raw
    if part_type == "image_url":
        raw = part.get("image_url", part.get("url"))
        if isinstance(raw, Mapping):
            raw = raw.get("url")
        if raw is not None:
            return raw
    raise RendererError(
        "GLM-5.3-Flash image part is missing image data "
        "(need PIL/path/data URI on 'image' or image_url.url)"
    )


def _as_pil(source: Any) -> Any:
    from PIL import Image

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if not isinstance(source, str):
        raise RendererError(
            "GLM-5.3-Flash image data must be a PIL image, path, or data URI"
        )
    if source.startswith("data:"):
        _, _, encoded = source.partition(",")
        try:
            payload = base64.b64decode(encoded, validate=True)
            return Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception as exc:
            raise RendererError(
                "GLM-5.3-Flash data URI is not a readable image"
            ) from exc
    if source.startswith(("http://", "https://")):
        raise RendererError(
            "GLM-5.3-Flash renderer does not fetch http(s) image URLs; "
            "pass a PIL image, local path, or data URI"
        )
    try:
        return Image.open(source).convert("RGB")
    except Exception as exc:
        raise RendererError(
            f"GLM-5.3-Flash cannot read image path {source!r}"
        ) from exc


def _expanded_image_placeholder(
    image_processor: Any, image: Any
) -> tuple[str, int, int]:
    """Return placeholder text, token count, and patch count.

    Official ``get_number_of_image_patches(height, width, images_kwargs)``
    takes height then width. ``None`` keeps the processor's own resize
    defaults; an empty dict would be the same fallback inside that method.
    """
    width, height = image.size
    patches = int(image_processor.get_number_of_image_patches(height, width, None))
    merge = int(image_processor.merge_size) ** 2
    if merge <= 0:
        raise RendererError("Glm5NextImageProcessor.merge_size must be positive")
    if patches % merge != 0:
        raise RendererError(
            f"official Flash patch count {patches} is not divisible by "
            f"merge_size**2={merge}"
        )
    num_tokens = patches // merge
    if num_tokens < 1:
        raise RendererError(
            f"official Flash processor produced no image tokens "
            f"(patches={patches}, merge={merge})"
        )
    return f"{_BEGIN_IMAGE}{_IMAGE * num_tokens}{_END_IMAGE}", num_tokens, patches


def _pixel_count(value: Any) -> int:
    if isinstance(value, (str, bytes)) or value is None:
        raise RendererError(
            "official Flash preprocess pixel_values must be a non-empty tensor"
        )
    if hasattr(value, "numel"):
        count = int(value.numel())
    elif hasattr(value, "shape"):
        count = 1
        shape = tuple(value.shape)
        if not shape:
            raise RendererError(
                "official Flash preprocess pixel_values must have a shape"
            )
        for dim in shape:
            count *= int(dim)
    else:
        raise RendererError(
            "official Flash preprocess pixel_values must be a non-empty tensor"
        )
    if count <= 0:
        raise RendererError(
            "official Flash preprocess pixel_values must be non-empty"
        )
    return count


def _image_grid_rows(grid: Any) -> list[tuple[int, int, int]]:
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if isinstance(grid, (str, bytes)) or not isinstance(grid, (list, tuple)) or not grid:
        raise RendererError(
            "official Flash image_grid_thw must be rows of [temporal, height, width]"
        )
    first = grid[0]
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        grid = [grid]
    rows: list[tuple[int, int, int]] = []
    for row in grid:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise RendererError(
                "official Flash image_grid_thw rows must be [temporal, height, width]"
            )
        temporal, height, width = (int(row[0]), int(row[1]), int(row[2]))
        if temporal < 1 or height < 1 or width < 1:
            raise RendererError(
                "official Flash image_grid_thw entries must be positive"
            )
        rows.append((temporal, height, width))
    return rows


def _preprocess_image(image_processor: Any, image: Any, patches: int) -> Any:
    """Run official Glm5Next preprocess and check it matches ``patches``."""
    try:
        features = image_processor(images=image, return_tensors="pt")
    except Exception as exc:
        raise RendererError(
            "official Glm5NextImageProcessor failed to produce pixel_values"
        ) from exc
    if not isinstance(features, Mapping) or "pixel_values" not in features:
        raise RendererError("official Flash preprocess must return pixel_values")
    if "image_grid_thw" not in features:
        raise RendererError("official Flash preprocess must return image_grid_thw")
    _pixel_count(features["pixel_values"])
    rows = _image_grid_rows(features["image_grid_thw"])
    if len(rows) != 1:
        raise RendererError(
            "official Flash preprocess of one image must return one image_grid_thw row"
        )
    temporal, height, width = rows[0]
    if temporal * height * width != patches:
        raise RendererError(
            "official Flash image_grid_thw does not match "
            f"get_number_of_image_patches ({patches})"
        )
    return features


class GLM53FlashRenderer(GLM53Renderer):
    """GLM-5.3 text renderer; images use official Glm5Next expansion."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        image_processor: Any = None,
        reasoning_effort: str = "max",
        enable_thinking: bool = True,
        supervise_reasoning: bool = True,
    ):
        super().__init__(
            tokenizer,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            enable_thinking=enable_thinking,
            supervise_reasoning=supervise_reasoning,
        )
        self.image_processor = _resolve_image_processor(image_processor)
        self.mm_inputs: list[dict[str, Any]] = []
        self._image_cache: dict[int, Any] = {}

    def _reset_multimodal(self) -> None:
        self.mm_inputs = []
        self._image_cache = {}

    def render_message(self, message: Message, ctx: RenderContext):
        token = _MESSAGE_INDEX.set(ctx.idx)
        try:
            return super().render_message(message, ctx)
        finally:
            _MESSAGE_INDEX.reset(token)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: str = "assistant",
        prefill: str | None = None,
    ):
        self._reset_multimodal()
        try:
            return super().build_generation_prompt(messages, role=role, prefill=prefill)
        except Exception:
            self._reset_multimodal()
            raise

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ):
        self._reset_multimodal()
        try:
            return super().build_supervised_example(messages, train_on_what)
        except Exception:
            self._reset_multimodal()
            raise

    def _media_placeholder(self, kind: str, part: Mapping[str, Any]) -> str:
        if kind == "video":
            return _VIDEO_PLACEHOLDER
        if kind == "audio":
            return _AUDIO_PLACEHOLDER
        if kind != "image":
            raise RendererError(f"unsupported GLM-5.3-Flash media type: {kind!r}")
        if self.image_processor is None:
            raise RendererError(
                "GLM-5.3-Flash image parts require official Glm5NextImageProcessor"
            )
        source = _image_source(part)
        # convert("RGB") returns a new object, so cache the caller's image.
        cache_key = source if isinstance(source, str) else id(source)
        cached = self._image_cache.get(cache_key)
        if cached is None:
            image = _as_pil(source)
            text, num_tokens, patches = _expanded_image_placeholder(
                self.image_processor, image
            )
            features = _preprocess_image(self.image_processor, image, patches)
            cached = (image.size, text, num_tokens, features)
            self._image_cache[cache_key] = cached
        size, text, num_tokens, features = cached
        # One record per occurrence, in token order. The same PIL object is
        # preprocessed once; later occurrences reuse that feature batch.
        self.mm_inputs.append(
            {
                "message_index": _MESSAGE_INDEX.get(),
                "size": size,
                "num_image_tokens": num_tokens,
                "features": features,
            }
        )
        return text


def register_glm53_flash_renderer() -> None:
    register_renderer(
        GLM53_FLASH_RENDERER,
        lambda tokenizer, image_processor=None: GLM53FlashRenderer(
            tokenizer, image_processor=image_processor
        ),
    )


register_glm53_flash_renderer()


__all__ = [
    "GLM53_FLASH_MODEL_ID",
    "GLM53_FLASH_RENDERER",
    "GLM53FlashRenderer",
    "load_glm53_flash_image_processor",
    "register_glm53_flash_renderer",
]
