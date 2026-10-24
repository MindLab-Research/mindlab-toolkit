"""Flash inherits GLM-5.2 text; images use official Glm5Next expansion."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

pytest.importorskip(
    "tinker_cookbook.renderers",
    reason="install mindlab-toolkit[test] to run GLM renderer tests",
)
pytest.importorskip("PIL")

from PIL import Image
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import Message, get_renderer

from mint.renderers import (
    GLM52Renderer,
    GLM53_FLASH_MODEL_ID,
    GLM53_FLASH_RENDERER,
    GLM53FlashRenderer,
    load_glm53_flash_image_processor,
)
from mint.renderers.glm52 import _TOOL_CONTEXT_KEY


class FlashTokenizer:
    """Character tokenizer with GLM role tokens and official media wrappers."""

    name_or_path = "zai-org/GLM-5.3-Flash-test-tokenizer"
    _specials: ClassVar[tuple[str, ...]] = (
        "<|endoftext|>",
        "<|observation|>",
        "<|assistant|>",
        "<|system|>",
        "<|user|>",
        "[gMASK]",
        "<sop>",
        "<|begin_of_image|>",
        "<|image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|video|>",
        "<|end_of_video|>",
        "<|begin_of_audio|>",
        "<|end_of_audio|>",
    )
    _token_by_text: ClassVar[dict[str, int]] = {
        text: 0x110000 + index for index, text in enumerate(_specials)
    }
    _text_by_token: ClassVar[dict[int, str]] = {
        token: text for text, token in _token_by_text.items()
    }

    def encode(
        self, text: str, add_special_tokens: bool = False, **_: Any
    ) -> list[int]:
        del add_special_tokens
        result: list[int] = []
        position = 0
        while position < len(text):
            special = next(
                (item for item in self._specials if text.startswith(item, position)),
                None,
            )
            if special is None:
                result.append(ord(text[position]))
                position += 1
            else:
                result.append(self._token_by_text[special])
                position += len(special)
        return result

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self._text_by_token[token] if token in self._text_by_token else chr(token)
            for token in tokens
        )


class _StubPixels:
    shape = (16,)

    def numel(self) -> int:
        return 16


class StubImageProcessor:
    merge_size = 2
    calls = 0

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict[str, Any] | None = None
    ) -> int:
        del height, width, images_kwargs
        return 16

    def __call__(self, images: Any, return_tensors: str = "pt") -> dict[str, Any]:
        del images, return_tensors
        self.calls += 1
        return {
            "pixel_values": _StubPixels(),
            "image_grid_thw": [[1, 4, 4]],
        }


@pytest.fixture
def tokenizer() -> FlashTokenizer:
    return FlashTokenizer()


def _decode(tokenizer: FlashTokenizer, model_input: Any) -> str:
    return tokenizer.decode(model_input.to_ints())


def _rgb(size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, (10, 20, 30))


def test_text_only_matches_glm52(tokenizer: FlashTokenizer):
    messages = [
        Message(role="user", content="question"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "reason"},
                {"type": "text", "text": "answer"},
            ],
        ),
    ]
    glm = GLM52Renderer(tokenizer)
    flash = GLM53FlashRenderer(tokenizer)
    glm_ids, glm_w = glm.build_supervised_example(list(messages))
    flash_ids, flash_w = flash.build_supervised_example(list(messages))
    assert flash_ids.to_ints() == glm_ids.to_ints()
    assert flash_w.tolist() == glm_w.tolist()
    assert _decode(tokenizer, flash.build_generation_prompt(messages[:1])) == _decode(
        tokenizer, glm.build_generation_prompt(messages[:1])
    )


def test_flash_registered_name(tokenizer: FlashTokenizer):
    renderer = get_renderer(
        GLM53_FLASH_RENDERER, tokenizer, model_name=tokenizer.name_or_path
    )
    assert isinstance(renderer, GLM53FlashRenderer)


def test_image_requires_official_processor(tokenizer: FlashTokenizer):
    renderer = GLM53FlashRenderer(tokenizer)
    with pytest.raises(RendererError, match="Glm5NextImageProcessor"):
        renderer.build_generation_prompt(
            [
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "see "},
                        {"type": "image", "image": _rgb()},
                    ],
                )
            ]
        )


def test_image_expands_official_token_count(tokenizer: FlashTokenizer):
    renderer = GLM53FlashRenderer(tokenizer, image_processor=StubImageProcessor())
    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "see "},
                    {"type": "image", "image": _rgb()},
                ],
            )
        ]
    )
    decoded = _decode(tokenizer, prompt)
    expected = "<|begin_of_image|>" + "<|image|>" * 4 + "<|end_of_image|>"
    assert f"see {expected}" in decoded
    assert "<reminder>" not in decoded
    assert renderer.mm_inputs[0]["message_index"] == 0
    assert renderer.mm_inputs[0]["num_image_tokens"] == 4
    assert renderer.mm_inputs[0]["features"]["pixel_values"].numel() == 16


def test_image_url_uses_same_expansion(tokenizer: FlashTokenizer):
    renderer = GLM53FlashRenderer(tokenizer, image_processor=StubImageProcessor())
    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": _rgb()}},
                ],
            )
        ]
    )
    decoded = _decode(tokenizer, prompt)
    assert decoded.count("<|image|>") == 4


@pytest.mark.parametrize(
    ("part", "placeholder"),
    [
        ({"type": "video", "video": "https://example.test/c.mp4"},
         "<|begin_of_video|><|video|><|end_of_video|>"),
        ({"type": "audio", "audio": "https://example.test/d.wav"},
         "<|begin_of_audio|><|end_of_audio|>"),
    ],
)
def test_video_audio_stay_template_placeholders(
    tokenizer: FlashTokenizer, part: dict[str, Any], placeholder: str
):
    renderer = GLM53FlashRenderer(tokenizer)
    prompt = renderer.build_generation_prompt(
        [Message(role="user", content=[{"type": "text", "text": "see "}, part])]
    )
    assert f"see {placeholder}" in _decode(tokenizer, prompt)


def test_assistant_image_keeps_think_scaffold(tokenizer: FlashTokenizer):
    renderer = GLM53FlashRenderer(tokenizer, image_processor=StubImageProcessor())
    model_input, _ = renderer.build_supervised_example(
        [
            Message(role="user", content="q"),
            Message(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "look"},
                    {"type": "text", "text": "here "},
                    {"type": "image", "image": _rgb()},
                ],
            ),
        ]
    )
    text = _decode(tokenizer, model_input)
    expanded = "<|begin_of_image|>" + "<|image|>" * 4 + "<|end_of_image|>"
    assert f"<|assistant|><think>look</think>here {expanded}" in text


def test_glm52_still_uses_reminder_for_images(tokenizer: FlashTokenizer):
    renderer = GLM52Renderer(tokenizer)
    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "inspect "},
                    {"type": "image", "image": _rgb()},
                ],
            )
        ]
    )
    assert "inspect <reminder>You are unable to process this image" in _decode(
        tokenizer, prompt
    )


def _load_official_or_skip():
    """Official files come from the Hub id, the same way the GLM tokenizer does."""
    try:
        return load_glm53_flash_image_processor()
    except Exception as exc:
        pytest.skip(f"{GLM53_FLASH_MODEL_ID} image processor is not available: {exc}")


def test_official_processor_matches_replace_image_token(tokenizer: FlashTokenizer):
    image_processor = _load_official_or_skip()
    image = _rgb((448, 448))
    renderer = GLM53FlashRenderer(tokenizer, image_processor=image_processor)
    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "see "},
                    {"type": "image", "image": image},
                ],
            )
        ]
    )
    patches = image_processor.get_number_of_image_patches(448, 448, {})
    expected_n = patches // image_processor.merge_size**2
    decoded = _decode(tokenizer, prompt)
    assert decoded.count("<|image|>") == expected_n
    assert "<|begin_of_image|>" in decoded
    assert "<|end_of_image|>" in decoded
    record = renderer.mm_inputs[0]
    assert record["num_image_tokens"] == expected_n
    features = record["features"]
    assert "pixel_values" in features
    assert "image_grid_thw" in features
    grid = features["image_grid_thw"][0]
    prod = int(grid[0]) * int(grid[1]) * int(grid[2])
    assert prod // image_processor.merge_size**2 == expected_n


def _ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def test_one_case_matches_official_hf_processor():
    """One multimodal turn matches HF Glm5NextProcessor token-for-token.

    The supervised example keeps the GLM training boundary ``<|user|>`` that
    the chat template itself does not emit. Everything before that token,
    including expanded ``<|image|>`` ids and ``pixel_values``, matches.
    """
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    _load_official_or_skip()
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(GLM53_FLASH_MODEL_ID)
        processor = transformers.AutoProcessor.from_pretrained(GLM53_FLASH_MODEL_ID)
    except Exception as exc:
        pytest.skip(f"{GLM53_FLASH_MODEL_ID} is not available: {exc}")
    image = _rgb((320, 240))
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "这张图里有什么？"},
        ],
    }
    assistant = {
        "role": "assistant",
        "reasoning_content": "先看颜色和形状。",
        "content": "一块蓝色矩形。",
    }
    renderer = GLM53FlashRenderer(tokenizer, image_processor=processor.image_processor)

    ours_gen = renderer.build_generation_prompt([Message(**user)]).to_ints()
    hf_gen = processor.apply_chat_template(
        [dict(user)],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    assert ours_gen == _ids(hf_gen["input_ids"])
    features = renderer.mm_inputs[0]["features"]
    assert torch.equal(features["pixel_values"], hf_gen["pixel_values"])
    assert torch.equal(features["image_grid_thw"], hf_gen["image_grid_thw"])

    ours_full, _weights = renderer.build_supervised_example(
        [Message(**user), Message(**assistant)]
    )
    hf_full = processor.apply_chat_template(
        [dict(user), dict(assistant)],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    ours_ids = ours_full.to_ints()
    user_id = tokenizer.convert_tokens_to_ids("<|user|>")
    assert ours_ids[-1] == user_id
    assert ours_ids[:-1] == _ids(hf_full["input_ids"])


def test_bad_image_inputs_are_renderer_errors(tokenizer: FlashTokenizer):
    renderer = GLM53FlashRenderer(tokenizer, image_processor=StubImageProcessor())
    with pytest.raises(RendererError, match="data URI"):
        renderer.build_generation_prompt(
            [
                Message(
                    role="user",
                    content=[{"type": "image", "image": "data:image/png,@@@"}],
                )
            ]
        )
    assert renderer.mm_inputs == []
    with pytest.raises(RendererError, match="cannot read image path"):
        renderer.build_generation_prompt(
            [
                Message(
                    role="user",
                    content=[{"type": "image", "image": "/no/such/flash-image.png"}],
                )
            ]
        )
    assert renderer.mm_inputs == []


class _StringPixels(StubImageProcessor):
    def __call__(self, images: Any, return_tensors: str = "pt") -> dict[str, Any]:
        del images, return_tensors
        return {"pixel_values": "stub-pixels", "image_grid_thw": [[1, 4, 4]]}


class _IndivisiblePatches(StubImageProcessor):
    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict[str, Any] | None = None
    ) -> int:
        del height, width, images_kwargs
        return 17


def test_string_pixel_values_are_rejected(tokenizer: FlashTokenizer):
    renderer = GLM53FlashRenderer(tokenizer, image_processor=_StringPixels())
    with pytest.raises(RendererError, match="pixel_values"):
        renderer.build_generation_prompt(
            [Message(role="user", content=[{"type": "image", "image": _rgb()}])]
        )
    assert renderer.mm_inputs == []


def test_patch_count_must_divide_by_merge(tokenizer: FlashTokenizer):
    renderer = GLM53FlashRenderer(tokenizer, image_processor=_IndivisiblePatches())
    with pytest.raises(RendererError, match="not divisible"):
        renderer.build_generation_prompt(
            [Message(role="user", content=[{"type": "image", "image": _rgb()}])]
        )
    assert renderer.mm_inputs == []


def test_same_image_is_preprocessed_once(tokenizer: FlashTokenizer):
    processor = StubImageProcessor()
    renderer = GLM53FlashRenderer(tokenizer, image_processor=processor)
    image = _rgb()
    renderer.build_supervised_example(
        [
            Message(role="user", content=[{"type": "image", "image": image}]),
            Message(
                role="assistant",
                content=[{"type": "text", "text": "seen "}, {"type": "image", "image": image}],
            ),
        ]
    )
    assert processor.calls == 1
    assert [record["message_index"] for record in renderer.mm_inputs] == [0, 1]
    assert renderer.mm_inputs[0]["features"] is renderer.mm_inputs[1]["features"]


def test_tool_declaration_message_uses_flash_placeholder(tokenizer: FlashTokenizer):
    renderer = GLM53FlashRenderer(tokenizer, image_processor=StubImageProcessor())
    message = Message(
        role="system",
        content=[{"type": "text", "text": "see "}, {"type": "image", "image": _rgb()}],
    )
    message[_TOOL_CONTEXT_KEY] = [{"name": "lookup"}]
    decoded = _decode(tokenizer, renderer.build_generation_prompt([message]))
    assert "<|begin_of_image|>" in decoded
    assert "<reminder>" not in decoded
