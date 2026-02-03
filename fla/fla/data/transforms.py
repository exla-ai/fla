"""Data transforms for VLA training.

Provides transforms for:
- Image resizing and normalization
- State/action normalization
- Prompt tokenization
- Action padding for cross-embodiment
"""

import dataclasses
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from fla.shared.normalize import NormStats, normalize, unnormalize


class DataTransform(Protocol):
    """Protocol for data transforms."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]: ...


@dataclasses.dataclass
class TransformedDataset:
    """Apply transforms to a dataset."""

    dataset: Any
    transforms: Sequence[DataTransform]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.dataset[index]
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclasses.dataclass
class CompositeTransform:
    """Chain multiple transforms."""

    transforms: Sequence[DataTransform]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclasses.dataclass
class Normalize:
    """Normalize state and actions using precomputed statistics.

    Supports both z-score and quantile normalization.
    """

    stats: dict[str, NormStats]
    use_quantiles: bool = False
    keys: tuple[str, ...] = ("state", "actions")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.keys:
            if key in data and key in self.stats:
                data[key] = normalize(
                    data[key], self.stats[key], use_quantiles=self.use_quantiles
                )
        return data


@dataclasses.dataclass
class Unnormalize:
    """Unnormalize model outputs."""

    stats: dict[str, NormStats]
    use_quantiles: bool = False
    keys: tuple[str, ...] = ("actions",)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.keys:
            if key in data and key in self.stats:
                data[key] = unnormalize(
                    data[key], self.stats[key], use_quantiles=self.use_quantiles
                )
        return data


@dataclasses.dataclass
class ResizeImages:
    """Resize images to target resolution."""

    height: int = 224
    width: int = 224

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "image" not in data:
            return data

        from PIL import Image

        resized = {}
        for key, img in data["image"].items():
            # Handle various input formats
            if isinstance(img, np.ndarray):
                if img.dtype == np.uint8:
                    pil_img = Image.fromarray(img)
                else:
                    # Assume [-1, 1] float
                    pil_img = Image.fromarray(
                        ((img + 1) * 127.5).astype(np.uint8)
                    )
            else:
                pil_img = img

            # Resize with padding to maintain aspect ratio
            pil_img = pil_img.resize((self.width, self.height), Image.BILINEAR)

            # Convert back to numpy
            resized[key] = np.array(pil_img)

        data["image"] = resized
        return data


@dataclasses.dataclass
class NormalizeImages:
    """Normalize images from [0, 255] uint8 to [-1, 1] float32."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "image" not in data:
            return data

        normalized = {}
        for key, img in data["image"].items():
            if img.dtype == np.uint8:
                normalized[key] = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            else:
                normalized[key] = img.astype(np.float32)

        data["image"] = normalized
        return data


@dataclasses.dataclass
class PadActions:
    """Pad actions and state to target dimension for cross-embodiment.

    This enables training on multiple robot types with different action dimensions.
    For example:
    - Single-arm robots: 7 DOF
    - ALOHA bimanual: 14 DOF
    - Mobile ALOHA: 16 DOF

    Pad to the maximum dimension (default 14) with zeros.
    """

    target_dim: int = 14

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Pad state
        if "state" in data:
            state = data["state"]
            if state.shape[-1] < self.target_dim:
                pad_width = self.target_dim - state.shape[-1]
                data["state"] = np.pad(
                    state, [(0, 0)] * (len(state.shape) - 1) + [(0, pad_width)]
                )

        # Pad actions
        if "actions" in data:
            actions = data["actions"]
            if actions.shape[-1] < self.target_dim:
                pad_width = self.target_dim - actions.shape[-1]
                data["actions"] = np.pad(
                    actions,
                    [(0, 0)] * (len(actions.shape) - 1) + [(0, pad_width)],
                )

        return data


@dataclasses.dataclass
class TokenizePrompt:
    """Tokenize language prompt for the model.

    Uses the PaliGemma tokenizer with SentencePiece.
    """

    max_length: int = 200
    vocab_path: str | None = None

    def __post_init__(self):
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                import sentencepiece as spm
            except ImportError:
                raise ImportError(
                    "sentencepiece required for tokenization. "
                    "Install with: pip install sentencepiece"
                )

            if self.vocab_path:
                self._tokenizer = spm.SentencePieceProcessor()
                self._tokenizer.Load(self.vocab_path)
            else:
                # Use default tokenizer from openpi
                import sys
                import os

                openpi_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")
                )
                if openpi_path not in sys.path:
                    sys.path.insert(0, openpi_path)

                from openpi.models.tokenizer import PaligemmaTokenizer

                self._tokenizer = PaligemmaTokenizer(self.max_length)

        return self._tokenizer

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "prompt" not in data:
            return data

        tokenizer = self._get_tokenizer()
        prompt = data.pop("prompt")

        if isinstance(prompt, np.ndarray):
            prompt = prompt.item()

        # Tokenize
        if hasattr(tokenizer, "tokenize"):
            # PaligemmaTokenizer
            state = data.get("state")
            tokens, mask = tokenizer.tokenize(prompt, state)
            data["tokenized_prompt"] = tokens
            data["tokenized_prompt_mask"] = mask
        else:
            # SentencePiece
            token_ids = tokenizer.EncodeAsIds(prompt)
            # Pad/truncate to max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[: self.max_length]
            else:
                token_ids = token_ids + [0] * (self.max_length - len(token_ids))

            data["tokenized_prompt"] = np.array(token_ids, dtype=np.int32)
            data["tokenized_prompt_mask"] = np.array(
                [1] * len(token_ids) + [0] * (self.max_length - len(token_ids)),
                dtype=bool,
            )

        return data


@dataclasses.dataclass
class RepackImages:
    """Repack image keys to standard format.

    Maps various LeRobot image key formats to the standard FLA format:
    - base_0_rgb: Main/overhead camera
    - left_wrist_0_rgb: Left wrist camera
    - right_wrist_0_rgb: Right wrist camera
    """

    key_mapping: dict[str, str] | None = None

    def __post_init__(self):
        if self.key_mapping is None:
            # Default mapping for common datasets
            self.key_mapping = {
                "observation.images.top": "base_0_rgb",
                "observation.images.cam_high": "base_0_rgb",
                "observation.images.left_wrist": "left_wrist_0_rgb",
                "observation.images.cam_left_wrist": "left_wrist_0_rgb",
                "observation.images.right_wrist": "right_wrist_0_rgb",
                "observation.images.cam_right_wrist": "right_wrist_0_rgb",
                "observation.image": "base_0_rgb",
            }

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "image" not in data:
            return data

        repacked = {}
        image_masks = {}

        for old_key, img in data["image"].items():
            new_key = self.key_mapping.get(old_key, old_key)
            repacked[new_key] = img
            image_masks[new_key] = True

        # Ensure all required keys exist
        required_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for key in required_keys:
            if key not in repacked:
                # Create dummy black image
                if repacked:
                    sample_img = next(iter(repacked.values()))
                    repacked[key] = np.zeros_like(sample_img)
                else:
                    repacked[key] = np.zeros((224, 224, 3), dtype=np.uint8)
                image_masks[key] = False

        data["image"] = repacked
        data["image_mask"] = image_masks
        return data


def create_transform_pipeline(
    *,
    norm_stats: dict[str, NormStats] | None = None,
    use_quantile_norm: bool = True,
    target_action_dim: int = 14,
    image_size: tuple[int, int] = (224, 224),
    max_token_len: int = 200,
) -> CompositeTransform:
    """Create a standard transform pipeline for VLA training.

    Args:
        norm_stats: Normalization statistics.
        use_quantile_norm: Whether to use quantile normalization.
        target_action_dim: Target action dimension for padding.
        image_size: Target image resolution.
        max_token_len: Maximum token length for prompts.

    Returns:
        Composite transform with full pipeline.
    """
    transforms = [
        RepackImages(),
        ResizeImages(height=image_size[0], width=image_size[1]),
        NormalizeImages(),
        PadActions(target_dim=target_action_dim),
    ]

    if norm_stats:
        transforms.append(
            Normalize(norm_stats, use_quantiles=use_quantile_norm)
        )

    transforms.append(TokenizePrompt(max_length=max_token_len))

    return CompositeTransform(transforms)
