"""Raster avatar renderer for the sagittal jumper view."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import atan2
from pathlib import Path

import numpy as np

_REPO_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "avatar_segments"
_PACKAGE_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "avatar_segments"


@dataclass(frozen=True)
class SpriteSpec:
    """Describe one raster segment sprite and its anchors."""

    filename: str
    distal_anchor_px: tuple[float, float]
    proximal_anchor_px: tuple[float, float]


def pillow_available() -> bool:
    """Return whether Pillow is available for raster rendering."""

    try:
        import PIL.Image  # noqa: F401
    except Exception:
        return False
    return True


def _asset_dir_candidates() -> tuple[Path, ...]:
    """Return the candidate directories containing the avatar raster assets."""

    return (_PACKAGE_ASSET_DIR, _REPO_ASSET_DIR)


def _asset_dir() -> Path:
    """Return the first existing asset directory."""

    for candidate in _asset_dir_candidates():
        if candidate.exists():
            return candidate
    return _REPO_ASSET_DIR


@lru_cache(maxsize=1)
def avatar_rendering_diagnostics() -> tuple[bool, str]:
    """Return whether raster avatar rendering is available and why."""

    if not pillow_available():
        return False, "Pillow n'est pas installe, retour au stick figure."

    asset_dir = _asset_dir()
    expected_files = ("cuisse.png", "jambe_pied_extension.png", "tronc_mains_hanches.png")
    missing_files = [filename for filename in expected_files if not (asset_dir / filename).exists()]
    if missing_files:
        return False, f"assets raster manquants: {', '.join(missing_files)}"

    try:
        for name in ("leg_foot", "thigh", "trunk"):
            sprite_spec(name)
    except Exception as exc:  # pragma: no cover - protective runtime branch
        return False, f"avatar raster indisponible: {exc}"

    return True, f"avatar raster actif ({asset_dir.name}, {len(expected_files)} images)"


def _is_dark(pixel: tuple[int, int, int]) -> bool:
    return pixel[0] < 65 and pixel[1] < 65 and pixel[2] < 65


def _is_foreground(pixel: tuple[int, int, int, int]) -> bool:
    return pixel[3] > 0 and (pixel[0] < 245 or pixel[1] < 245 or pixel[2] < 245)


def _looks_like_light_neutral_background(pixel: tuple[int, int, int, int]) -> bool:
    """Return whether one pixel likely belongs to the exported checker background."""

    if pixel[3] == 0:
        return False
    color_span = max(pixel[:3]) - min(pixel[:3])
    mean_value = sum(pixel[:3]) / 3.0
    return color_span <= 12 and mean_value >= 225.0


def _component_centers(image) -> list[tuple[float, float]]:
    width, height = image.size
    pixels = image.load()
    dark = bytearray(width * height)
    for y in range(height):
        row = y * width
        for x in range(width):
            dark[row + x] = 1 if _is_dark(pixels[x, y][:3]) else 0

    seen = bytearray(width * height)
    centers: list[tuple[float, float]] = []
    for start in range(width * height):
        if not dark[start] or seen[start]:
            continue
        stack = [start]
        seen[start] = 1
        count = 0
        sum_x = 0
        sum_y = 0
        min_x = width
        max_x = -1
        min_y = height
        max_y = -1
        while stack:
            index = stack.pop()
            y, x = divmod(index, width)
            count += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                row = ny * width
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = row + nx
                    if dark[neighbor] and not seen[neighbor]:
                        seen[neighbor] = 1
                        stack.append(neighbor)

        box_width = max_x - min_x + 1
        box_height = max_y - min_y + 1
        fill_ratio = count / max(box_width * box_height, 1)
        round_dot = (
            150 <= count <= 5000
            and 12 <= box_width <= 80
            and 12 <= box_height <= 80
            and abs(box_width - box_height) <= 12
            and 0.35 <= fill_ratio <= 0.95
        )
        if round_dot:
            centers.append((sum_x / count, sum_y / count))
    return sorted(centers, key=lambda center: (center[1], center[0]))


def _distal_anchor_from_silhouette(image) -> tuple[float, float]:
    width, height = image.size
    pixels = image.load()
    rightmost = 0
    y_values: list[int] = []
    for y in range(height):
        for x in range(width):
            if _is_foreground(pixels[x, y]):
                if x > rightmost:
                    rightmost = x
                    y_values = [y]
                elif x >= rightmost - 2:
                    y_values.append(y)
    if not y_values:
        raise ValueError("No foreground pixels found in sprite silhouette")
    y_values.sort()
    return float(rightmost), float(y_values[len(y_values) // 2])


def _proximal_anchor_from_top_silhouette(image) -> tuple[float, float]:
    width, height = image.size
    pixels = image.load()
    topmost = height
    x_values: list[int] = []
    for y in range(height):
        row_x = [x for x in range(width) if _is_foreground(pixels[x, y])]
        if not row_x:
            continue
        if y < topmost:
            topmost = y
            x_values = row_x
        elif y <= topmost + 6:
            x_values.extend(row_x)
    if not x_values:
        raise ValueError("No foreground pixels found near top silhouette")
    x_values.sort()
    return float(x_values[len(x_values) // 2]), float(topmost)


@lru_cache(maxsize=8)
def _load_transparent_sprite(filename: str):
    from PIL import Image

    image = Image.open(_asset_dir() / filename).convert("RGBA")
    pixels = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if _looks_like_light_neutral_background((r, g, b, a)):
                pixels[x, y] = (255, 255, 255, 0)
            else:
                pixels[x, y] = (r, g, b, a)
    return image


def _foreground_bbox(image) -> tuple[int, int, int, int]:
    """Return the bounding box of non-transparent pixels."""

    alpha = np.asarray(image)[..., 3]
    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        raise ValueError("Sprite image does not contain any visible foreground pixels")
    top = int(nonzero[:, 0].min())
    bottom = int(nonzero[:, 0].max())
    left = int(nonzero[:, 1].min())
    right = int(nonzero[:, 1].max())
    return left, top, right + 1, bottom + 1


@lru_cache(maxsize=8)
def sprite_spec(name: str) -> SpriteSpec:
    """Return one cached sprite specification."""

    if not pillow_available():
        raise RuntimeError("Pillow is required for raster avatar rendering")

    filename_by_name = {
        "leg_foot": "jambe_pied_extension.png",
        "thigh": "cuisse.png",
        "trunk": "tronc_mains_hanches.png",
    }
    filename = filename_by_name[name]
    image = _load_transparent_sprite(filename)
    centers = _component_centers(image)

    if name == "thigh":
        if len(centers) < 2:
            raise ValueError(f"Expected two anchors in {filename}, found {len(centers)}")
        return SpriteSpec(filename=filename, distal_anchor_px=centers[-1], proximal_anchor_px=centers[0])

    if name == "leg_foot":
        if len(centers) < 1:
            raise ValueError(f"Expected one proximal anchor in {filename}, found none")
        return SpriteSpec(
            filename=filename,
            distal_anchor_px=_distal_anchor_from_silhouette(image),
            proximal_anchor_px=centers[0],
        )

    if name == "trunk":
        if len(centers) < 1:
            raise ValueError(f"Expected one distal anchor in {filename}, found none")
        return SpriteSpec(
            filename=filename,
            distal_anchor_px=centers[-1],
            proximal_anchor_px=_proximal_anchor_from_top_silhouette(image),
        )

    raise ValueError(f"Unknown sprite name: {name}")


@lru_cache(maxsize=8)
def sprite_array_and_anchors(
    name: str,
    *,
    flip_horizontal: bool = False,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Return one flipped sprite array with lower-origin anchors."""

    image = _load_transparent_sprite(sprite_spec(name).filename)
    spec = sprite_spec(name)
    left, top, right, bottom = _foreground_bbox(image)
    image = image.crop((left, top, right, bottom))
    width, height = image.size
    distal_anchor = (
        spec.distal_anchor_px[0] - left,
        height - (spec.distal_anchor_px[1] - top),
    )
    proximal_anchor = (
        spec.proximal_anchor_px[0] - left,
        height - (spec.proximal_anchor_px[1] - top),
    )
    array = np.flipud(np.asarray(image))
    if flip_horizontal:
        array = np.fliplr(array)
        distal_anchor = (width - distal_anchor[0], distal_anchor[1])
        proximal_anchor = (width - proximal_anchor[0], proximal_anchor[1])
    return array, distal_anchor, proximal_anchor


def draw_segment_image(
    axis,
    name: str,
    *,
    distal_point: tuple[float, float],
    proximal_point: tuple[float, float],
    alpha: float = 1.0,
    flip_horizontal: bool = False,
    zorder: float = 3.0,
) -> bool:
    """Draw one raster sprite aligned with the target segment."""

    available, _ = avatar_rendering_diagnostics()
    if not available:
        return False

    from matplotlib.transforms import Affine2D

    sprite_array, distal_anchor, proximal_anchor = sprite_array_and_anchors(
        name,
        flip_horizontal=flip_horizontal,
    )
    sprite_array = np.asarray(sprite_array).copy()
    sprite_array[..., 3] = np.clip(sprite_array[..., 3].astype(float) * float(alpha), 0.0, 255.0).astype(
        np.uint8
    )
    height, width = sprite_array.shape[0], sprite_array.shape[1]
    source_vector = (
        proximal_anchor[0] - distal_anchor[0],
        proximal_anchor[1] - distal_anchor[1],
    )
    target_vector = (
        proximal_point[0] - distal_point[0],
        proximal_point[1] - distal_point[1],
    )
    source_length = max(float(np.hypot(*source_vector)), 1e-12)
    target_length = max(float(np.hypot(*target_vector)), 1e-12)
    angle = atan2(target_vector[1], target_vector[0]) - atan2(source_vector[1], source_vector[0])
    scale = target_length / source_length

    transform = (
        Affine2D()
        .translate(-distal_anchor[0], -distal_anchor[1])
        .scale(scale, scale)
        .rotate(angle)
        .translate(distal_point[0], distal_point[1])
        + axis.transData
    )
    axis.imshow(
        sprite_array,
        origin="lower",
        extent=(0.0, float(width), 0.0, float(height)),
        transform=transform,
        interpolation="bilinear",
        zorder=zorder,
        clip_on=False,
    )
    return True
