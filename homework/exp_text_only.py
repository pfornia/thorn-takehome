"""E001 — Does the *semantic content* of rendered text affect the prediction?

Renders plain black text on a white background and scores it. If "Dog" scores higher than
control words, the model has some OCR-like sensitivity and watermark *wording* matters.
If all words score alike, any watermark effect is texture/layout-driven, not semantic.

Includes a blank-white control to separate "effect of the word" from "effect of a mostly-white
image with some dark marks on it".
"""

from PIL import Image, ImageDraw, ImageFont

from score import format_score, score_image

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
CANVAS = 512


def render_text(text: str, font_size: int = 160, font_path: str = FONT_PATH) -> Image.Image:
    """Black text centred on a white canvas."""
    img = Image.new("RGB", (CANVAS, CANVAS), "white")
    if not text:
        return img
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((CANVAS - (right - left)) / 2 - left, (CANVAS - (bottom - top)) / 2 - top),
        text,
        fill="black",
        font=font,
    )
    return img


WORDS = [
    "",            # blank white control — isolates "white canvas" from "word"
    "Dog",
    "Cat",
    "DOG",
    "dog",
    "Dogs",
    "Puppy",
    "Labrador",
    "Other",
    "Car",
    "Tree",
    "Ocean",
    "Xqzptl",      # pronounceable-ish gibberish control
    "|||||||",     # pure vertical strokes — texture control, no semantics
]

if __name__ == "__main__":
    from score import HOMEWORK_DIR

    out_dir = HOMEWORK_DIR / "outputs" / "e001_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E001 — plain black text on white, Arial 160pt, 512x512 canvas")
    print("-" * 112)
    results = []
    for word in WORDS:
        img = render_text(word)
        img.save(out_dir / f"{word or 'BLANK'}.png")
        r = score_image(img)
        margin = r["logits"]["dog"] - r["logits"]["other"]
        results.append((word, r, margin))
        label = repr(word) if word else "(blank white)"
        print(f"{format_score(label, r)}  dog_margin={margin:+.3f}")

    print("-" * 112)
    print("Ranked by dog margin (higher = closer to a dog prediction):")
    for word, r, margin in sorted(results, key=lambda t: -t[2]):
        label = repr(word) if word else "(blank white)"
        print(f"  {label:<18} dog_margin={margin:+8.3f}   dog_prob={r['probs']['dog']:.6f}")
