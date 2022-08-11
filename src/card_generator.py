import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from textwrap import wrap

from typing import Optional

MARGIN_SIDE = 36
MARGIN_TOP = 66

NAME_MARGINS = (38, 38)
TEXT_MARGINS = (40, 370)

card_template_path = "data/card_template_transparent.png"

def generate_card(text, name, flavor, art, show=False):
    card_template = Image.open(card_template_path)

    art_numpy = (art * 255).to(torch.uint8).numpy()
    art_pil = Image.fromarray(art_numpy)
    scaled_width = card_template.size[0] - MARGIN_SIDE * 2
    scaled_art = art_pil.resize((scaled_width, scaled_width))

    padded_art = Image.new("RGB", card_template.size, "white")
    padded_art.paste(scaled_art, box=(MARGIN_SIDE, MARGIN_TOP))
    padded_art.paste(card_template, (0, 0), card_template)

    art_draw = ImageDraw.Draw(padded_art)

    # TODO
    # Make name text shrink if too large
    # Multi line descriptions
    # Flavor text
    # Land cost icons

    name_font = ImageFont.truetype("data/fonts/mplanti1.ttf", 22)
    art_draw.text(NAME_MARGINS, name, "black", font=name_font)

    text_font = ImageFont.truetype("data/fonts/mplantin.ttf", 16)
    art_draw.text(TEXT_MARGINS, text, "black", font=text_font)

    flavor_font = ImageFont.truetype("data/fonts/mplantinit.ttf", 16)

    print(name)
    print(text)
    print(flavor)

    if show:
        plt.imshow(padded_art)
        plt.axis('off')
        plt.show()
