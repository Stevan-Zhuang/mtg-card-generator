import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import re
import pandas as pd
import random

ICON_SIZE = 15

CARD_SIZE = (400, 560)

MARGIN_SIDE = 36
MARGIN_TOP = 66

NAME_MARGINS = (38, 36)

TEXT_MARGINS = (38, 360)
MAX_TEXT_WIDTH = CARD_SIZE[0] - TEXT_MARGINS[0] * 2

TEXT_PADDING = 5

CARD_TYPES = ["Land", "Creature", "Artifact",
              "Enchantment", "Instant", "Sorcery"]

TYPE_MARGINS = (38, 320)

MANA_MARGINS = (362, 36)
MANA_SIZE = 20

STATS_MARGINS = (332, 502)

class TextWrapper(object):
    """
    Helper class to wrap text in lines, based on given text, font and max
    allowed line width.

    Code is modified from:
    https://stackoverflow.com/questions/7698231/pil-draw-multiline-text-on-image
    """

    def __init__(self, text, font, max_width):
        self.text = text
        self.text_lines = [
            ' '.join([w.strip() for w in l.split(' ') if w])
            for l in text.split('\n')
            if l
        ]
        self.font = font
        self.max_width = max_width

        self.draw = ImageDraw.Draw(
            Image.new(
                mode='RGB',
                size=(100, 100)
            )
        )

        self.space_width = self.draw.textsize(
            text=' ',
            font=self.font
        )[0]

    def get_text_width(self, text):
        text_width = 0

        slices = []
        for match in re.finditer("{[^{^}]*}", text):
            span = match.span()
            slices.append(span)
            symbol_key = text[span[0] + 1: span[1] - 1].replace('/', '')
            try:
                Image.open(f"data/symbols/{symbol_key}.png")
                text_width += ICON_SIZE
            except:
                pass
        for span in slices[::-1]:
            text = text[:span[0]] + text[span[1]:]
        text = text.replace("{", "").replace("}", "")
            
        text_width += self.draw.textsize(
            text=text,
            font=self.font
        )[0]
        return text_width

    def wrapped_text(self):
        wrapped_lines = []
        buf = []
        buf_width = 0

        for line in self.text_lines:
            for word in line.split(' '):
                word_width = self.get_text_width(word)

                expected_width = word_width if not buf else \
                    buf_width + self.space_width + word_width

                if expected_width <= self.max_width:
                    # word fits in line
                    buf_width = expected_width
                    buf.append(word)
                else:
                    # word doesn't fit in line
                    wrapped_lines.append(' '.join(buf))
                    buf = [word]
                    buf_width = word_width

            if buf:
                wrapped_lines.append(' '.join(buf))
                buf = []
                buf_width = 0

        return '\n'.join(wrapped_lines)

def generate_card(text, name, flavor, art, show=False):
    """Return a card image built from all generated data."""
    card_type = random.choice(CARD_TYPES)
    card_type = "Creature"

    card_template = Image.open("data/card_template_transparent.png")
    if card_type == "Creature":
        card_template = Image.open(
            "data/card_template_transparent_creature.png"
        )
    # Card art
    art_numpy = (art * 255).to(torch.uint8).numpy()
    art_pil = Image.fromarray(art_numpy)
    scaled_width = card_template.size[0] - MARGIN_SIDE * 2
    scaled_art = art_pil.resize((scaled_width, scaled_width))

    padded_art = Image.new("RGB", card_template.size, "white")
    padded_art.paste(scaled_art, box=(MARGIN_SIDE, MARGIN_TOP))
    padded_art.paste(card_template, (0, 0), card_template)

    art_draw = ImageDraw.Draw(padded_art)

    def draw_text_with_symbols(text, font, xy):
        slices = [(0, 0)]
        for match in re.finditer("{[^{^}]*}", text):
            span = match.span()
            slices.append(span)

        split_text = []
        is_symbol = []
        for span in slices[::-1]:
            if len(text[span[1]:]) > 0:
                clean_text = text[span[1]:].replace("{", "").replace("}", "")
                split_text.append(clean_text)
                is_symbol.append(False)

            symbol_key = text[span[0] + 1: span[1] - 1].replace('/', '')
            try:
                Image.open(f"data/symbols/{symbol_key}.png")
                split_text.append(symbol_key)
                is_symbol.append(True)
            except:
                pass

            text = text[:span[0]]

        split_text = split_text[::-1]
        is_symbol = is_symbol[::-1]

        shift = 0
        for idx, part in enumerate(split_text):
            if is_symbol[idx]:
                symbol = Image.open(f"data/symbols/{part}.png")
                symbol = symbol.resize((ICON_SIZE, ICON_SIZE))
                padded_art.paste(symbol, (xy[0] + shift, xy[1]), symbol)
                shift += ICON_SIZE
            else:
                art_draw.text((xy[0] + shift, xy[1]), part, "black", font=font)
                shift += art_draw.textsize(text=part, font=font)[0]

    # Draw mana and name
    def weighted_random(c, f=1):
        return min(abs(int(random.gauss(f, c ** 0.5))), c)

    mana_cost = []
    if random.choice([True, False]):
        mana_cost.append(weighted_random(20))

    mana_types = "GWUBR"
    mana_selection = random.sample(mana_types, k=weighted_random(5, 2))
    for mana_type in mana_selection:
        mana_cost.extend([mana_type] * (weighted_random(2) + 1))

    if card_type == "Creature":
        creature_types = pd.read_json("data/creature-types.json")
        card_type += f" — {random.choice(creature_types['data'])}"

        attack = weighted_random(10, 3)
        defense = weighted_random(9, 3) + 1
        stats = f"{attack}/{defense}"

        stats_font = ImageFont.truetype("data/fonts/mplanti1.ttf", 24)
        w = art_draw.textsize(text=stats, font=stats_font)[0]
        art_draw.text((STATS_MARGINS[0] - (w // 2), STATS_MARGINS[1]),
                      stats, "black", font=stats_font)

    elif card_type == "Artifact":
        mana_cost = [mana for mana in mana_cost if isinstance(mana, int)]
    try:
        if mana_cost[0] == 0 and len(mana_cost) > 1:
            mana_cost.pop(0)
    except:
        mana_cost.append(0)
    if card_type == "Land":
        mana_cost = []

    if len(mana_cost) > 6:
        mana_cost = mana_cost[0:1] + random.sample(mana_cost[1:], k=5)

    def mana_cost_sort(x):
        if isinstance(x, int):
            return -1
        return mana_types.index(x)
        
    mana_cost.sort(key=mana_cost_sort)

    mana_shift = MANA_SIZE
    for mana in mana_cost[::-1]:
        symbol = Image.open(f"data/symbols/{mana}.png")
        symbol = symbol.resize((MANA_SIZE, MANA_SIZE))
        padded_art.paste(
            symbol, (MANA_MARGINS[0] - mana_shift, MANA_MARGINS[1]), symbol
        )
        mana_shift += MANA_SIZE

    name_font_size = 22
    name_font = ImageFont.truetype("data/fonts/mplanti1.ttf", name_font_size)
    name_shift = 0
    while True:
        fit = MAX_TEXT_WIDTH - mana_shift
        if art_draw.textsize(text=name, font=name_font)[0] <= fit:
            break
        name_font_size -= 2
        name_shift += 1
        name_font = ImageFont.truetype(
            "data/fonts/mplanti1.ttf", name_font_size
        )
    art_draw.text((NAME_MARGINS[0], NAME_MARGINS[1] + name_shift), name, "black",
                  font=name_font)

    # Draw description
    height = TEXT_MARGINS[1]

    text_font = ImageFont.truetype("data/fonts/mplantin.ttf", 16)
    for block in text.split("\n")[:2]:
        lines = TextWrapper(block, text_font, MAX_TEXT_WIDTH).wrapped_text()
        for line in lines.split("\n"):
            draw_text_with_symbols(line, text_font, (TEXT_MARGINS[0], height))
            height += art_draw.textsize(text=line, font=text_font)[1]
        height += TEXT_PADDING

    height += TEXT_PADDING
    flavor_font = ImageFont.truetype("data/fonts/mplantinit.ttf", 16)
    lines = TextWrapper(flavor, flavor_font, MAX_TEXT_WIDTH).wrapped_text()
    art_draw.text((TEXT_MARGINS[0], height), lines, "black", font=flavor_font)

    # Card type
    type_font = ImageFont.truetype("data/fonts/mplanti1.ttf", 20)
    art_draw.text(TYPE_MARGINS, card_type, "black", font=type_font)

    if show:
        plt.imshow(padded_art)
        plt.axis('off')
        plt.show()

    return padded_art