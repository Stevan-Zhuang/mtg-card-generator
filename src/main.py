from config import get_config
from text_model import MTGTextModel
from art_model import MTGArtModel, single_image
from mtg_text_generator import run_text
from mtg_art_generator import run_art
from card_generator import generate_card
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = get_config()

    if config.text_training:
        run_text(config)
    if config.art_training:
        run_art(config)

    if config.infer:
        text_model = MTGTextModel.load_from_checkpoint(
            config.text_model_checkpoint
        )
        name_model = MTGTextModel.load_from_checkpoint(
            config.name_model_checkpoint
        )
        flavor_model = MTGTextModel.load_from_checkpoint(
            config.flavor_model_checkpoint
        )
        art_model = MTGArtModel.load_from_checkpoint(
            config.art_model_checkpoint
        )

        for idx in range(config.n_cards):
            text = text_model()
            name = name_model()
            flavor = flavor_model()
            art = single_image(art_model)

            card = generate_card(text, name, flavor, art)
            card.save(f"showcase/{idx + 1}.png")