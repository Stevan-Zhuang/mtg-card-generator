from config import get_config
from text_model import MTGTextModel
from art_model import MTGArtModel, single_image
from mtg_text_generator import run_text
from mtg_art_generator import run_art
from card_generator import generate_card

if __name__ == "__main__":
    config = get_config()

    if config.text_training:
        run_text(config)
    if config.art_training:
        run_art(config)

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

    generate_card(text_model(), name_model(), flavor_model(),
                  single_image(art_model), show=True)