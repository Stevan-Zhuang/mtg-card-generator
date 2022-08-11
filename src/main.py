from config import get_config
from text_model import MTGTextModel
from art_model import MTGArtModel, demo_single, demo_grid
from mtg_text_generator import run_text
from mtg_art_generator import run_art

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

    demo_single(art_model)
    demo_grid(art_model)
