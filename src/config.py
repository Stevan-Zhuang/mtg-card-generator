import argparse
import string

def get_config():
    """Build a config from command line argument and return it."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_training", type=bool, default=False)
    parser.add_argument("--art_training", type=bool, default=False)

    # Text defaults
    name_vocab = string.ascii_letters + " ,'-$"
    text_vocab = string.ascii_letters + string.digits + ",'-.\n \"+/:;—−()[]{}$"
    flavor_vocab = string.ascii_letters + string.digits + "!?#,'-. \"/:;—−$"

    parser.add_argument("--name_vocab", type=str, default=name_vocab)
    parser.add_argument("--name_vocab_size", type=int, default=len(name_vocab))
    parser.add_argument("--name_char2idx", type=dict, default={
        char: idx for idx, char in enumerate(name_vocab)})
    parser.add_argument("--name_max_length", type=int, default=33)

    parser.add_argument("--text_vocab", type=str, default=text_vocab)
    parser.add_argument("--text_vocab_size", type=int, default=len(text_vocab))
    parser.add_argument("--text_char2idx", type=dict, default={
        char: idx for idx, char in enumerate(text_vocab)})
    parser.add_argument("--text_max_length", type=int, default=580)

    parser.add_argument("--flavor_vocab", type=str, default=flavor_vocab)
    parser.add_argument("--flavor_vocab_size", type=int, default=len(flavor_vocab))
    parser.add_argument("--flavor_char2idx", type=dict, default={
        char: idx for idx, char in enumerate(flavor_vocab)})
    parser.add_argument("--flavor_max_length", type=int, default=403)

    parser.add_argument("--accent2normal", type=dict, default={
        "Ä": "A", "Æ": "AE", "á": "a", "â": "a", "æ": "ae",
        "é": "e", "ö": "o", "û": "u", "ü": "u"})

    # Text training
    parser.add_argument("--text_gpu", type=bool, default=False)
    parser.add_argument("--text_data_dir", type=str,
                        default="data/magic-the-gathering-cards")
    parser.add_argument("--text_lr", type=float, default=1e-2)
    parser.add_argument("--text_n_epochs", type=int, default=25)
    parser.add_argument("--text_patience", type=int, default=5)
    parser.add_argument("--text_batch_size", type=int, default=100)
    parser.add_argument("--text_hidden_size", type=int, default=128)
    parser.add_argument("--text_n_layers", type=int, default=2)

    # Art training
    parser.add_argument("--art_gpu", type=bool, default=False)
    parser.add_argument("--art_data_dir", type=str,
                        default="data/magic-the-gathering-art")
    parser.add_argument("--art_n_epochs", type=int, default=40)
    parser.add_argument("--art_batch_size", type=int, default=128)
    parser.add_argument("--art_n_channels", type=int, default=3)
    parser.add_argument("--art_image_size", type=int, default=64)

    # Checkpoints
    parser.add_argument("--name_model_checkpoint", type=str,
                        default="checkpoints/name_model_2.ckpt")
    parser.add_argument("--text_model_checkpoint", type=str,
                        default="checkpoints/text_model_2.ckpt")
    parser.add_argument("--flavor_model_checkpoint", type=str,
                        default="checkpoints/flavor_model_2.ckpt")
    parser.add_argument("--art_model_checkpoint", type=str,
                        default="checkpoints/art_model_4.ckpt")

    return parser.parse_args()