from pathlib import Path

from occany.utils.runtime_paths import prepend_vendored_import_paths


prepend_vendored_import_paths(Path(__file__).resolve().parent)

from occany.training_multiview import get_args_parser, train


if __name__ == "__main__":
    train(get_args_parser().parse_args())
