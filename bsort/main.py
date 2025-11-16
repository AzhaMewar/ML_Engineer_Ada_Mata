"""
bsort: Main CLI application for the bottle cap sorter.
"""

import typer
from typing_extensions import Annotated
from pathlib import Path
import warnings

# Import your script functions
from . import preprocess, train, infer

# Suppress warnings from libraries
warnings.filterwarnings("ignore")

# Create the main Typer app
app = typer.Typer(
    name="bsort",
    help="Ada Mata Bottle Cap Sorter CLI",
    add_completion=False,
)

# --- Common Options ---
ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Path to the settings.yaml config file.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
]

# --- Register Commands ---

@app.command(help="Preprocess and re-label the raw dataset.")
def run_preprocess(config: ConfigOption):
    """
    Runs the data preprocessing and re-labeling pipeline.
    """
    preprocess.run(config_path=config)


@app.command(help="Train the YOLOv8 model.")
def run_train(config: ConfigOption):
    """
    Runs the model training pipeline.
    """
    train.run(config_path=config)


@app.command(help="Run inference on a single image.")
def run_infer(
    config: ConfigOption,
    image: Annotated[
        Path,
        typer.Option(
            "--image",
            "-i",
            help="Path to the input image.",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to save the annotated image.",
            resolve_path=True,
        ),
    ] = Path("output.jpg"),
):
    """
    Runs inference on a single image.
    """
    infer.run(config_path=config, image_path=image, output_path=output)


if __name__ == "__main__":
    app()