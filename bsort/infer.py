"""
bsort: Model Inference Script
"""

import yaml
from pathlib import Path
from ultralytics import YOLO
from rich.console import Console

console = Console()

def run(config_path: Path, image_path: Path, output_path: Path):
    """
    Main function to run inference on a single image.
    """
    console.rule("[bold blue]Model Inference[/bold blue]")

    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Find the best optimized model
    # We prioritize the TFLite model for edge inference
    
    # Find the latest training run
    try:
        run_dir = sorted(list(Path(config['wandb_project']).glob("yolo_run*")))[-1]
    except IndexError:
        console.print(f"[bold red]Error:[/bold red] No training runs found in '{config['wandb_project']}'.")
        console.print("Please run 'python bsort/main.py run-train' first.")
        return
    
    model_path = run_dir / "weights/best_int8.tflite"

    if not model_path.exists():
        console.print(f"[bold red]Error:[/bold red] No TFLite model found at {model_path}.")
        console.print("Trying ONNX model...")
        model_path = run_dir / "weights/best.onnx"
        if not model_path.exists():
            console.print(f"[bold red]Error:[/bold red] No ONNX model found either.")
            console.print("Please run 'python bsort/main.py run-train' first.")
            return

    console.print(f"Loading model: {model_path.name}")
    
    # 3. Load Model
    model = YOLO(model_path)
    
    # 4. Run Prediction
    console.print(f"Running inference on: {image_path.name}")
    results = model(image_path)
    
    # 5. Print and Save Results
    console.print("\n[bold]Detections:[/bold]")
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = config['class_names'].get(str(class_id), "Unknown")
        confidence = float(box.conf[0])
        console.print(f"  - [bold]{class_name}[/bold] (Confidence: {confidence:.2f})")
        
    # Save the annotated image
    results[0].save(filename=str(output_path))
    console.print(f"\nAnnotated image saved to: {output_path}")