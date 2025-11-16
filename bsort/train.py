"""
bsort: Model Training Script
"""

import yaml
import wandb
from pathlib import Path
from ultralytics import YOLO
from rich.console import Console

console = Console()

def run(config_path: Path):
    """
    Main function to run the model training pipeline.
    """
    console.rule("[bold blue]Model Training[/bold blue]")

    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize WandB
    console.print("Initializing WandB...")
    try:
        wandb.login()
    except Exception as e:
        console.print(f"[bold red]WandB login failed:[/bold red] {e}")
        console.print("Set WANDB_API_KEY environment variable.")
        return

    wandb.init(
        project=config['wandb_project'],
        entity=config['wandb_entity'],
        config=config,
        name=f"train_{config['model_name']}_e{config['epochs']}_b{config['batch_size']}"
    )

    # 3. Load Model
    model = YOLO(config['model_name'])
    
    # 4. Run Training
    console.print(f"Starting training with model: {config['model_name']}")
    console.print(f"Dataset: {config['data_yaml']}")
    
    model.train(
        data=config['data_yaml'],
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['img_size'],
        lr0=config['learning_rate'],
        project=config['wandb_project'], # Saves logs locally
        name="yolo_run", # Local folder name
        exist_ok=True,
        save=True,  # Save checkpoints
        val=True,   # Run validation
    )
    
    console.print("[bold green]Training complete.[/bold green]")

    # 5. Save final model from WandB (best.pt)
    # The 'best.pt' is saved locally by YOLO in 'runs/detect/train'
    # We just need to find it and export it.
    
    # Find the latest training run
    run_dir = list(Path(config['wandb_project']).glob("yolo_run*"))[-1]
    best_model_path = run_dir / "weights/best.pt"
    
    if not best_model_path.exists():
        console.print(f"[bold red]Could not find best.pt in {run_dir}[/bold red]")
        return
        
    console.print(f"Found best model: {best_model_path}")

    # 6. Export to ONNX and TFLite
    console.print("Exporting to ONNX and TFLite...")
    model = YOLO(best_model_path) # Load the best model
    
    # Export ONNX
    onnx_path = model.export(format="onnx")
    console.print(f"Exported ONNX: {onnx_path}")
    
    # Export TFLite (INT8 quantized)
   #tflite_path = model.export(format="tflite", int8=True, data=config['data_yaml'])
    #console.print(f"Exported TFLite (INT8): {tflite_path}")
    
    console.print("[bold green]All tasks complete.[/bold green]")
    wandb.finish()