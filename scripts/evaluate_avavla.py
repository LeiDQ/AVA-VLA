"""
evaluate_avavla.py

Evaluation script for comparing AVA-VLA with baseline OpenVLA on benchmarks.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_scripts.deploy_avavla import (
    load_avavla_model,
    predict_action,
    compute_efficiency_metrics,
)


def evaluate_latency(
    model,
    processor,
    image: Image,
    instruction: str,
    unnorm_key: str,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cuda",
) -> Dict:
    """
    Measure inference latency.
    
    Returns:
        metrics: Dictionary with latency metrics
    """
    # Warmup runs
    for _ in range(warmup_runs):
        model.reset_latent_history()
        _ = predict_action(
            model=model,
            processor=processor,
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            device=device,
            update_history=False,
        )
    
    # Measure latency
    latencies = []
    reasoning_steps_list = []
    
    for _ in range(num_runs):
        model.reset_latent_history()
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        actions, reasoning_info = predict_action(
            model=model,
            processor=processor,
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            device=device,
            update_history=False,
        )
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
        reasoning_steps_list.append(reasoning_info['num_steps_performed'])
    
    return {
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p90_latency_ms": np.percentile(latencies, 90),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_reasoning_steps": np.mean(reasoning_steps_list),
        "throughput_hz": 1000.0 / np.mean(latencies),
    }


def evaluate_on_dataset(
    model,
    processor,
    dataset_path: Path,
    unnorm_key: str,
    num_samples: int = 100,
    device: str = "cuda",
) -> Tuple[Dict, List]:
    """
    Evaluate model on a dataset.
    
    Returns:
        metrics: Dictionary of evaluation metrics
        reasoning_infos: List of reasoning information
    """
    # Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    # Limit number of samples
    if num_samples > 0:
        data = data[:num_samples]
    
    # Evaluate
    predictions = []
    reasoning_infos = []
    latencies = []
    
    for sample in tqdm(data, desc="Evaluating"):
        image_path = Path(sample["image_path"])
        instruction = sample["instruction"]
        ground_truth_actions = np.array(sample["actions"])
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Predict
        model.reset_latent_history()
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        actions, reasoning_info = predict_action(
            model=model,
            processor=processor,
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            device=device,
            update_history=False,
        )
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Compute error
        error = np.linalg.norm(actions - ground_truth_actions)
        
        predictions.append({
            "error": error,
            "actions": actions,
            "ground_truth": ground_truth_actions,
        })
        reasoning_infos.append(reasoning_info)
        latencies.append((end_time - start_time) * 1000)
    
    # Compute metrics
    errors = [p["error"] for p in predictions]
    reasoning_steps = [info["num_steps_performed"] for info in reasoning_infos]
    
    metrics = {
        "num_samples": len(predictions),
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
        "min_error": np.min(errors),
        "max_error": np.max(errors),
        "median_error": np.median(errors),
        "mean_latency_ms": np.mean(latencies),
        "p90_latency_ms": np.percentile(latencies, 90),
        "mean_reasoning_steps": np.mean(reasoning_steps),
        "std_reasoning_steps": np.std(reasoning_steps),
        "min_reasoning_steps": np.min(reasoning_steps),
        "max_reasoning_steps": np.max(reasoning_steps),
        "throughput_hz": 1000.0 / max(float(np.mean(latencies)), 1e-8),
    }
    
    # Compute efficiency metrics
    efficiency_metrics = compute_efficiency_metrics(
        reasoning_infos,
        max_reasoning_steps=getattr(model, "max_reasoning_steps", None),
    )
    metrics.update(efficiency_metrics)
    
    return metrics, reasoning_infos


def compare_models(
    baseline_checkpoint: Path,
    avavla_checkpoint: Path,
    dataset_path: Path,
    unnorm_key: str,
    num_samples: int = 100,
    device: str = "cuda",
) -> Dict:
    """
    Compare AVA-VLA with baseline OpenVLA.
    
    Returns:
        comparison_results: Dictionary with comparison results
    """
    print("=" * 80)
    print("AVA-VLA Evaluation")
    print("=" * 80)
    
    # Load AVA-VLA with latent reasoning enabled
    print("\n[1] Loading AVA-VLA model...")
    avavla_model, avavla_processor, _, _ = load_avavla_model(
        checkpoint_path=avavla_checkpoint,
        device=device,
        enable_latent_reasoning=True,
        max_reasoning_steps=5,
        exit_threshold=0.55,
    )
    
    # Evaluate AVA-VLA
    print("\n[2] Evaluating AVA-VLA on dataset...")
    avavla_metrics, avavla_reasoning_infos = evaluate_on_dataset(
        avavla_model,
        avavla_processor,
        dataset_path,
        unnorm_key,
        num_samples=num_samples,
        device=device,
    )
    
    # Load AVA-VLA without latent reasoning (ablation)
    print("\n[3] Loading AVA-VLA model (w/o latent reasoning)...")
    avavla_no_reasoning_model, avavla_no_reasoning_processor, _, _ = load_avavla_model(
        checkpoint_path=avavla_checkpoint,
        device=device,
        enable_latent_reasoning=False,
        allow_no_reasoning_ablation=True,
    )
    
    print("\n[4] Evaluating AVA-VLA (w/o latent reasoning) on dataset...")
    avavla_no_reasoning_metrics, _ = evaluate_on_dataset(
        avavla_no_reasoning_model,
        avavla_no_reasoning_processor,
        dataset_path,
        unnorm_key,
        num_samples=num_samples,
        device=device,
    )
    
    # Compile comparison results
    comparison_results = {
        "avavla": avavla_metrics,
        "avavla_no_reasoning": avavla_no_reasoning_metrics,
        "improvements": {
            "error_reduction_pct": (
                (avavla_no_reasoning_metrics["mean_error"] - avavla_metrics["mean_error"])
                / avavla_no_reasoning_metrics["mean_error"] * 100
            ),
            "latency_reduction_pct": (
                (avavla_no_reasoning_metrics["mean_latency_ms"] - avavla_metrics["mean_latency_ms"])
                / avavla_no_reasoning_metrics["mean_latency_ms"] * 100
            ),
        },
    }
    
    return comparison_results


def run_libero_benchmark(args) -> int:
    """Run the real LIBERO rollout evaluator with AVA-VLA wired in as the policy."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/robot/libero/run_libero_eval.py"),
        "--model_family",
        "avavla",
        "--pretrained_checkpoint",
        str(args.avavla_checkpoint),
        "--task_suite_name",
        args.task_suite,
        "--num_trials_per_task",
        str(args.num_trials_per_task),
        "--num_images_in_input",
        "2",
        "--use_proprio",
        "True",
        "--enable_latent_reasoning",
        str(args.enable_latent_reasoning),
        "--use_history_state",
        str(args.use_history_state),
        "--center_crop",
        "True",
    ]
    if args.max_reasoning_steps is not None:
        cmd.extend(["--max_reasoning_steps", str(args.max_reasoning_steps)])
    if args.fixed_reasoning_steps is not None:
        cmd.extend(["--fixed_reasoning_steps", str(args.fixed_reasoning_steps)])
    if args.exit_threshold is not None:
        cmd.extend(["--exit_threshold", str(args.exit_threshold)])
    if args.initial_states_path is not None:
        cmd.extend(["--initial_states_path", args.initial_states_path])
    if args.extra_args:
        cmd.extend(args.extra_args)

    print("Running LIBERO benchmark command:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def run_external_benchmark(args, benchmark_name: str) -> int:
    """Run an external LIBERO+ evaluator against an AVA-VLA checkpoint."""
    if args.external_eval_script is None:
        raise SystemExit(
            f"`--benchmark {benchmark_name}` requires `--external-eval-script` because this repository "
            f"does not include a native {benchmark_name.upper()} environment/evaluator."
        )

    cmd = [
        sys.executable,
        str(args.external_eval_script),
        "--checkpoint",
        str(args.avavla_checkpoint),
    ]
    if args.task_suite:
        cmd.extend(["--task_suite", args.task_suite])
    if args.extra_args:
        cmd.extend(args.extra_args)

    print(f"Running external {benchmark_name.upper()} benchmark command:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def run_calvin_benchmark(args) -> int:
    """Run the repository's native official-protocol CALVIN ABC->D evaluator."""
    if args.dataset is None:
        raise SystemExit("`--benchmark calvin` requires `--dataset` pointing to task_ABC_D/debug root.")
    output = args.output or "results/calvin/evaluation_results.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/robot/calvin/run_calvin_eval.py"),
        "--checkpoint",
        str(args.avavla_checkpoint),
        "--dataset-root",
        str(args.dataset),
        "--output",
        str(output),
    ]
    if args.max_reasoning_steps is not None:
        cmd.extend(["--max-reasoning-steps", str(args.max_reasoning_steps)])
    if args.exit_threshold is not None:
        cmd.extend(["--exit-threshold", str(args.exit_threshold)])
    if args.extra_args:
        cmd.extend(args.extra_args)
    print("Running native CALVIN ABC->D command:")
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def print_comparison_results(comparison_results: Dict):
    """Print comparison results in a formatted way."""
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    
    print("\n[AVA-VLA (with latent reasoning)]")
    print("-" * 40)
    print(f"  Mean Error: {comparison_results['avavla']['mean_error']:.4f}")
    print(f"  Median Error: {comparison_results['avavla']['median_error']:.4f}")
    print(f"  Mean Latency: {comparison_results['avavla']['mean_latency_ms']:.2f} ms")
    print(f"  P90 Latency: {comparison_results['avavla']['p90_latency_ms']:.2f} ms")
    print(f"  Throughput: {comparison_results['avavla']['throughput_hz']:.2f} Hz")
    print(f"  Avg Reasoning Steps: {comparison_results['avavla']['mean_reasoning_steps']:.2f}")
    print(f"  Early Exit Rate: {comparison_results['avavla']['early_exit_rate']:.2%}")
    
    print("\n[AVA-VLA (without latent reasoning)]")
    print("-" * 40)
    print(f"  Mean Error: {comparison_results['avavla_no_reasoning']['mean_error']:.4f}")
    print(f"  Median Error: {comparison_results['avavla_no_reasoning']['median_error']:.4f}")
    print(f"  Mean Latency: {comparison_results['avavla_no_reasoning']['mean_latency_ms']:.2f} ms")
    print(f"  P90 Latency: {comparison_results['avavla_no_reasoning']['p90_latency_ms']:.2f} ms")
    print(f"  Throughput: {comparison_results['avavla_no_reasoning']['throughput_hz']:.2f} Hz")
    
    print("\n[Improvements]")
    print("-" * 40)
    print(f"  Error Reduction: {comparison_results['improvements']['error_reduction_pct']:+.2f}%")
    print(f"  Latency Reduction: {comparison_results['improvements']['latency_reduction_pct']:+.2f}%")
    
    print("\n" + "=" * 80)


def save_results(results: Dict, output_path: Path):
    """Save evaluation results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AVA-VLA model")
    parser.add_argument("--benchmark", type=str, default="json",
                       choices=["json", "libero", "calvin", "libero_plus"],
                       help="Evaluation backend: offline JSON metrics, real LIBERO rollout, or external benchmark adapter")
    parser.add_argument("--avavla-checkpoint", type=str, required=True,
                       help="Path to AVA-VLA checkpoint")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to evaluation dataset (JSON)")
    parser.add_argument("--unnorm-key", type=str, default=None,
                       help="Dataset name for action unnormalization")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for results JSON")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                       help="LIBERO task suite or external benchmark suite/split")
    parser.add_argument("--num-trials-per-task", type=int, default=50,
                       help="Number of real LIBERO rollouts per task")
    parser.add_argument("--initial-states-path", type=str, default=None,
                       help="Optional LIBERO initial-states JSON path")
    parser.add_argument("--enable-latent-reasoning", action=argparse.BooleanOptionalAction, default=True,
                       help="Enable AVA-VLA latent reasoning")
    parser.add_argument("--use-history-state", action=argparse.BooleanOptionalAction, default=True,
                       help="Carry h_{t-1} between policy queries during rollout")
    parser.add_argument("--max-reasoning-steps", type=int, default=None,
                       help="Override maximum latent reasoning steps")
    parser.add_argument("--fixed-reasoning-steps", type=int, default=None,
                       help="Use a fixed number of reasoning steps instead of adaptive exit")
    parser.add_argument("--exit-threshold", type=float, default=None,
                       help="Override early-exit threshold")
    parser.add_argument("--external-eval-script", type=Path, default=None,
                       help="External LIBERO+ evaluator script (CALVIN is native)")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                       help="Additional arguments forwarded to the selected benchmark evaluator")
    
    args = parser.parse_args()
    
    args.avavla_checkpoint = Path(args.avavla_checkpoint)

    if args.benchmark == "libero":
        raise SystemExit(run_libero_benchmark(args))
    if args.benchmark == "calvin":
        raise SystemExit(run_calvin_benchmark(args))
    if args.benchmark == "libero_plus":
        raise SystemExit(run_external_benchmark(args, args.benchmark))

    if args.dataset is None or args.unnorm_key is None:
        raise SystemExit("`--benchmark json` requires both `--dataset` and `--unnorm-key`.")

    # Run offline JSON evaluation
    comparison_results = compare_models(
        baseline_checkpoint=None,  # Not needed for this comparison
        avavla_checkpoint=args.avavla_checkpoint,
        dataset_path=Path(args.dataset),
        unnorm_key=args.unnorm_key,
        num_samples=args.num_samples,
        device=args.device,
    )
    
    # Print results
    print_comparison_results(comparison_results)
    
    # Save results
    if args.output:
        save_results(comparison_results, Path(args.output))


if __name__ == "__main__":
    main()
