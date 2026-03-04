"""
Heretic Progress Monitor — watch abliteration in real-time.
Usage: python monitor.py   (run in a separate terminal)
"""
import os
import re
import time
import sys

LOG = os.path.join(os.path.dirname(__file__), "heretic_output.log")

def parse_log():
    if not os.path.exists(LOG):
        return None
    with open(LOG, encoding="utf-8", errors="replace") as f:
        text = f.read()

    info = {
        "model_name": "",
        "gpu": "",
        "model_loaded": False,
        "quantized": False,
        "lora_targets": "",
        "layers": 0,
        "components": [],
        "batch_size": 0,
        "batch_tests": [],
        "prefix": "",
        "base_refusals": "",
        "trials": [],
        "current_trial": 0,
        "total_trials": 50,
        "elapsed": "",
        "remaining": "",
        "memory": "",
        "status": "Starting...",
    }

    # Model name
    m = re.search(r"Loading model (.+?)\.\.\.", text)
    if m:
        info["model_name"] = m.group(1)

    # GPU
    m = re.search(r"GPU 0: (.+?) \((.+?)\)", text)
    if m:
        info["gpu"] = f"{m.group(1)} ({m.group(2)})"

    # Model loaded
    if "Ok (quantized" in text:
        info["model_loaded"] = True
        info["quantized"] = True
    elif "Ok" in text.split("\n")[0:20]:
        info["model_loaded"] = True

    # LoRA
    m = re.search(r"LoRA adapters initialized \(targets: (.+?)\)", text)
    if m:
        info["lora_targets"] = m.group(1)

    # Layers
    m = re.search(r"Transformer model with (\d+) layers", text)
    if m:
        info["layers"] = int(m.group(1))

    # Components
    for m in re.finditer(r"\* ([\w.]+): (\d+) modules per layer", text):
        info["components"].append(f"{m.group(1)} ({m.group(2)}/layer)")

    # Batch size tests
    for m in re.finditer(r"Trying batch size (\d+)\.\.\. Ok \((\d+) tokens/s\)", text):
        info["batch_tests"].append((int(m.group(1)), int(m.group(2))))
    m = re.search(r"Chosen batch size: (\d+)", text)
    if m:
        info["batch_size"] = int(m.group(1))

    # Response prefix
    m = re.search(r"Prefix found: (.+)", text)
    if m:
        info["prefix"] = m.group(1).strip()
    elif "None found" in text:
        info["prefix"] = "(none)"

    # Base refusals
    m = re.search(r"Base refusals: (\d+)/(\d+)", text)
    if m:
        info["base_refusals"] = f"{m.group(1)}/{m.group(2)}"

    # Trials
    for m in re.finditer(
        r"Running trial (\d+) of (\d+)", text
    ):
        trial_num = int(m.group(1))
        total = int(m.group(2))
        info["current_trial"] = trial_num
        info["total_trials"] = total

    # Parse trial results (refusals + KL divergence)
    for m in re.finditer(
        r"Trial\s+(\d+)\]\s+Refusals:\s+(\d+)/(\d+),\s+KL divergence:\s+([\d.]+)",
        text,
    ):
        info["trials"].append({
            "num": int(m.group(1)),
            "refusals": int(m.group(2)),
            "total_eval": int(m.group(3)),
            "kl_div": float(m.group(4)),
        })

    # Score lines from evaluation
    for m in re.finditer(
        r"Refusals: (\d+)/\d+.*?KL divergence: ([\d.]+)",
        text,
    ):
        pass  # Already captured above

    # Elapsed / remaining
    for m in re.finditer(r"Elapsed time: (.+)", text):
        info["elapsed"] = m.group(1).strip()
    for m in re.finditer(r"Estimated remaining time: (.+)", text):
        info["remaining"] = m.group(1).strip()

    # Memory
    for m in re.finditer(r"VRAM: (.+)", text):
        info["memory"] = m.group(1).strip()

    # Status
    if "Optimization finished" in text:
        info["status"] = "FINISHED"
    elif info["current_trial"] > 0:
        info["status"] = f"Trial {info['current_trial']}/{info['total_trials']}"
    elif "Calculating per-layer" in text:
        info["status"] = "Calculating refusal directions..."
    elif "Checking for common" in text:
        info["status"] = "Checking response prefix..."
    elif "Loading bad prompts" in text:
        info["status"] = "Loading prompts..."
    elif "Determining optimal" in text:
        info["status"] = "Testing batch sizes..."
    elif info["model_loaded"]:
        info["status"] = "Model loaded, initializing..."
    else:
        info["status"] = "Loading model..."

    return info


def display(info):
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 60)
    model = info['model_name'] or "Unknown Model"
    print(f"  HERETIC — {model} Abliteration Monitor")
    print("=" * 60)
    print()
    print(f"  GPU:          {info['gpu']}")
    print(f"  Quantization: {'4-bit (bnb)' if info['quantized'] else 'Loading...'}")
    print(f"  Layers:       {info['layers']}")
    print(f"  LoRA targets: {info['lora_targets']}")
    if info["components"]:
        print(f"  Components:   {', '.join(info['components'])}")
    print()

    if info["batch_tests"]:
        print("  Batch Size Tests:")
        for bs, tps in info["batch_tests"]:
            bar = "#" * (tps // 5)
            print(f"    BS {bs:>3}: {tps:>4} tok/s  {bar}")
        if info["batch_size"]:
            print(f"    -> Chosen: {info['batch_size']}")
        print()

    if info["prefix"]:
        print(f"  Response prefix: {info['prefix']}")
    if info["base_refusals"]:
        print(f"  Base refusals:   {info['base_refusals']}")
    print()

    # Trial progress
    print(f"  STATUS: {info['status']}")
    if info["current_trial"] > 0:
        pct = info["current_trial"] / info["total_trials"] * 100
        bar_len = 40
        filled = int(bar_len * info["current_trial"] / info["total_trials"])
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  [{bar}] {pct:.0f}%")
        print()
        if info["elapsed"]:
            print(f"  Elapsed:   {info['elapsed']}")
        if info["remaining"]:
            print(f"  Remaining: {info['remaining']}")
        if info["memory"]:
            print(f"  VRAM:      {info['memory']}")

    # Best trials so far
    if info["trials"]:
        print()
        print("  Best Trials (Pareto front):")
        print("  " + "-" * 45)
        # Sort by refusals then KL
        sorted_trials = sorted(info["trials"], key=lambda t: (t["refusals"], t["kl_div"]))
        # Pareto front
        min_kl = float("inf")
        pareto = []
        for t in sorted_trials:
            if t["kl_div"] < min_kl:
                min_kl = t["kl_div"]
                pareto.append(t)
        for t in pareto[:10]:
            print(f"    Trial {t['num']:>3}: Refusals {t['refusals']:>2}/{t['total_eval']}, KL {t['kl_div']:.4f}")

    print()
    print("  [Ctrl+C to stop monitoring]")


def main():
    print("Monitoring Heretic progress...")
    print(f"Log file: {LOG}")
    try:
        while True:
            info = parse_log()
            if info:
                display(info)
            else:
                print("Waiting for heretic_output.log to appear...")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
