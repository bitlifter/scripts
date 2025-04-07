import subprocess
import re
import itertools

# ----------------------
# CONFIGURATION
# ----------------------
threads_options = [8, 16, 32, 64]  # Example thread counts
precision_options = ['fp32', 'fp16', 'int8']  # Example precision modes
use_gpu_options = [True, False]  # Test with and without GPU acceleration

# Add a list of available LLM models (adjust according to your use-case)
models = [
    "modelA.bin", 
    "modelB.bin"
]  # Replace these with the actual model identifiers or file paths.

# Path to your LM Studio executable.
lmstudio_exe = "lmstudio.exe"

# ----------------------
# AUTOTUNING FUNCTION
# ----------------------
def run_benchmark(model, threads, precision, use_gpu):
    """
    Runs LM Studio in benchmark mode with the specified settings.
    Assumes that LM Studio outputs a line like "Average tokens/sec: X.XX" in its stdout.
    Returns the tokens/sec as a float if successful; otherwise, returns None.
    """
    # Build the command-line arguments.
    cmd = [lmstudio_exe, "--benchmark", "--model", model,
           "--threads", str(threads), "--precision", precision]
    if use_gpu:
        cmd.append("--use-gpu")  # Enable GPU acceleration if flag is available

    print(f"Running benchmark: model={model}, threads={threads}, precision={precision}, use_gpu={use_gpu}")
    
    try:
        # Run the command and capture the output.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"Timeout for configuration: model={model}, threads={threads}, precision={precision}, use_gpu={use_gpu}")
        return None

    output = result.stdout
    # Example: Look for a line in the output like "Average tokens/sec: 123.45"
    match = re.search(r"Average tokens/sec:\s*([\d\.]+)", output)
    if match:
        tokens_sec = float(match.group(1))
        print(f"Result: {tokens_sec} tokens/sec")
        return tokens_sec
    else:
        print("Could not parse tokens/sec from output:")
        print(output)
        return None

# ----------------------
# MAIN AUTOTUNING LOOP
# ----------------------
def autotune():
    results = []
    
    # Iterate over all combinations of models and the performance parameters.
    for model, threads, precision, use_gpu in itertools.product(models, threads_options, precision_options, use_gpu_options):
        tokens_sec = run_benchmark(model, threads, precision, use_gpu)
        if tokens_sec is not None:
            results.append({
                'model': model,
                'threads': threads,
                'precision': precision,
                'use_gpu': use_gpu,
                'tokens_sec': tokens_sec
            })
    
    if results:
        # Find the configuration with the highest tokens/sec
        best_config = max(results, key=lambda x: x['tokens_sec'])
        print("\n--- Best Configuration Found ---")
        print(f"Model: {best_config['model']}")
        print(f"Threads: {best_config['threads']}")
        print(f"Precision: {best_config['precision']}")
        print(f"Use GPU: {best_config['use_gpu']}")
        print(f"Tokens/sec: {best_config['tokens_sec']}")
    else:
        print("No valid results obtained from benchmarking.")

if __name__ == "__main__":
    autotune()
