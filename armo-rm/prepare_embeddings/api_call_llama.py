import subprocess
import time
import argparse
import signal
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_index", type=int, default=0)
parser.add_argument("--num_batches", type=int, default=10)
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# === Paths ===
model_path = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/"
labeling_script = "data_generation_vllm_llama.py"

def main():

    server_log_path = f"/cluster/work/vogtlab/Group/slaguna/logs/server_llama_log_batch_{args.batch_index}.txt"
    labeling_log_path = f"/cluster/work/vogtlab/Group/slaguna/logs/labeling_llama_log_batch_{args.batch_index}.txt"

    server_proc = None

    try:
        # 1. Start vLLM API server
        print("Starting vLLM server...")
        server_log = open(server_log_path, "w")
        server_proc = subprocess.Popen([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            # "--tensor-parallel-size=1",
            # "--enforce-eager",
            # "--disable-custom-all-reduce",
            "--host", "localhost",
            "--port", str(args.port)
        ], stdout=server_log, stderr=server_log)
        time.sleep(100)  # Wait for server to be up

        # 2. Run labeling script
        print("Starting labeling script...")
        with open(labeling_log_path, "w") as labeling_log:
            subprocess.run([
                "python", labeling_script,
                "--batch_index", str(args.batch_index),
                "--num_batches", str(args.num_batches),
                "--port", str(args.port)
                # "--system_prompt", "You are an AI assistant that helps scoring a system response."
            ], check=True, stdout=labeling_log, stderr=labeling_log)

    finally:
        print("Shutting down server...")
        server_proc.send_signal(signal.SIGINT)  # Or server_proc.terminate()
        server_proc.wait()
        server_log.close()
        print("All done!")

if __name__ == "__main__":
    main()
