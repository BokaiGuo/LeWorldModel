"""Deterministic fault-injection transport peer, not a simulator."""
import base64
import hashlib
import json
import socket
import subprocess
import sys
import time

fd, mode, pid_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
if pid_path != "-":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(pid_path, "x") as f:
        f.write(str(child.pid))
sock = socket.socket(fileno=fd)
stream = sock.makefile("rb")
count = 0
resets = 0
for line in stream:
    req = json.loads(line)
    if req["op"] == "reset":
        count = 0
        resets += 1
    else:
        if mode == "hang":
            time.sleep(20)
        if mode == "partial":
            sock.sendall(b'{"seq":')
            time.sleep(20)
        if mode == "exit":
            sys.exit(9)
        count += len(req["actions"])
    raw = bytes([count % 251]) * (224 * 224 * 3)
    res = {"seq": req["seq"], "step": count, "rgb": base64.b64encode(raw).decode(),
           "sha256": hashlib.sha256(raw).hexdigest()}
    if req["op"] == "reset" and resets > 1 and mode == "reset_clock":
        res["step"] = 1
    if req["op"] == "step":
        if mode == "clock":
            res["step"] += 1
        if mode == "sequence":
            res["seq"] += 1
        if mode == "hash":
            res["sha256"] = "0" * 64
        if mode == "extra":
            res["qpos"] = [1, 2]
    sock.sendall(json.dumps(res).encode() + b"\n")
