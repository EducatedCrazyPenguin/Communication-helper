"""Test-only pipe peer; never used by production provider selection."""
import os
import sys
import threading
import time

from communication_gym.worker.protocol import encode_frame, read_frame

lock = threading.Lock()
events = {}


def send(request_id, op, payload):
    with lock:
        sys.stdout.buffer.write(encode_frame({"version": 1, "request_id": request_id, "op": op, "payload": payload}))
        sys.stdout.buffer.flush()


def execute(message, event):
    request_id, operation = message["request_id"], message["op"]
    if operation == "crash":
        os._exit(4)
    if operation in {"transcribe", "progress_forever", "silence", "ignore_cancel"}:
        while not event.wait(0.005) or operation == "ignore_cancel":
            if operation == "progress_forever":
                send(request_id, "progress", {"phase": "transcribing", "progress": 0.5})
            if operation == "ignore_cancel":
                time.sleep(0.01)
        send(request_id, "cancelled", {})
    else:
        send(request_id, "result", {"released": True})
    events.pop(request_id, None)


send("startup", "ready", {"protocol_version": 1, "pid": os.getpid()})
while True:
    message = read_frame(sys.stdin.buffer)
    if message is None:
        break
    if message["op"] == "cancel":
        event = events.get(message["payload"].get("request_id"))
        if event:
            event.set()
        continue
    event = threading.Event()
    events[message["request_id"]] = event
    threading.Thread(target=execute, args=(message, event), daemon=True).start()
