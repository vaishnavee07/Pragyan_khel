"""End-to-end WebSocket test for autofocus selective blur."""
import asyncio, json, sys

async def recv_next_frame(ws, timeout=8):
    """Receive messages until one contains a 'frame' field."""
    for _ in range(10):
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        msg = json.loads(raw)
        if msg.get("type"):          # protocol ack — skip
            print(f"  (skipped ack: type={msg['type']})")
            continue
        if "frame" in msg:
            return msg
    raise AssertionError("No frame message received in 10 attempts")


async def test():
    try:
        import websockets
    except ImportError:
        print("[SKIP] websockets not installed — run: pip install websockets")
        return

    uri = "ws://localhost:8000/ws/video"
    print(f"Connecting to {uri} ...")
    async with websockets.connect(uri) as ws:

        # 1) First frame
        msg0 = await recv_next_frame(ws)
        print(f"[OK] Frame 1  mode={msg0['mode']}  fps={msg0['fps']}")

        # 2) Switch to autofocus mode
        await ws.send(json.dumps({"type": "switch_mode", "mode": "autofocus"}))
        print("     → switch_mode:autofocus sent")

        # 3) Drain until we get a frame with mode=autofocus
        for attempt in range(20):
            raw = await asyncio.wait_for(ws.recv(), timeout=8)
            msg = json.loads(raw)
            if msg.get("type") == "mode_switched":
                print(f"[OK] Mode switch ack  success={msg.get('success')}  mode={msg.get('mode')}")
                continue
            if "frame" in msg:
                print(f"[OK] Post-switch frame  mode={msg['mode']}")
                break
        else:
            raise AssertionError("Never received frame after mode switch")

        # 4) Send click at centre (320, 240)
        await ws.send(json.dumps({"type": "autofocus_click", "x": 320, "y": 240}))
        print("     → autofocus_click (320,240) sent")

        for attempt in range(10):
            raw = await asyncio.wait_for(ws.recv(), timeout=8)
            msg = json.loads(raw)
            if msg.get("type") == "autofocus_ack":
                print(f"[OK] Click ack  x={msg['x']}  y={msg['y']}")
                break

        # 5) Receive a few blurred frames and decode one
        import base64, numpy as np, cv2
        for _ in range(5):
            frame_msg = await recv_next_frame(ws)

        raw_jpg = base64.b64decode(frame_msg["frame"])
        nparr   = np.frombuffer(raw_jpg, np.uint8)
        img     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img is not None and img.shape[2] == 3
        print(f"[OK] Blurred frame decoded  shape={img.shape}")

        print()
        print("=== END-TO-END TEST PASSED ===")

asyncio.run(test())

