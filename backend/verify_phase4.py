"""Quick verification that all Phase 4 todos are done."""
import pathlib, sys

ROOT = pathlib.Path(__file__).parent

def chk(label, cond, msg=""):
    if cond:
        print(f"[OK] {label}")
    else:
        print(f"[FAIL] {label}: {msg}")
        sys.exit(1)

# ── Imports ──────────────────────────────────────────────────────────
from core.depth_engine    import DepthEngine;     chk("depth_engine import", True)
from core.autofocus_engine import AutofocusEngine; chk("autofocus_engine import", True)
from core.blur_compositor  import BlurCompositor;  chk("blur_compositor import", True)
from modules.autofocus_module import AutofocusModule; chk("autofocus_module import", True)
from api.websocket_handler    import WebSocketHandler; chk("websocket_handler import", True)

# ── WebSocket routing ─────────────────────────────────────────────
src_ws = (ROOT / "api/websocket_handler.py").read_text()
chk("_handle_message",         "_handle_message"       in src_ws)
chk("autofocus_click route",   "autofocus_click"       in src_ws)
chk("autofocus_double_click",  "autofocus_double_click" in src_ws)
chk("autofocus_config route",  "autofocus_config"      in src_ws)

# ── main.py wiring ───────────────────────────────────────────────
src_main = (ROOT / "main.py").read_text()
chk("AutofocusModule imported",  "AutofocusModule"     in src_main)
chk("register_module autofocus", "register_module"     in src_main)

# ── Frontend VideoFeed ────────────────────────────────────────────
vf_path = ROOT.parent / "hackathon-frontend/src/components/VideoFeed.jsx"
chk("VideoFeed.jsx exists", vf_path.exists())
vf = vf_path.read_text()
chk("isAutofocusActive prop",    "isAutofocusActive"     in vf)
chk("autofocus_click send",       "autofocus_click"       in vf)
chk("autofocus_double_click send","autofocus_double_click" in vf)
chk("focusRing animation",        "focusRing"             in vf)
chk("Cinematic Mode badge",       "Cinematic Mode Active" in vf)

# ── AutofocusPanel ────────────────────────────────────────────────
ap_path = ROOT.parent / "hackathon-frontend/src/components/AutofocusPanel.jsx"
chk("AutofocusPanel.jsx exists",  ap_path.exists())

print()
print("=" * 46)
print("  ALL PHASE 4 TODOS — COMPLETE & VERIFIED  ")
print("=" * 46)
