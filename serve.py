import json, sys, threading, numpy as np, torch
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

d, port = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4126
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev).eval()
c = np.load(f"{d}/calib.npz")
lock = threading.Lock()


def hardness(text):
    with lock, torch.no_grad():
        lg = m(**tok(text, truncation=True, max_length=768, return_tensors="pt").to(dev)).logits[0].float().cpu()
    pr = (lg[:3].softmax(-1) + lg[3:6].softmax(-1)) / 2
    return (np.searchsorted(c["aj"], pr[2].item()) + np.searchsorted(c["g"], -lg[6].item())) / 2 / len(c["aj"])


def p_solve(text):
    # Hardness percentile on the validation set, mapped onto the target output distribution in calib.npz.
    q = np.searchsorted(c["h"], hardness(text)) / len(c["h"])
    return float(np.quantile(c["out"], 1 - q))


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        s = json.loads(self.rfile.read(int(self.headers["content-length"])))["state"]
        text = "\n\n".join(x for x in (s.get("opening_task", "")[:3000], s.get("latest_follow_up", "")) if x)
        body = json.dumps({"answers": {"solve": {"noul": p_solve(text)}}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


p_solve("warmup")
print(f"serving {d} on 127.0.0.1:{port} ({dev})", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
