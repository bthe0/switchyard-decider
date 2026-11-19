import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MAX_LEN = 768
TASK_CHARS = 3000


def request_text(state):
    return "\n\n".join(x for x in (state.get("opening_task", "")[:TASK_CHARS], state.get("latest_follow_up", "")) if x)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def hardness(logits, calib):
    # Mean of two validation percentiles: P(hard) over arena+judge heads, and the negated gold logit (higher = easier).
    lg = np.asarray(logits, dtype=np.float32)
    pr = (softmax(lg[:3]) + softmax(lg[3:6])) / 2
    aj, g = np.searchsorted(calib["aj"], pr[2].item()), np.searchsorted(calib["g"], -lg[6].item())
    return (aj + g) / 2 / len(calib["aj"])


def p_solve(h, calib):
    q = np.searchsorted(calib["h"], h) / len(calib["h"])
    return float(np.quantile(calib["out"], 1 - q))


class Handler(BaseHTTPRequestHandler):
    score = None

    def do_POST(self):
        s = json.loads(self.rfile.read(int(self.headers["content-length"])))["state"]
        body = json.dumps({"answers": {"solve": {"noul": self.score(request_text(s))}}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def load_scorer(model_dir, dev=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = dev or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    dtype = torch.float16 if dev != "cpu" else torch.float32
    m = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=dtype).to(dev).eval()
    calib = np.load(f"{model_dir}/calib.npz")
    lock = threading.Lock()

    def score(text):
        with lock, torch.no_grad():
            x = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
            lg = m(**x).logits[0].float().cpu()
        return p_solve(hardness(lg.numpy(), calib), calib)

    return score, dev


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve switchyard-decider over HTTP.")
    ap.add_argument("model_dir")
    ap.add_argument("--port", type=int, default=4126)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    score, dev = load_scorer(a.model_dir)
    score("warmup")
    Handler.score = staticmethod(score)
    print(f"serving {a.model_dir} on {a.host}:{a.port} ({dev})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
