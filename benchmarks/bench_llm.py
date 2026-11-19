"""Compare switchyard-decider with OpenRouter chat and decision models on the bench_dec.py subset.

bench_llm.py run <model...>    zero-shot chat-completion routers (N, default 120)
bench_llm.py dec <model...>    decision models via /alpha/decisions, with latency and cost
bench_llm.py ours [url]        the served switchyard-decider (N, default 300), sequential
bench_llm.py compare           tables from dec_bench/ and llm_bench/ -> results/openrouter_compare.json
bench_llm.py plot              docs/pareto.png from that json (needs matplotlib)

Run from the data directory (val.jsonl). Paid models are refused unless ALLOW_PAID=1; SPEND_CAP (dollars, default 1)
stops when key usage rises that much above BASE_USAGE (default: usage at start), FREE_RESERVE keeps that many free-model requests of the daily quota unused.
"""
import json, os, re, sys, random, time, threading, urllib.request, urllib.error, concurrent.futures as cf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_dec import Q, auc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = {"Authorization": "Bearer " + os.environ.get("OPENROUTER_API_KEY", ""), "Content-Type": "application/json"}
API = "https://openrouter.ai/api"
N = int(os.environ.get("N", 0))
MAX_CALLS, RESERVE = int(os.environ.get("MAX_CALLS", 900)), int(os.environ.get("FREE_RESERVE", 20))
SPEND_CAP, TRUNC = float(os.environ.get("SPEND_CAP", 1)), int(os.environ.get("TRUNC", 12000))
MAX_TOKENS, EFFORT = int(os.environ.get("MAX_TOKENS", 2048)), os.environ.get("REASONING", "low")
OURS = "switchyard-decider v1.1"
PROMPT = ("You route requests for a coding assistant. " + Q["solve"]["instructions"] + "\n\nEstimate the probability "
          "(0-100) that the request below needs a frontier-level model. Reply with only JSON: {\"p_frontier\": <0-100>}"
          "\n\n<request>\n%s\n</request>")
lock, state = threading.Lock(), {"calls": 0, "stop": None}


def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60))


def key():
    return get(API + "/v1/key")["data"]


def subset(va, n):
    idx = [i for i, r in enumerate(va) if r["label"] in (0, 2)]
    random.Random(0).shuffle(idx)
    return sorted(idx[:n])


def fname(m):
    return f"llm_bench/{m.replace('/', '_').replace('~', '')}.json"


def guard(m, start):
    k = key()
    if k["usage"] - start >= SPEND_CAP:
        state["stop"] = f"spend cap: usage {k['usage']:.4f} vs start {start:.4f}"
    elif m.endswith(":free") and k["free_model_daily_requests"]["remaining"] <= RESERVE:
        state["stop"] = f"free quota: {k['free_model_daily_requests']}"


def parse(c):
    hits = re.findall(r'p_frontier"?\s*[:=]\s*"?(\d+(?:\.\d+)?)', c) or re.findall(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$", c)
    return min(100.0, float(hits[-1])) if hits else None


def chat(m, t):
    body = {"model": m, "messages": [{"role": "user", "content": PROMPT % t[:TRUNC]}], "temperature": 0,
            "max_tokens": MAX_TOKENS, "reasoning": {"effort": EFFORT}, "usage": {"include": True}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions",
                                                                json.dumps(body).encode(), H), timeout=300))
    if "error" in r:
        raise RuntimeError(str(r["error"])[:120])
    c, u = r["choices"][0]["message"].get("content") or "", r.get("usage") or {}
    p = parse(c)
    return {"s": None if p is None else 1 - p / 100, "p": p, "tok_in": u.get("prompt_tokens"),
            "tok_out": u.get("completion_tokens"), "cost": u.get("cost", 0), "raw": None if p is not None else c[-200:]}


def decide(m, t):
    body = {"model": m, "state": {"opening_task": t[:TRUNC], "latest_follow_up": ""}, "questions": Q}
    r = json.load(urllib.request.urlopen(urllib.request.Request(API + "/alpha/decisions", json.dumps(body).encode(), H),
                                         timeout=120))
    u = r.get("usage") or {}
    return {"s": r["answers"]["solve"]["noul"], "tok_in": u.get("prompt_tokens"), "tok_out": u.get("completion_tokens"),
            "cost": u.get("cost", 0)}


def call(fn, m, t, start):
    err = None
    for k in range(5):
        with lock:
            if state["stop"] or state["calls"] >= MAX_CALLS:
                state["stop"] = state["stop"] or "max calls"
                return {"s": None, "err": state["stop"]}
            state["calls"] += 1
            if state["calls"] % int(os.environ.get("GUARD_EVERY", 20)) == 0:
                guard(m, start)
        t0 = time.perf_counter()
        try:
            return fn(m, t) | {"ms": (time.perf_counter() - t0) * 1000, "tries": k + 1}
        except urllib.error.HTTPError as e:
            err = e.code
            if err not in (408, 429, 500, 502, 503, 504):
                break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:80]}"
        time.sleep(min(60, 5 * 2 ** k))
    return {"s": None, "err": str(err)}


def run(fn, models, n):
    va = [json.loads(line) for line in open("val.jsonl")]
    idx, start = subset(va, n), float(os.environ.get("BASE_USAGE") or key()["usage"])
    os.makedirs("llm_bench", exist_ok=True)
    for m in models:
        if not m.endswith(":free") and os.environ.get("ALLOW_PAID") != "1":
            print(m, "refused: not a :free model (set ALLOW_PAID=1)", flush=True)
            continue
        guard(m, start)
        if state["stop"]:
            break
        f = fname(m)
        res = {int(k): v for k, v in json.load(open(f)).items()} if os.path.exists(f) else {}
        todo = [i for i in idx if i not in res or (res[i]["s"] is None and os.environ.get("RETRY_FAILED") == "1")]
        with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
            res |= dict(zip(todo, ex.map(lambda i: call(fn, m, va[i]["text"], start), todo)))
        res = {i: v for i, v in res.items() if v.get("err") not in ("max calls",) and not str(v.get("err")).startswith(
            ("spend cap", "free quota"))}
        json.dump(res, open(f, "w"))
        ok = [i for i in idx if i in res and res[i]["s"] is not None]
        y = np.array([va[i]["label"] for i in ok])
        print(m, json.dumps({"n": len(ok), "of": len(idx), "auc": round(auc(np.array([res[i]["s"] for i in ok]), y), 4)
                             if len(set(y)) == 2 else None, "calls": state["calls"]}), flush=True)
    k = key()
    print("usage", round(k["usage"], 6), "start", round(start, 6), "free", k["free_model_daily_requests"], state["stop"])


def ours(url, n):
    va = [json.loads(line) for line in open("val.jsonl")]
    res = {}
    for i in subset(va, n):
        body = json.dumps({"state": {"opening_task": va[i]["text"][:TRUNC], "latest_follow_up": ""}}).encode()
        t0 = time.perf_counter()
        r = json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type": "application/json"})))
        res[i] = {"s": r["answers"]["solve"]["noul"], "ms": (time.perf_counter() - t0) * 1000, "cost": 0}
    os.makedirs("llm_bench", exist_ok=True)
    json.dump(res, open(fname("ours"), "w"))
    ms = [v["ms"] for v in res.values()]
    print(OURS, len(res), "p50", round(np.median(ms), 1), "p95", round(np.percentile(ms, 95), 1))


def metrics(sc, va, idx, rng):
    ok = [i for i in idx if sc.get(i) is not None]
    s, y = np.array([sc[i] for i in ok]), np.array([va[i]["label"] for i in ok])
    code = np.array([va[i]["domain"] == "coding" for i in ok])
    boot = []
    for _ in range(2000):
        b = rng.integers(0, len(ok), len(ok))
        if len(set(y[b])) == 2:
            boot.append(auc(s[b], y[b]))
    # Exactly 40% to the frontier, by rank with random tie-breaks so discrete scores are not penalised.
    order = np.lexsort((rng.random(len(s)), s))
    front = np.zeros(len(s), bool)
    front[order[:round(0.4 * len(s))]] = True
    return {"n": len(ok), "failures": len(idx) - len(ok), "auc": round(auc(s, y), 4),
            "auc_ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)],
            "auc_coding": round(auc(s[code], y[code]), 4), "hard_to_frontier@40": round(float(front[y == 2].mean()), 3),
            "easy_to_mid@40": round(float((~front)[y == 0].mean()), 3)}


def compare():
    va = [json.loads(line) for line in open("val.jsonl")]
    n_small, n_big = N or 120, 300
    small, big = subset(va, n_small), subset(va, n_big)
    try:
        listed = {m["id"]: m for m in get(API + "/v1/models?output_modalities=all")["data"]}
    except Exception:
        listed = {}
    names = {fname(m)[10:]: m for m in listed}
    rows = []

    def add(name, kind, sc, meas, where):
        rng = np.random.default_rng(0)
        r = {"model": name, "kind": kind, "where": where, "params": None, **metrics(sc, va, small, rng)}
        if all(i in sc for i in big):
            r["auc_300"] = round(auc(np.array([sc[i] for i in big if sc[i] is not None]),
                                     np.array([va[i]["label"] for i in big if sc[i] is not None])), 4)
        if meas:
            v = [meas[i] for i in small if i in meas]
            ms = [x["ms"] for x in v if x.get("ms") and x["s"] is not None]
            r["p50_ms"], r["p95_ms"] = round(float(np.median(ms)), 1), round(float(np.percentile(ms, 95)), 1)
            r["tokens_per_decision"] = round(float(np.mean([(x.get("tok_in") or 0) + (x.get("tok_out") or 0)
                                                            for x in v if x["s"] is not None])), 1)
            r["usd_per_1m"] = round(sum(x.get("cost") or 0 for x in v) / max(1, sum(x["s"] is not None for x in v)) * 1e6, 2)
        p = re.search(r"(\d+(?:\.\d+)?)b(?:-a(\d+)b)?\b", name.split("/")[-1])
        r["params"] = "395M" if kind == "ours" else (p and (p[1] + "B" + (f" ({p[2]}B active)" if p[2] else "")))
        rows.append(r)

    meas = {f[:-5]: {int(k): v for k, v in json.load(open("llm_bench/" + f)).items()}
            for f in sorted(os.listdir("llm_bench")) if f.endswith(".json")}
    ours = meas.pop("ours")
    add(OURS, "ours", {i: v["s"] for i, v in ours.items()}, ours, "local, Apple M4 Max (MPS)")
    for f in sorted(os.listdir("dec_bench")):
        d = {int(k): v for k, v in json.load(open("dec_bench/" + f)).items()}
        sc = {i: v if isinstance(v, float) else None for i, v in d.items()}
        if sum(v is not None for v in sc.values()) >= 30:
            add(names.get(f, f[:-5]), "decision", sc, meas.pop(f[:-5], None), "OpenRouter API")
    excluded = {}
    for f, d in meas.items():
        if sum(d.get(i, {}).get("s") is not None for i in small) < 0.9 * len(small):
            excluded[names.get(f + ".json", f)] = f"{len(d)} prompts run, errors: {sorted({str(v.get('err')) for v in d.values()})}"
            continue
        add(names.get(f + ".json", f), "chat", {i: v["s"] for i, v in d.items()}, d, "OpenRouter API")
    rows.sort(key=lambda r: -r["auc"])
    os.makedirs(f"{ROOT}/benchmarks/results", exist_ok=True)
    out = {"subset": {"n": n_small, "n_large": n_big, "labels": "easy (0) vs hard (2) from val.jsonl",
                      "construction": "random.Random(0).shuffle of label in {0,2} indices, first n"},
           "prompt": PROMPT % "<text>", "reasoning_effort": EFFORT, "max_tokens": MAX_TOKENS, "models": rows,
           "excluded": excluded}
    json.dump(out, open(f"{ROOT}/benchmarks/results/openrouter_compare.json", "w"), indent=1)
    table(rows)


def table(rows):
    f = lambda v, d="n/a": d if v is None else v
    base = next(r for r in rows if r["kind"] == "ours")
    print("| model | kind | params | runs on | p50 ms | p95 ms | tokens/decision | $ / 1M decisions | p50 vs ours | AUC (95% CI) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "p50_ms" in r:
            x = "1x" if r is base else f"{r['p50_ms'] / base['p50_ms']:.0f}x slower"
            print(f"| {r['model']} | {r['kind']} | {f(r['params'])} | {r['where']} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | "
                  f"{f(r.get('tokens_per_decision'))} | {r['usd_per_1m']:.2f} | {x} | {r['auc']:.3f} ({r['auc_ci95'][0]:.3f}-{r['auc_ci95'][1]:.3f}) |")
    print()
    print("| model | kind | n | failures | AUC (95% CI) | AUC coding | hard → frontier @40% | easy → mid-tier @40% | AUC on 300 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['kind']} | {r['n']} | {r['failures']} | {r['auc']:.3f} ({r['auc_ci95'][0]:.3f}-"
              f"{r['auc_ci95'][1]:.3f}) | {r['auc_coding']:.3f} | {r['hard_to_frontier@40']:.3f} | {r['easy_to_mid@40']:.3f} | "
              f"{f(r.get('auc_300'), '')} |")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in json.load(open(f"{ROOT}/benchmarks/results/openrouter_compare.json"))["models"] if "p50_ms" in r]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    col = {"ours": "tab:red", "chat": "tab:blue", "decision": "tab:green"}
    for a, k, lab in ((ax[0], "p50_ms", "p50 latency per decision (ms, log)"),
                      (ax[1], "usd_per_1m", "USD per 1M decisions (log; free plotted at 0.1)")):
        for r in rows:
            x = max(r[k], 0.1)
            a.errorbar(x, r["auc"], yerr=[[r["auc"] - r["auc_ci95"][0]], [r["auc_ci95"][1] - r["auc"]]], fmt="o",
                       color=col[r["kind"]], ms=8 if r["kind"] == "ours" else 5, alpha=.8, capsize=2)
            a.annotate(r["model"].split("/")[-1].replace(":free", " (free)"), (x, r["auc"]), fontsize=7,
                       xytext=(4, 3), textcoords="offset points")
        a.set_xscale("log"); a.set_xlabel(lab); a.set_ylabel("AUC easy vs hard (n=120, 95% CI)"); a.grid(alpha=.3)
    for k, c in col.items():
        ax[0].plot([], [], "o", color=c, label={"ours": "switchyard-decider (local)", "chat": "chat LLM, zero-shot",
                                                "decision": "OpenRouter decision model"}[k])
    ax[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/docs/pareto.png", dpi=130)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        run(chat, args, N or 120)
    elif cmd == "dec":
        run(decide, args, N or 120)
    elif cmd == "ours":
        ours(args[0] if args else "http://127.0.0.1:4126/v1/systemone", N or 300)
    elif cmd == "compare":
        compare()
    elif cmd == "plot":
        plot()
