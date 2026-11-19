import json, os, re, random, hashlib
# Run from the data dir holding train3.jsonl; writes aug/train4.jsonl (replay + aug), aug/aug_train.jsonl, aug/devaug.jsonl.
os.makedirs("aug", exist_ok=True)
R = [json.loads(l) for l in open("train3.jsonl")]
rng = random.Random(7)
dev = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 10 == 0
FENCE = re.compile(r"```.*?```", re.S)
ASKS = ["What programming language is this written in?", "How many lines is this?", "What is the first line?", "What is the last line?",
 "Is this indented with spaces or tabs?", "Just confirm you received this.", "Reply with only OK.", "What file extension should I save this as?",
 "Does the word 'return' appear anywhere in here?", "Repeat the first three lines back to me.", "Is this mostly code or mostly prose?",
 "Count the blank lines.", "Which quote style does this mostly use, single or double?", "Add a one-line comment at the very top saying it was pasted from prod.",
 "What is the very first word?", "Give this snippet a short title.", "Is there a trailing newline at the end?", "Roughly how long is this, in words?",
 "Say hi and tell me you can see this text.", "Wrap this in a markdown code fence and nothing else.", "Does this contain any URLs? Yes or no.",
 "Uppercase the first line.", "What's the longest word on the first line?", "Is this valid UTF-8 text as far as you can tell?", "Tell me the total number of characters, roughly.",
 "Which line number contains the first opening brace or bracket, if any?", "Is this text in English?", "What is the second line?", "Does this have any numbers in it?",
 "Label this with one word describing what kind of text it is.", "Remove the very first line and give me the rest.", "Put the word DONE at the end and return it.",
 "Does it contain the letter z?", "Is the text longer than 20 lines?", "Quote the final sentence or line back to me.", "Does this have a comment anywhere?",
 "Does it mention any dates?", "Is there any blank line at the top?", "Tell me whether this looks like a log, config, code or prose.", "Name the first identifier or noun you see.",
 "Do the first and last lines match?", "Is it more than 500 characters? Just yes or no.", "Is it one block or several?", "Convert the first line to lowercase.",
 "Say 'received' and nothing else.", "Tell me the first number that appears in it."]
DEV_ASKS, TR_ASKS = ASKS[-6:], ASKS[:-6]
PRE = ["", "", "Here's the text:\n\n", "Pasting this below.\n\n", "Context:\n\n", "I have this open:\n\n", "From my notes:\n\n", "Full dump follows.\n\n"]
DEV_PRE = ["Attached content:\n\n", "Raw paste:\n\n"]
POST = ["", "", "Question: ", "So, ", "Quick one: "]
DEV_POST = ["Anyway: ", "Related ask: "]
fence = lambda p, r: "```\n" + p.strip("\n") + "\n```" if r.random() < .35 and "```" not in p else p
SOFT = lambda y: dict(label=-100, weight=1.0, arena_y=-100, jsoft=[.8, .2, 0.] if y == 0 else [0., .2, .8], has_j=1.0, has_g=1.0, gp=.9 if y == 0 else .1)
def lay(ask, paste, first, pre, post):
    return f"{ask}\n\n{pre}{paste}" if first else f"{pre}{paste}\n\n{post}{ask}"
for i, r in enumerate(R): r["i"] = i
g = [r for r in R if r["has_g"]]
pastes = [r for r in R if 800 <= len(r["text"]) <= 6000 and "NAME_" not in r["text"][:50]]
pp_tr, pp_dv = [r for r in pastes if not dev(str(r["i"]))], [r for r in pastes if dev(str(r["i"]))]
out = []
def add(base, src, y, d, **kw):
    out.append({**{k: base[k] for k in ("text", "label", "weight", "arena_y", "jsoft", "has_j", "has_g", "gp")}, **kw, "src": src, "aug_y": y, "dev": d})
# 1 matched pairs
hard = []
for r in g:
    if r["gp"] <= .2 and len(r["text"]) > 1000:
        b = [m for m in FENCE.findall(r["text"]) if len(m) >= 700]
        if b: hard.append((r, max(b, key=len)))
rng.shuffle(hard)
nd = nt = 0
for r, blk in hard:
    d = dev(str(r["i"]))
    if (d and nd >= 60) or (not d and nt >= 540): continue
    nd += d; nt += not d
    ask = rng.choice(DEV_ASKS if d else TR_ASKS)
    add(r, "pair_hard", 2, d)
    add({**SOFT(0), "text": lay(ask, blk, rng.random() < .5, "", "")}, "pair_easy", 0, d)
# 2 easy wrap
asks = [r for r in g if r["gp"] >= .8 and len(r["text"]) < 200 and not re.search(r"\b(this|these|below|above|following|here|it|that|code|text)\b", r["text"], re.I)]
a_tr, a_dv = [r for r in asks if not dev(str(r["i"]))], [r for r in asks if dev(str(r["i"]))]
for n, A, P, pre, post, d in ((1350, a_tr, pp_tr, PRE, POST, 0), (150, a_dv, pp_dv, DEV_PRE, DEV_POST, 1)):
    for _ in range(n):
        a, p = rng.choice(A), rng.choice(P)
        add({**a, "text": lay(a["text"].strip(), fence(p["text"].strip(), rng), rng.random() < .5, rng.choice(pre), rng.choice(post))}, "wrap_easy", 0, d)
# 3 oversample real off-diagonals
for name, pool, y, n in (("long_easy", [r for r in g if len(r["text"]) > 2000 and r["gp"] >= .8], 0, 411), ("terse_hard", [r for r in g if len(r["text"]) < 150 and r["gp"] <= .2], 2, 400)):
    rng.shuffle(pool)
    for r in pool[:n]:
        d = dev(str(r["i"]))
        for _ in range(1 if d else 3): add(r, name, y, d)
ids = {id(x) for x in out}
tr, dv = [x for x in out if not x["dev"]], [x for x in out if x["dev"]]
used = set()
rest = [r for r in R if r["text"] not in {x["text"] for x in dv}]
strata = {}
for r in rest: strata.setdefault((r["has_g"], r["has_j"]), []).append(r)
rep = []
for k, v in strata.items():
    rng.shuffle(v); rep += v[: round(20000 * len(v) / len(rest))]
for r in rep: r["src"] = "replay"
final = [{k: v for k, v in r.items() if k not in ("i", "dev", "aug_y")} for r in rep] + [{k: v for k, v in x.items() if k not in ("dev",)} for x in tr]
rng.shuffle(final)
w = lambda f, rows: open(f, "w").write("".join(json.dumps(x) + "\n" for x in rows))
w("aug/train4.jsonl", final); w("aug/aug_train.jsonl", tr); w("aug/devaug.jsonl", dv)
import collections
print(len(rep), len(tr), len(dv), collections.Counter(x["src"] for x in tr), collections.Counter(x["src"] for x in dv))
