import json,gzip,glob,hashlib,random,re,sys
import pyarrow.parquet as pq
code = re.compile(r"```|\bdef |\bfunction\b|\bimport |#include|\bSELECT\b|\bclass \w+|"
                  r"\b(python|javascript|typescript|java|c\+\+|golang|rust|sql|bash|regex|"
                  r"html|css|react|api|json|compile|traceback|exception|stack trace|code|script)\b", re.I)
key=lambda t:hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()
ex=set()
for f in ("val.jsonl","train2.jsonl"):
    for l in open(f): ex.add(key(json.loads(l)["text"]))
def ok(t):
    if not 40<=len(t)<=6000: return False
    return sum(c.isascii() for c in t)/len(t)>0.9
E="ext_data/"
def txt(c):
    if isinstance(c,str): return c
    if isinstance(c,list): return "\n".join(x.get("text","") if isinstance(x,dict) else str(x) for x in c)
    return ""
def wild():
    for f in sorted(glob.glob(E+"WildChat-1M/data/*.parquet")):
        for r in pq.read_table(f,columns=["conversation_hash","conversation","language","toxic","redacted"]).to_pylist():
            if r["language"]=="English" and not r["toxic"] and not r["redacted"] and r["conversation"]:
                yield r["conversation_hash"],r["conversation"][0]["content"] or ""
def arena():
    for f in sorted(glob.glob(E+"arena-human-preference-140k/data/*.parquet")):
        for r in pq.read_table(f,columns=["id","conversation_a","language"]).to_pylist():
            if r["language"]=="en" and r["conversation_a"]:
                yield r["id"],txt(r["conversation_a"][0]["content"])
def swe():
    for n,f in [("SWE-bench","SWE-bench/data/*.parquet"),("SWE-rebench","SWE-rebench/data/*.parquet"),("SWE-Gym","SWE-Gym/data/*.parquet")]:
        for g in sorted(glob.glob(E+f)):
            for r in pq.read_table(g,columns=["instance_id","problem_statement"]).to_pylist():
                yield n,r["instance_id"],r["problem_statement"] or ""
def oasst():
    for i,l in enumerate(gzip.open(E+"oasst2/2023-11-05_oasst2_ready.messages.jsonl.gz","rt")):
        r=json.loads(l)
        if r["role"]=="prompter" and r["parent_id"] is None and r["lang"]=="en" and not r["deleted"]: yield r["message_id"],r["text"]
def jl(p,k):
    for i,l in enumerate(open(E+p)): yield str(i),json.loads(l)[k]
seen=set(); out=[]
def add(src,lic,it,cap,seed=0,cq=None):
    c=[]
    for i,t in it:
        t=(t or "").strip()
        if not ok(t): continue
        k=key(t)
        if k in ex or k in seen: continue
        seen.add(k); c.append({"id":f"{src}:{i}","source":src,"license":lic,"text":t})
    random.Random(seed).shuffle(c)
    if cq is not None:
        cc=[x for x in c if code.search(x["text"])]; g=[x for x in c if not code.search(x["text"])]
        c=cc[:cq]+g[:cap-min(cq,len(cc))]; cap=len(c)
    for x in c[cap:]: seen.discard(key(x["text"]))
    out.extend(c[:cap]); print(src,len(c),cap,file=sys.stderr)
sw=[]
for n,i,t in swe():
    sw.append((n,i,t))
lic={"SWE-bench":"MIT","SWE-rebench":"CC-BY-4.0","SWE-Gym":"MIT"}
c=[]
for n,i,t in sw:
    t=t.strip()
    if ok(t) and key(t) not in ex and key(t) not in seen:
        seen.add(key(t)); c.append({"id":f"{n}:{i}","source":n,"license":lic[n],"text":t})
random.Random(0).shuffle(c); 
for x in c[6000:]: seen.discard(key(x["text"]))
out.extend(c[:6000]); print("swe",len(c),file=sys.stderr)
add("wildchat","ODC-BY-1.0",wild(),20000,cq=11000)
add("arena140k","CC-BY-4.0",arena(),12000,cq=4000)
add("oasst2","Apache-2.0",oasst(),5000)
add("magicoder","Apache-2.0",jl("Magicoder-Evol-Instruct-110K/data-evol_instruct-decontaminated.jsonl","instruction"),8000)
add("codefeedback","Apache-2.0",jl("CodeFeedback-Filtered-Instruction/CodeFeedback-Filtered-Instruction.jsonl","query"),8000)
random.Random(0).shuffle(out)
with open("gold_pool.jsonl","w") as f:
    for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
