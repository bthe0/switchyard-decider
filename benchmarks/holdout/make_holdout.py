import json
import os
import statistics

from cores import C
from gens import build, jobj, tm

HERE = os.path.dirname(os.path.abspath(__file__))
# Optional: path to the fresh100 generator, to assert no holdout ask is copied from it.
FRESH = os.environ.get("FRESH100_MAKE", "")

SHORT_EASY = [
    "What does `ls -la` show?",
    "Which key toggles insert mode off in vi?",
    "What's the file extension for Rust source files? My editor isn't highlighting anything and I think I named the file wrong.",
    "Python: how do I get the length of a list?",
    "What does `pwd` print?",
    "How do I print hello world in Go?",
    "What is the keyboard shortcut to save a file in VS Code on Windows, and is there one that saves every open file at once?",
    "How do I comment out a line in YAML?",
    "I'm in a repo someone else cloned for me. Which command shows the git branch I'm currently on?",
    "What does JSON stand for?",
    "How do I concatenate two strings in Java? I have firstName and lastName and want them joined with a space in between.",
    "Give me the SQL to count the rows in a table named events. Plain count of everything, no filters or grouping needed.",
    "In CSS, how do I make text bold?",
    "What does `==` versus `===` mean in JavaScript, in one sentence?",
    "How do I create a virtual environment with Python's built-in venv module, and how do I activate it on macOS?",
    "I always forget this: in a bash script, how do I check whether a file exists before trying to read it? Just the if statement please.",
    "What's the command to see how much free disk space I have on a Linux server, in human-readable units?",
    "Quick one: I have a TypeScript array of numbers and want the largest value. What's the shortest way to write that?",
    "How do I declare a constant in Go, and can I group several constants together in one block? A tiny example is enough.",
    "My colleague mentioned I should use a .gitignore file. What is it for, and what would a line look like that ignores the node_modules folder?",
    "I'm new to SQL. If I have a table called products with a column called price, how do I get the rows ordered from most expensive to cheapest? Just the query, no explanation needed.",
    "Can you remind me how string interpolation works in Kotlin? I just want to print a variable called name inside a greeting, something like Hello followed by the name. One line of code is fine.",
    "I have a Python dictionary and I want to loop over it and print each key together with its value. What is the usual way to write that loop? I know this is basic, I just can't remember whether it's items() or something else.",
    "In a Dockerfile, which instruction sets the working directory for the commands that follow it? I keep mixing it up with the one that copies files. Just name the instruction and show a one-line example using /app.",
    "We use kubectl at work. What is the command to list all the pods in a namespace called staging? And is there a flag to make it show which node each pod is on? Nothing fancy, I only need the two commands written out.",
    "What does the HTTP verb PATCH mean compared to PUT? One or two sentences.",
    "How do I round a float to two decimals in Python? I'm printing prices and 3.14159 should come out as 3.14.",
    "Write a C `for` loop that prints 0 through 9, one number per line. Nothing else, I only need the loop itself.",
    "What does `cd -` do in bash?",
    "How do I add an element to the end of a Rust Vec? I've got a Vec of i32 and just want to append the number 7.",
    "Terraform: which command previews changes without applying them?",
    "In Java, how do I convert the string \"42\" to an int?",
    "What does `tail -f` do?",
    "I need a one-line shell command that prints the number of CPU cores on a Linux box. I don't care which tool it uses as long as it's installed by default on Ubuntu.",
    "For a README: what's the Markdown syntax for a link, an image and a fenced code block? I mix up the brackets and parentheses every single time, so just show me one tiny example of each and I'll copy them.",
]

SHORT_HARD = [
    "Why is double-checked locking broken without volatile in Java, and is the C++11 version safe?",
    "Lock-free MPMC queue, ABA-safe: code.",
    "Explain Raft joint consensus and why naive single-step membership changes are unsafe.",
    "Linear-scan regalloc with lifetime holes: how?",
    "Does SI prevent write skew? Prove it.",
    "Why can multithreaded EPOLLET lose wakeups?",
    "Design exactly-once Kafka to Postgres.",
    "Where does SSA construction need phi nodes, and how do dominance frontiers place them minimally?",
    "Hazard pointers vs epoch-based reclamation vs RCU: tradeoffs when a thread stalls.",
    "Why does HM with refs need the value restriction?",
    "Design idempotent payment capture across retries, timeouts and a crashed coordinator.",
    "G1 remark pauses hit 2s with heavy DirectByteBuffer use. Why, and what do I tune?",
    "Can a Raft follower that lost its disk rejoin with an empty log? Show the committed-entry loss.",
    "Wait-free SPSC ring in C11; justify each memory_order.",
    "How would you get linearizable reads from a Raft cluster without appending to the log on every read? Compare ReadIndex and leader leases, including the clock assumptions each one makes.",
    "Tokio service: throughput collapses past 512 concurrent connections while CPU sits idle. List the likely causes and how to tell them apart.",
    "Choose between leveled and tiered compaction for an LSM store with a write-heavy, scan-light workload; quantify write amplification and space amplification for both.",
    "HMAC over concatenated fields with no delimiters: show the canonicalization forgery and the fix.",
    "Build serializable multi-key transactions on a KV store that only offers single-key CAS. Sketch the protocol, the commit record, and how readers resolve in-doubt writes.",
    "Why can one io_uring shared across threads scale worse than epoll? Name the contention points and the standard designs that avoid them.",
    "Go: how can a goroutine observe a partially initialised struct published through an unsynchronised pointer? Argue from the Go memory model, not intuition.",
    "Plan a zero-downtime split of a hot 2TB Postgres table: dual writes, backfill, cutover. Call out the race in each phase and how you close it.",
    "Design a rate limiter that stays globally accurate across 40 regions with at most 5% overshoot and no synchronous cross-region call on the hot path.",
    "Explain false sharing in an MPSC counter array, how perf c2c exposes it, and how alignas padding interacts with struct layout and cache-line prefetching.",
    "Why is `SELECT ... FOR UPDATE SKIP LOCKED` as a job queue still prone to starvation and bloat? Design it properly.",
    "Why does an early-exit token compare leak over a network despite jitter? Give a constant-time one.",
    "CRDT vs OT for a collaborative text editor with long offline edits: which interleaving anomalies does each allow?",
    "Dynamo-style store, N=3, R=1, W=2, sloppy quorum plus hinted handoff: derive the guarantees and give a concrete stale-read anomaly.",
    "Rust: why can't an async fn recurse without boxing, and what does the size of the generated state machine have to do with it?",
    "How does a tracing JIT decide to deoptimize, and how are interpreter frames rebuilt at on-stack replacement exits?",
    "We want Spanner-style commit wait but our clocks are only NTP-synced. What uncertainty bound must we assume, how do we enforce it, and what does it cost in commit latency?",
    "Thundering herd on expiry for a key read 50k/s with a 2s recompute: compare request coalescing, probabilistic early expiry and stale-while-revalidate, including how each one fails.",
    "Spectre v1 inside a JIT sandbox: explain the gadget and why index masking works.",
    "Cancellation-safe structured concurrency in C++20 coroutines: design it.",
    "A 2PC coordinator crashes after prepare. What can participants do, and why doesn't 3PC fix it under partitions?",
]


def b(kind, seed, *needles, **kw):
    return lambda n: build(kind, seed, n, *needles, **kw)


def two(ka, sa, la, kb, sb, lb, *nb):
    def f(n):
        h = (n - len(la) - len(lb) - 4) // 2
        return la + "\n" + build(ka, sa, h) + "\n\n" + lb + "\n" + build(kb, sb, h, *nb)
    return f


def cc(name, kind, seed, la, lb, *needles, first=True):
    def f(n):
        a = la + "\n" + C[name]
        z = lb + "\n" + build(kind, seed, max(n - len(a) - len(lb) - 3, 400), *needles)
        return a + "\n\n" + z if first else z + "\n\n" + a
    return f


def sl(proc, msg):
    return lambda r, i: "%s %s %s %s: %s" % (r.mon, tm(r, i), r.host, proc, msg)


def al(level, logger, msg):
    return lambda r, i: "%s %s,%03d %-7s [%s] %s" % (r.day, tm(r, i), i * 37 % 1000, level, logger, msg)


def pl(pid, msg):
    return lambda r, i: "%s %s.%03d UTC [%d] %s" % (r.day, tm(r, i), i * 53 % 1000, pid, msg)


def jl(**d):
    return lambda r, i: json.dumps({"ts": "%sT%s.%03dZ" % (r.day, tm(r, i), i * 71 % 1000), **d})


def order(r, i):
    return jobj({
        "id": 889001,
        "customer": {"name": "Ilse Brandt", "email": "ilse.brandt@bergen.example"},
        "items": [{"sku": s, "qty": q, "price": p} for s, q, p in [("BK-0417", 1, 18.5), ("EL-2290", 2, 49.0), ("HM-0031", 1, 7.25), ("TY-8812", 3, 12.99), ("BK-1108", 1, 22.0)]],
        "status": "paid",
        "shipping": {"city": "Bergen", "method": "express"},
    })


def mysqld(r, i):
    return "%-9s %6d %4.1f %4.1f %8d %7d ?        %-4s %s %3d:%02d %s" % (
        "mysql", 400 + i * 37 + 5, 3.2, 18.4, 2917344, 1502296, "Ssl", "03:12", 412, 7, "/usr/sbin/mysqld --defaults-file=/etc/mysql/my.cnf")


TRACEBACK = """unhandled error in worker
Traceback (most recent call last):
  File "/srv/orders/sync.py", line 88, in run
    batch = self.source.pull(cursor)
  File "/srv/orders/source.py", line 41, in pull
    return self._decode(raw)
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 17: invalid continuation byte"""

METABASE = """  metabase:
    image: metabase/metabase:v0.50.21
    restart: unless-stopped
    environment:
      MB_DB_TYPE: postgres
    ports:
      - "3030:3000"
    mem_limit: 2048m
"""

BASTION = """resource "aws_instance" "bastion" {
  ami           = "ami-0c4e1f9a2b7d33e10"
  instance_type = "t4g.nano"
  subnet_id     = aws_subnet.public_a.id

  tags = {
    Name  = "bastion"
    Owner = "platform"
    Env   = "prod"
  }
}
"""

REPORTING = "[reporting]\nhost = reporting.internal\nport = 7310\nworkers = 11\ntimeout = 45\nlog_level = info\nfeature_export_pdf = on\n"

COUPONS = "CREATE TABLE coupons (\n  code varchar(24) PRIMARY KEY,\n  percent_off smallint NOT NULL,\n  expires_at timestamptz,\n  max_redemptions integer NOT NULL\n);\n"

LONG_EASY = [
    ("Which client IP got the 502 on the refunds approve endpoint in this access log?",
     b("nginx", 101, lambda r, i: '185.44.12.9 - - [%s:%s +0000] "POST /api/v2/refunds/8841/approve HTTP/1.1" 502 173 "-" "okhttp/4.12.0" 1.204' % (r.nday, tm(r, i)))),
    ("From which address did deploybot log in?",
     b("syslog", 102, sl("sshd[4411]", "Accepted publickey for deploybot from 172.31.44.203 port 51122 ssh2"))),
    ("There's exactly one traceback buried in this log. What exception type is it?",
     b("applog", 103, al("ERROR", "orders.sync", TRACEBACK))),
    ("How many line items does the order for Ilse Brandt have?",
     b("json", 104, order)),
    ("One pod in these events could not be scheduled. Name it.",
     b("k8sev", 105, lambda r, i: "%-7s %-8s %-12s pod/%s   %s" % ("%dm" % (i * 2 + 1), "Warning", "FailedScheduling", "reports-exporter-6d9f7c5b8-xk2lp", "0/9 nodes are available: 9 Insufficient memory."))),
    ("List the column names of this CSV, nothing else.",
     b("csv", 106)),
    ("Rewrite only the first statement so it inserts into a table called clients instead of customers.",
     b("sqlins", 107)),
    ("What's the short hash of the revert commit in this history?",
     b("gitlog", 108, lambda r, i: '9f3c2ab %s %-17s revert: "feat(pricing): merge coupon cache"' % (r.day, "ines.costa"))),
    ("What are these two files, and which Go tool maintains the first one? One line is enough.",
     two("gosum", 109, "go.sum", "go", 1090, "internal/model/types.go")),
    ("What is process_resident_memory_bytes for the ledger service in this scrape?",
     b("prom", 110, 'process_resident_memory_bytes{service="ledger",instance="10.8.3.41:9100"} 412876800')),
    ("Give me the PID of mysqld from this process listing.",
     b("ps", 111, mysqld)),
    ("Exactly one test failed in this run. What's its node id?",
     b("pytest", 112, "tests/billing/test_invoice.py::test_render_invoice_rounding FAILED")),
    ("Which Go test failed here? Just the name.",
     b("gotest", 113, "=== RUN   TestMergeQuota/overlapping_window\n    quota_test.go:142: got 7, want 9\n--- FAIL: TestMergeQuota/overlapping_window (0.03s)")),
    ("Which compiler produced the errors listed under the type definitions?",
     two("ts", 114, "src/types.ts", "tsc", 1140, "build output")),
    ("Describe in one sentence what these compiler warnings are complaining about.",
     b("cargo", 115)),
    ("What is the outermost exception class in the stack trace?",
     two("javatrace", 116, "stack trace from the pod:", "applog", 1160, "app log from the minute before:")),
    ("Which image tag does the metabase service use in this compose file?",
     b("compose", 117, METABASE)),
    ("What kind of Kubernetes object are all of these documents?",
     b("k8s", 118)),
    ("What instance type is the bastion host?",
     b("tf", 119, BASTION)),
    ("According to the summary line, how many resources will this plan destroy?",
     b("tfplan", 120, tail="\nPlan: 14 to add, 3 to change, 2 to destroy.")),
    ("What port is the SMTP relay set to in this env file?",
     b("env", 121, "SMTP_RELAY_PORT=2525")),
    ("How many workers does the [reporting] section configure?",
     b("ini", 122, REPORTING)),
    ("I want to rename this class to ClientRecord. Which line or lines of the file have to change?",
     b("java", 123)),
    ("Are these type definitions TypeScript or Flow?",
     b("ts", 124)),
    ("What package do these structs belong to?",
     b("go", 125)),
    ("What value is UART2_BAUD_DIV defined as in this header?",
     b("ch", 126, "#define UART2_BAUD_DIV 0x01A1U")),
    ("Is the .proto file here proto2 or proto3?",
     two("proto", 127, "fulfilment.proto", "go", 1270, "the Go structs we map them to")),
    ("What is the primary key column of the coupons table?",
     b("ddl", 128, COUPONS)),
    ("What does the title tag of this page say?",
     b("html", 129)),
    ("Which release in this changelog has a Security section, and what date was it?",
     b("changelog", 130, "## [3.27.4] - 2026-03-09\n### Security\n- Rotated the signing key used for webhook payloads (#7741)\n")),
    ("What are the title and version of this API spec?",
     b("openapi", 131)),
    ("When does rotate_keys.sh run according to this crontab? Plain English.",
     b("cron", 132, "30 4 * * 0 /opt/jobs/rotate_keys.sh >> /var/log/jobs/rotate_keys.log 2>&1")),
    ("Translate the value of checkout.error.card_declined into German.",
     b("props", 133, "checkout.error.card_declined=Your card was declined. Please try another payment method.")),
    ("Which two standard commands produced these outputs?",
     two("dmesg", 134, "first:", "ps", 1340, "second:")),
    ("Where does the status record point in this zone file?",
     b("zone", 135, "%-26s %-5d IN %-5s %s" % ("status", 300, "CNAME", "acme.statuspage.example."))),
]

QUOTE_DEP = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: quote-api
  namespace: prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quote-api
  template:
    metadata:
      labels:
        app: quote-api
    spec:
      containers:
        - name: quote-api
          image: registry.internal/quote-api:4e1b7a2
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080"""

QUOTE_SVC = """apiVersion: v1
kind: Service
metadata:
  name: quote-api
  namespace: prod
spec:
  selector:
    app: quote-api
    tier: backend
  ports:
    - port: 80
      targetPort: 8080"""

QUOTE_ING = """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quote-api
  namespace: prod
spec:
  rules:
    - host: quotes.acme.example
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: quote-api
                port:
                  number: 80"""

LW_DEP = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: ledger-writer
  namespace: prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ledger-writer
  template:
    metadata:
      labels:
        app: ledger-writer
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app: ledger-writer
              topologyKey: kubernetes.io/hostname
      containers:
        - name: ledger-writer
          image: registry.internal/ledger-writer:9c0d2e1
          resources:
            limits:
              cpu: 1000m
              memory: 1024Mi"""

LW_PDB = """apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ledger-writer
  namespace: prod
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: ledger-writer"""

TF_WORKERS = """variable "worker_names" {
  default = ["ingest", "resize", "notify", "billing-sync"]
}

resource "aws_instance" "worker" {
  count         = length(var.worker_names)
  ami           = "ami-0f3a9c1d2e4b5a6c7"
  instance_type = "m6i.large"

  tags = {
    Name = var.worker_names[count.index]
  }
}

resource "aws_eip" "worker" {
  count    = length(var.worker_names)
  instance = aws_instance.worker[count.index].id
}
"""

TF_BUCKET = """resource "aws_s3_bucket_policy" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "OnlyFromVpcEndpoint"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.uploads.arn, "${aws_s3_bucket.uploads.arn}/*"]
      Condition = {
        StringNotEquals = { "aws:SourceVpce" = aws_vpc_endpoint.s3.id }
      }
    }]
  })
}
"""

TF_LAMBDA = """resource "aws_lambda_function" "thumbnailer" {
  function_name = "thumbnailer"
  role          = aws_iam_role.thumbnailer.arn
  handler       = "index.handler"
  runtime       = "nodejs20.x"
  filename      = "build/thumbnailer.zip"
  timeout       = 30
}

resource "aws_iam_role_policy" "thumbnailer_s3" {
  role = aws_iam_role.thumbnailer.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject"]
      Resource = "${aws_s3_bucket.uploads.arn}/*"
    }]
  })
}
"""

NGX_APP = """server {
    listen 443 ssl;
    server_name app.acme.example;
    ssl_certificate /etc/ssl/app.pem;

    location /api/ {
        proxy_pass http://api_upstream/;
    }

    location ~* \\.(png|jpg|css|js)$ {
        root /var/www/static;
        expires 30d;
    }

    location /api/v2/ {
        proxy_pass http://api_v2_upstream;
    }
}
"""

GHA_PRT = """  integration:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - run: npm ci && npm run test:integration
        env:
          STAGING_DB_URL: ${{ secrets.STAGING_DB_URL }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
"""

COMPOSE_API = """  api:
    build: ./api
    command: node src/server.js
    depends_on:
      - db
    volumes:
      - ./api:/app
    environment:
      DATABASE_URL: postgres://app:app@db:5432/app
    ports:
      - "3000:3000"
"""

COMPOSE_DB = """  db:
    image: postgres:16
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
    volumes:
      - ./initdb:/docker-entrypoint-initdb.d
      - pgdata:/var/lib/postgresql/data
"""

PROM_HEAD = "prometheus.yml:\nglobal:\n  scrape_interval: 60s\n  evaluation_interval: 30s\nrule_files:\n  - rules/*.yml\n\nrules/service-alerts.yml:\ngroups:\n  - name: service-alerts\n    rules:\n"

PROM_RULE = """      - alert: CheckoutErrorBurst
        expr: rate(http_requests_total{service="checkout",code=~"5.."}[30s]) > 0.2
        for: 1m
        labels:
          severity: page
          team: checkout
        annotations:
          summary: "checkout 5xx burst on {{ $labels.instance }}"
"""

DEADLOCK = "fatal error: all goroutines are asleep - deadlock!\n\ngoroutine 1 [semacquire]:\nsync.(*WaitGroup).Wait(0xc00011e010)\n\t/usr/local/go/src/sync/waitgroup.go:116 +0x48\ningest.(*Pool).Run(0xc00011e000, {0x10f4a28, 0xc000012120}, 0x8, {0xc00020a000, 0x200, 0x200})\n\t/src/ingest/pool.go:27 +0x1d4"

LONG_HARD = [
    ("This pool sometimes hangs forever in CI, and on other runs it returns fewer results than jobs. Find every concurrency bug in Run and rewrite it correctly.",
     cc("go_pool", "gotest", 201, "ingest/pool.go", "go test ./... -race, tail:", DEADLOCK)),
    ("Under load we see dozens of token fetches for the same tenant within one second, and unrelated tenants queue behind each other. Explain why this cache stampedes and serialises, then redesign it.",
     cc("py_token_cache", "applog", 202, "auth/token_cache.py", "what the service logged during the spike", first=False)),
    ("We intermittently get null price rules, and once on an old JDK a thread spun at 100% CPU inside rule(). Identify all thread-safety problems in this class and give a correct version that never blocks readers during a reload.",
     cc("java_registry", "jsonlog", 203, "PricingRegistry.java:", "request log around one of the null results:")),
    ("Two transfers between the same pair of accounts in opposite directions occasionally freeze the service, and a transfer from an account to itself always does. Explain both and fix them without a global lock.",
     cc("rust_bank", "cargo", 204, "src/bank.rs", "cargo build is clean apart from these", first=False)),
    ("This single-producer single-consumer ring between an ISR and the main loop occasionally hands the consumer a message with stale contents on our multi-core ARM board. Explain precisely why, and give a correct lock-free version with the right memory ordering.",
     cc("c_ring", "dmesg", 205, "ring.c", "kernel log from the board for context")),
    ("Memory on the export service grows until it is OOM-killed, and when several exports run at once some requests never get a response. Find the causes in this handler and fix them.",
     cc("ts_export", "nginx", 206, "export.ts:", "proxy access log from the same hour:", first=False)),
    ("This lookup takes four seconds despite the index on order_ref. Using the schema, the Go call and the EXPLAIN, say why the index is ignored and fix it at the right layer.",
     cc("mysql_cast", "applog", 207, "table, caller and plan", "slow request log", al("WARNING", "shipping.fetch", "ByOrder ref=48211973 took 4127ms rows=2"))),
    ("A coupon with max_redemptions=100 was redeemed 131 times, and a few users got the wallet credit twice. The database is Postgres at its default isolation level. Show the interleavings that cause each and rewrite redeem so neither can happen.",
     cc("py_coupon", "pglog", 208, "coupons/services.py", "postgres log from the campaign launch", first=False)),
    ("Goroutine count on this service climbs steadily whenever the backend is slow, and backend load doesn't drop when callers time out. Explain both from Lookup and fix it.",
     cc("go_lookup", "prom", 209, "client.go", "/metrics excerpt", 'go_goroutines{service="profile",instance="10.8.1.17:9100"} 184211')),
    ("Last night this deploy script pointed /srv/current at an empty directory and reloaded the app even though the artifact download had failed. It has set -e. Why didn't it stop, what else in it is dangerous, and how should it be written?",
     cc("sh_deploy", "syslog", 210, "deploy.sh", "syslog on the host", sl("deploy.sh[20114]", "curl: (22) The requested URL returned error: 404"), first=False)),
    ("After a single network blip every lease silently expired and nothing was renewed again until we restarted; separately, two threads sometimes both believe they acquired the same key. Explain both from this class and fix it.",
     cc("java_lease", "jsonlog", 211, "LeaseRenewer.java", "service log")),
    ("On our current_thread tokio runtime this session code freezes completely when an ack arrives while a send is in flight. Explain the freeze and restructure the code, keeping seq assignment and the unacked bookkeeping consistent.",
     cc("rust_session", "strace", 212, "session.rs:", "strace -f of the stuck process, last lines before it went quiet:", first=False)),
    ("The daemon crashes in reap_idle with invalid reads under valgrind, and large uploads corrupt the heap. Find both memory bugs and fix them.",
     cc("c_conn", "syslog", 213, "conn.c", "host log", sl("kernel", "relayd[8812]: segfault at 18 ip 000055d0c1a2b3f4 sp 00007ffd3a1c9e20 error 4 in relayd[55d0c1a29000+6000]"))),
    ("RSS of the report workers grows without bound and each build call is slower than the previous one. Find every reason in this module and fix them.",
     cc("py_report", "ps", 214, "reports/builder.py", "ps aux on the worker box after two days", first=False)),
    ("The race detector flags this registry and in production we hit `fatal error: concurrent map writes`. Identify each data race, say what Sweep actually closes, and give a corrected implementation that doesn't dial while holding the lock.",
     cc("go_registry", "gotest", 215, "registry.go", "test run")),
    ("This hook shows results for an older query after fast typing, duplicates or drops pages, and its auto-advance never gets past page 1. Explain each bug and rewrite the hook.",
     cc("ts_search", "tsc", 216, "useSearch.ts", "unrelated type errors the build also prints", first=False)),
    ("These two code paths deadlock a few times a day and inventory.reserved occasionally goes negative. Explain the lock-ordering problem and the double release, and fix both.",
     cc("sql_sessions", "pglog", 217, "the two transactions", "server log", pl(31902, "ERROR:  deadlock detected\nDETAIL:  Process 31902 waits for ShareLock on transaction 88120417; blocked by process 30771.\n\tProcess 30771 waits for ShareLock on transaction 88120422; blocked by process 31902."))),
    ("After a rebalance or a crash this consumer both loses events and double-applies others to balances. Explain how each happens with this code and redesign it for effectively-once processing.",
     cc("py_kafka", "kafka", 218, "ledger_writer.py", "kafka-consumer-groups --describe --all-groups", first=False)),
    ("Two settle runs for the same account overlapped and the fee was charged twice; another time the lock key stayed forever after a deploy. Explain every flaw in this lock and write a safe version.",
     cc("py_lock", "redis", 219, "locks.py", "redis MONITOR sample")),
    ("The pentest report says this middleware can be bypassed three different ways. Find them and rewrite it securely.",
     cc("ts_jwt", "nginx", 220, "middleware/auth.ts", "access log the pentesters attached", first=False)),
    ("This limiter lets through far more than `rate` under concurrency, and almost nothing at low rates when it is called very frequently. Explain both failure modes and write a correct lock-free version.",
     cc("go_bucket", "prom", 221, "limiter/bucket.go:", "scrape from the gateway:")),
    ("This cache segfaults under concurrent readers, and after repeated puts of the same key it evicts live entries. Explain both and fix it.",
     cc("cpp_lru", "applog", 222, "lru_cache.h", "sidecar log", first=False)),
    ("Why does the planner estimate 1 row where there are 211874 and choose a nested loop that runs for a minute? Give the fix that corrects the estimate rather than a workaround that forces the plan.",
     cc("pg_plan", "pglog", 223, "schema, query, plan", "postgresql log")),
    ("Since we put pgbouncer in front of the ledger database, nightly_close randomly fails with 'prepared statement \"close_stmt\" does not exist', sometimes updates the wrong year's schema, and twice two runs overlapped. Explain each from the config and code and fix the job.",
     cc("pgbouncer", "pglog", 224, "pgbouncer.ini and the job", "server side", pl(18841, 'ERROR:  prepared statement "close_stmt" does not exist'), first=False)),
    ("systemd kills and restarts this service every 90 seconds although gunicorn serves requests fine in between. Why, and what should the unit look like?",
     cc("systemd", "syslog", 225, "/etc/systemd/system/ledger-api.service", "journal", sl("systemd[1]", "ledger-api.service: start operation timed out. Terminating."))),
    ("The image builds but crashes at startup loading sharp and bcrypt, and security found our npm token inside the published image. Explain both from this Dockerfile and rewrite it.",
     cc("dockerfile", "docker", 226, "Dockerfile, then the dependencies block of package.json", "build output from an unrelated image on the same runner", first=False)),
    ("About once a week relayd hangs forever right after a SIGHUP reload, stuck in a futex; other times it crashes inside handle. Explain both and restructure the reload.",
     cc("c_signal", "strace", 227, "relayd.c", "strace of a hung instance", lambda r, i: "%d %s.%06d --- SIGHUP {si_signo=SIGHUP, si_code=SI_USER, si_pid=1, si_uid=0} ---\n%d %s.%06d futex(0x7f3a1c0008f0, FUTEX_WAIT_PRIVATE, 2, NULL <unfinished ...>" % (r.base % 30000 + 1200, tm(r, i, 2), 118220, r.base % 30000 + 1200, tm(r, i, 2), 118901))),
    ("quotes.acme.example returns 503 from the ingress even though the quote-api pods are Running and Ready. Everything applied to the cluster is below. What is wrong and what is the minimal fix?",
     b("k8s", 228, QUOTE_DEP, QUOTE_SVC, QUOTE_ING)),
    ("Node drains during our cluster upgrade hang forever; kubectl drain keeps reporting that it cannot evict a pod. From these manifests, which objects are responsible, and how should they change so the upgrade proceeds without losing availability?",
     b("k8s", 229, LW_DEP, LW_PDB)),
    ("I need to decommission the resize worker by deleting it from worker_names. Explain exactly what terraform plan will do to the other workers and their EIPs, and give a migration (code change plus state commands) that removes only resize.",
     b("tf", 230, TF_WORKERS)),
    ("The thumbnailer Lambda gets AccessDenied on s3:GetObject for the uploads bucket although its role allows it. All the Terraform is below. Why, and what are the options to fix it without opening the bucket up?",
     b("tf", 231, TF_BUCKET, TF_LAMBDA)),
    ("Requests for /api/v2/exports/report.css and /api/avatars/17.png get a 404 from disk instead of reaching a backend, and the v1 backend sees paths without the /api prefix while v2 sees them with it. Explain each behaviour from this config and rewrite the app.acme.example server so API paths always proxy, with consistent prefixes.",
     b("nginxconf", 232, NGX_APP)),
    ("Security review says this workflow lets any outside contributor steal our secrets. Walk through the exact attack, then restructure it so fork PRs still get integration-tested without exposing anything.",
     b("gha", 233, GHA_PRT, head="name: ci\non:\n  pull_request_target:\n    types: [opened, synchronize]\njobs:\n")),
    ("On a fresh `docker compose up` the api container crashes with ECONNREFUSED to db about half the time, and on my Linux laptop it also dies with `Cannot find module 'express'` although the image builds fine. Diagnose both from this file and fix them properly.",
     b("compose", 234, COMPOSE_API, COMPOSE_DB, tail="\nvolumes:\n  pgdata:\n")),
    ("CheckoutErrorBurst never fired during last night's outage even though the dashboard shows checkout 5xx at about 3/s for ten minutes. Config and rules are below. Why is the expression always empty, and what should the rule look like?",
     b("promrule", 235, PROM_RULE, head=PROM_HEAD)),
]

RENDERER = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: report-renderer
  namespace: prod
spec:
  replicas: 4
  selector:
    matchLabels:
      app: report-renderer
  template:
    metadata:
      labels:
        app: report-renderer
    spec:
      containers:
        - name: report-renderer
          image: registry.internal/report-renderer:1f0c9ab
          env:
            - name: JAVA_TOOL_OPTIONS
              value: "-Xms2g -Xmx2g -XX:MaxDirectMemorySize=1g"
          resources:
            limits:
              cpu: 2000m
              memory: 2048Mi"""

TRACE = "7c1e44aa90b2d3f1"

TF_DB = """resource "aws_db_instance" "ledger" {
  identifier              = "ledger-prod"
  engine                  = "postgres"
  engine_version          = "15.7"
  instance_class          = "db.r6g.xlarge"
  allocated_storage       = 400
  username                = "ledger_admin"
  password                = "Winter2026!"
  publicly_accessible     = true
  storage_encrypted       = false
  backup_retention_period = 0
  skip_final_snapshot     = true
  deletion_protection     = false
  vpc_security_group_ids  = [aws_security_group.ledger_db.id]
}

resource "aws_security_group_rule" "ledger_db_in" {
  type              = "ingress"
  from_port         = 5432
  to_port           = 5432
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.ledger_db.id
}
"""

VACUUM = 'LOG:  automatic vacuum of table "ledger_db.public.entries": index scans: 0\n\tpages: 0 removed, 11420331 remain, 0 skipped due to pins, 0 skipped frozen\n\ttuples: 0 removed, 48211973 remain, 48100211 are dead but not yet removable, oldest xmin: 771203344'


def audit_lag(r, i):
    rows = []
    for p, lag in enumerate([0, 0, 0, 1841220, 0, 0]):
        end = 52000000 + p * 91877
        rows.append("%-18s %-18s %-3d %-10d %-10d %-6d consumer-%s-%d-%s /%s" % ("audit-consumer", "audit.events", p, end - lag, end, lag, "audit", p % 3 + 1, "c41d07e2", "10.8.7.%d" % (20 + p % 3)))
    return "\n".join(rows)


GHA_TEST = """  test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v5
      - run: npm ci && npm test
"""

GHA_DEPLOY = """  deploy:
    runs-on: ubuntu-24.04
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: ./scripts/migrate.sh production && ./scripts/deploy.sh production
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
"""

PAIRS = [
    (cc("py_mirror", "applog", 301, "mirror.py", "log from the last run"),
     "What is the default value of max_workers in the Mirror class below?",
     "Mirror.run reports fewer files done than were written, and some files on disk hold another URL's content. Explain both and fix them."),
    (cc("go_pull", "strace", 302, "syncer.go", "strace excerpt", first=False),
     "I'd like the receiver in PullAll renamed from s to sy. No need to rewrite the whole function, just show me what the updated signature line would look like after the rename.",
     "PullAll dies with 'too many open files' on big batches and connections are never reused. Why? Fix it."),
    (cc("pg_page", "ddl", 303, "invoices table, the listing query and its plan:", "rest of the schema:"),
     "What's the name of the index on the invoices table?",
     "Page 400 of this listing takes three seconds. Using the plan, explain where the time goes and redesign the query and index so deep pages cost the same as page 1."),
    (cc("rust_graph", "cargo", 304, "graph.rs", "compiler output", first=False),
     "Quick sanity check before I forward this to someone: which programming language is the graph code written in, and which two smart-pointer types wrap each Node? A one-line answer is all I need.",
     "propagate panics with 'already mutably borrowed' and overflows the stack on some graphs. Explain exactly when, and rewrite it safely."),
    (cc("java_audit", "javatrace", 305, "AuditFormatter.java", "an unrelated trace from the same pod"),
     "What date pattern string does AuditFormatter pass to SimpleDateFormat?",
     "Audit lines occasionally carry garbage dates and count() drifts from the truth under load. Explain why and fix both without one global lock around everything."),
    (cc("c_header", "syslog", 306, "header.c", "syslog", first=False),
     "How many bytes is the local name buffer in parse_header? I'm filling in a table of buffer sizes for our docs and just need the number that appears in the declaration.",
     "Audit parse_header and copy_value for memory-safety bugs. For each: a concrete triggering input and the fix."),
    (cc("ts_refund", "jsonlog", 307, "routes/refunds.ts", "gateway log"),
     "Which HTTP method and path does this route handle?",
     "This endpoint always answers refunded: 0, and customers were refunded at the gateway while their orders stayed 'paid'. Explain the control flow behind both and rewrite it to be correct under partial failure."),
    (cc("sh_backup", "cron", 308, "backup.sh", "crontab on the same box", first=False),
     "Which remote hostname does the backup script rsync its archives to? I only need the host part, not the user or the path, because I'm adding it to our firewall allowlist ticket.",
     "This backup skipped files with spaces last week and once archived the wrong directory. Find every way it loses data and rewrite it."),
    (cc("py_validate", "pytest", 309, "importer/validate.py", "pytest -v"),
     "Which function does import_file call to load the rows?",
     "Every row passes or fails based only on the last schema field, a second import of the same file imports nothing, and the returned count is always 0. Explain all three and fix them."),
    (cc("go_watcher", "gotest", 310, "watcher.go", "go test -v", first=False),
     "What is the name of the method that Loop calls when its timer case fires? I'm writing the call graph by hand for a design doc and only need that one method name.",
     "Under a steady stream of events flush never runs, and when it does, batches arrive overwritten. Explain both and fix."),
    (cc("sql_migration", "pglog", 311, "migration 0412", "postgres log from staging"),
     "Which column does this migration add and what default does it get?",
     "events has 900M rows and takes writes around the clock on Postgres 15. Explain what each statement locks and for how long, then give a zero-downtime plan that reaches the same end state."),
    (cc("java_payout", "mvn", 312, "PayoutService.java", "mvn output", first=False),
     "Which Spring annotation is placed on the payOne method in this class? Just the annotation name please; I'm grepping the codebase for other usages and want to be sure I search for the right one.",
     "A BankException left payouts marked SENT with no money moved. Give the two independent reasons nothing rolled back, and fix it."),
    (b("k8s", 313, RENDERER),
     "How many replicas does report-renderer run with?",
     "report-renderer is OOMKilled (exit 137) every few hours with no java.lang.OutOfMemoryError anywhere in its logs. From its manifest, explain why and how heap, off-heap and the container limit should be sized."),
    (b("jsonlog", 314,
       jl(level="info", service="gateway", msg="proxy checkout", trace_id=TRACE, duration_ms=0, region="eu-west-1", status=0, timeout_ms=3000),
       jl(level="warn", service="checkout", msg="payments call timed out, retrying", trace_id=TRACE, duration_ms=2500, region="eu-west-1", status=504, attempt=1, timeout_ms=2500),
       jl(level="error", service="gateway", msg="upstream timeout", trace_id=TRACE, duration_ms=3000, region="eu-west-1", status=504),
       jl(level="info", service="payments", msg="charge card", trace_id=TRACE, duration_ms=2840, region="eu-west-1", status=200, attempt=1),
       jl(level="info", service="payments", msg="charge card", trace_id=TRACE, duration_ms=1910, region="eu-west-1", status=200, attempt=2)),
     "Could you count for me how many of these log lines carry the trace_id 7c1e44aa90b2d3f1? I only need the number, I'm cross-checking it against what our tracing UI shows for that request.",
     "Follow trace 7c1e44aa90b2d3f1: the customer saw an error yet was charged twice. Explain the timeout and retry interaction and how to budget it across the three services."),
    (b("tf", 315, TF_DB),
     "What engine version is the ledger database on?",
     "Do a security and durability review of the ledger database in this Terraform: every risky setting, why it matters, the corrected resource, and which changes force a replacement or downtime."),
    (b("cron", 316, "*/5 * * * * /opt/jobs/reconcile_ledger.sh --since=$(date +%Y-%m-%d) >> /var/log/jobs/reconcile.log 2>&1"),
     "What address does cron send mail to in this crontab? It should be set near the top of the file; I need it to check whether the alias still exists before we migrate mail providers.",
     "reconcile_ledger.sh never receives its --since argument and copies of it pile up overnight. Explain both from its crontab line and fix it."),
    (b("pglog", 317,
       pl(4411, "LOG:  connection authorized: user=reports_ro database=ledger_db application_name=metabase"),
       pl(51220, VACUUM),
       pl(51377, "WARNING:  oldest xmin is far in the past\nHINT:  Close open transactions soon to avoid wraparound problems.")),
     "Which table does the automatic vacuum entry in these logs refer to?",
     "The entries table has bloated to 90 GB; autovacuum runs but removes nothing. Using these logs, explain what is holding cleanup back, how to find the culprit, and how to stop it recurring."),
    (b("kafka", 318, audit_lag),
     "What is the lag on partition 3 of audit.events in this output? Please just read the number off the table for me, I'm pasting it into the incident channel and don't need any analysis.",
     "Only one partition of audit.events lags while its siblings sit at zero. Rank the likely causes, say how to confirm each, and what to change."),
    (b("ipt", 319,
       "-A INPUT -s 10.0.0.0/8 -p tcp -m tcp --dport 22 -j REJECT --reject-with icmp-port-unreachable",
       "-A INPUT -s 10.20.0.0/16 -p tcp -m tcp --dport 22 -j ACCEPT"),
     "What is the default policy of the INPUT chain in this ruleset?",
     "Admins on 10.20.0.0/16 cannot SSH to this host although there is an explicit ACCEPT for them. Find out why from this ruleset and give the exact iptables commands that fix it live without flushing anything."),
    (b("gha", 320, GHA_TEST, GHA_DEPLOY),
     "What is the name of this workflow as it would appear in the Actions tab? It's whatever the top-level name key says; I can't find it quickly in a file this long.",
     "Twice a broken build reached production, and once two deploys overlapped and left a half-migrated database. Explain how this workflow allows both and fix it."),
]

BORDER_SHORT = [
    "Write a Python function that merges overlapping intervals, treating touching intervals like [1,3] and [3,5] as overlapping, without mutating the input.",
    "Write a fully typed TypeScript debounce with leading and trailing options and a cancel method, preserving the wrapped function's parameter types.",
    "Postgres: top 3 products by revenue per category, including ties at third place, from order_items(product_id, qty, price) and products(id, category_id).",
    "Bash script that finds duplicate files by content under a directory and prints them in groups, handling filenames with spaces and newlines.",
    "Implement a generic O(1) LRU cache in Go 1.22 with Get, Put and a capacity, using container/list, safe for concurrent use.",
    "Regex that matches valid IPv4 addresses only (0-255 per octet, no leading zeros), anchored. Explain each part.",
    "Why does a for loop with var and setTimeout print the same number N times in JavaScript? Show three different fixes.",
    "Rust function returning the longest common prefix of a slice of Strings as a &str, with no allocation per comparison and correct for multi-byte UTF-8.",
    "Java bounded blocking queue using only synchronized, wait and notifyAll, with put and take that handle spurious wakeups and interruption.",
    "Write a multi-stage Dockerfile for a FastAPI app managed with Poetry that runs as a non-root user and caches the dependency layer.",
]

BORDER_LONG = [
    ("Write an awk script over this access log that prints, per HTTP status code, the request count plus the average and 95th percentile of the last field (request time).",
     b("nginx", 401), 3100),
    ("Convert the first three services in this compose file into Kubernetes Deployment and Service manifests, carrying over env vars, ports and memory limits.",
     b("compose", 402), 4200),
    ("Give me a jq command that reads these log lines and outputs, per service, how many entries have status >= 500 and duration_ms > 1000, sorted by that count descending.",
     b("jsonlog", 403), 5600),
    ("Write a migration that adds soft deletes (deleted_at) to every table here and replaces each index with a partial index covering only live rows. It must be safe to re-run.",
     b("ddl", 404), 3600),
    ("Group these errors by TS error code with counts, tell me which single tsconfig strictness flag accounts for the most of them, and show the idiomatic fix for the two most common codes.",
     b("tsc", 405), 6800),
    ("Write a pandas script that loads this CSV and reports revenue (qty * unit_price) per country converted to EUR using a rates dict I can fill in, excluding cancelled and refunded orders, with the top SKU per country.",
     b("csv", 406), 4900),
    ("Write one reflection-based Validate(v any) error for these structs that rejects zero values for every field whose json tag doesn't end in _at, plus a table-driven test covering two of the types.",
     b("go", 407), 2900),
    ("Turn these manifests into a kustomize base plus a staging overlay that halves every replica count (minimum 1), moves everything to the staging namespace and appends -stg to image tags.",
     b("k8s", 408), 7900),
    ("Convert this POJO into a Java 17 record with a nested builder, keeping the same component names, and show how Jackson must be configured so existing JSON still deserialises.",
     b("java", 409), 3300),
    ("Translate this iptables ruleset into an equivalent nftables config file, preserving rule order and the logging and limit rules.",
     b("ipt", 410), 5200),
]


def wrap(ask, paste, pos, style):
    body = "```\n" + paste + "\n```" if style % 3 == 0 else paste
    gap = "\n\n" if style % 2 else "\n"
    return ask + gap + body if pos == "first" else body + gap + ask


def row(text, label, cell, pair_id, ask_pos):
    return {"text": text, "label": label, "cell": cell, "pair_id": pair_id, "ask_pos": ask_pos}


def auc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def main():
    rows = [row(t, 0, "short_easy", None, "none") for t in SHORT_EASY]
    rows += [row(t, 2, "short_hard", None, "none") for t in SHORT_HARD]
    target = [2900 + int(8500 * ((i * 17 % 35) / 34) ** 1.6) for i in range(35)]
    for items, label, cell in ((LONG_EASY, 0, "long_easy"), (LONG_HARD, 2, "long_hard")):
        for i, (ask, fn) in enumerate(items):
            pos = "first" if i % 2 == 0 else "last"
            rows.append(row(wrap(ask, fn(target[i] - len(ask) - 10), pos, i), label, cell, None, pos))
    combos = [("first", "first"), ("last", "last"), ("first", "last"), ("last", "first")]
    pastes = []
    for j, (fn, easy, hard) in enumerate(PAIRS):
        paste = fn(2700 + 330 * (j * 7 % 20))
        pastes.append(paste)
        pe, ph = combos[j % 4]
        rows.append(row(wrap(easy, paste, pe, j), 0, "pair_easy", "p%02d" % (j + 1), pe))
        rows.append(row(wrap(hard, paste, ph, j), 2, "pair_hard", "p%02d" % (j + 1), ph))
    rows += [row(t, 1, "borderline", None, "none") for t in BORDER_SHORT]
    for i, (ask, fn, n) in enumerate(BORDER_LONG):
        pos = "first" if i % 2 else "last"
        rows.append(row(wrap(ask, fn(n), pos, i + 1), 1, "borderline", None, pos))

    want = {"short_easy": 35, "short_hard": 35, "long_easy": 35, "long_hard": 35, "pair_easy": 20, "pair_hard": 20, "borderline": 20}
    lens = {c: [len(r["text"]) for r in rows if r["cell"] == c] for c in want}
    assert len(rows) == 200 and {c: len(v) for c, v in lens.items()} == want
    assert len({r["text"] for r in rows}) == 200
    assert max(lens["short_easy"] + lens["short_hard"]) < 300
    assert min(lens["long_easy"] + lens["long_hard"] + lens["pair_easy"] + lens["pair_hard"]) > 2500
    assert all(len(p) > 2500 for p in pastes) and len(set(pastes)) == 20
    for j, p in enumerate(pastes):
        assert sum(p in r["text"] for r in rows if r["pair_id"] == "p%02d" % (j + 1)) == 2
    asks = SHORT_EASY + SHORT_HARD + BORDER_SHORT + [x[0] for x in LONG_EASY + LONG_HARD + BORDER_LONG] + [a for p in PAIRS for a in p[1:]]
    assert len(set(asks)) == len(asks)
    if os.path.exists(FRESH):
        src = open(FRESH).read()
        bad = [a for a in asks if a in src or json.dumps(a)[1:-1] in src]
        assert not bad, bad
    blen = [len(r["text"]) for r in rows if r["cell"] == "borderline"]
    assert sum(n < 300 for n in blen) == 10 and sum(n > 2500 for n in blen) == 10

    with open(os.path.join(HERE, "holdout.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    for c, v in lens.items():
        print("%-11s n=%2d min=%5d median=%7.1f mean=%7.1f max=%5d" % (c, len(v), min(v), statistics.median(v), statistics.mean(v), max(v)))
    for a, z in (("short_hard", "short_easy"), ("long_hard", "long_easy"), ("pair_hard", "pair_easy")):
        print("auc %s vs %s: %.4f" % (a, z, auc(lens[a], lens[z])))
    for c in ("long_easy", "long_hard", "pair_easy", "pair_hard"):
        print(c, "ask first/last:", sum(r["ask_pos"] == "first" for r in rows if r["cell"] == c), sum(r["ask_pos"] == "last" for r in rows if r["cell"] == c))
    hard = [len(r["text"]) for r in rows if r["label"] == 2]
    easy = [len(r["text"]) for r in rows if r["label"] == 0]
    total = auc(hard, easy)
    print("rows=%d length-only AUC (label 2 vs 0, %d vs %d rows): %.4f" % (len(rows), len(hard), len(easy), total))
    assert 0.45 <= total <= 0.55


main()
