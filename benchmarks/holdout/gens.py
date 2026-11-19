import base64
import json
import random
import textwrap

SVC = "auth billing cart catalog checkout gateway inventory ledger mailer notifier orders payments pricing profile search shipping".split()
FIRST = "amara bjorn chidi dmitri elif farah goran hana ines jomo kavya leif mei nuno oksana priya quinn rafa sana tomas".split()
LAST = "okafor lindqvist anagonye volkov demir haddad petrov sato costa mwangi rao berg lin alves bondar nair reyes ortiz malik novak".split()
NOUN = "account address batch coupon device invoice item ledger message order payment price profile quota refund session shipment stock token user".split()
VERB = "create update delete fetch sync validate render parse retry flush merge archive".split()
PAST = "created updated deleted fetched synced validated rendered parsed retried flushed merged archived".split()
CITY = "lagos oslo izmir beirut sofia osaka porto nairobi pune bergen taipei recife kyiv kochi quito".split()
ATTR = "Id Name Code Count Total Status Flag Limit Note Hash".split()
ENVS = ["prod", "staging", "dev", "canary"]
REGION = ["eu-west-1", "us-east-2", "ap-south-1", "sa-east-1"]
PORTS = [22, 80, 443, 5432, 6379, 8080, 8443, 9090, 9100, 3306, 27017]
PKG = "libpq-dev curl ca-certificates build-essential libjpeg-dev tzdata git openssl libxml2 zlib1g-dev".split()
UA = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "curl/8.6.0",
    "okhttp/4.12.0",
    "python-requests/2.32.3",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) Mobile/15E148",
    "Go-http-client/2.0",
    "kube-probe/1.31",
]
FT = {
    "id": ("bigint", "number", "int64", "int64"),
    "name": ("text", "string", "string", "string"),
    "email": ("text", "string", "string", "string"),
    "status": ("varchar(16)", "string", "string", "string"),
    "amount": ("numeric(12,2)", "number", "float64", "double"),
    "currency": ("char(3)", "string", "string", "string"),
    "created_at": ("timestamptz", "Date", "time.Time", "int64"),
    "updated_at": ("timestamptz", "Date", "time.Time", "int64"),
    "owner_id": ("bigint", "number", "int64", "int64"),
    "region": ("text", "string", "string", "string"),
    "sku": ("varchar(32)", "string", "string", "string"),
    "quantity": ("integer", "number", "int", "int32"),
    "notes": ("text", "string", "string", "string"),
    "retries": ("smallint", "number", "int", "int32"),
    "priority": ("smallint", "number", "int", "int32"),
    "checksum": ("char(64)", "string", "string", "string"),
    "locale": ("varchar(8)", "string", "string", "string"),
    "timezone": ("text", "string", "string", "string"),
    "enabled": ("boolean", "boolean", "bool", "bool"),
    "version": ("integer", "number", "int", "int32"),
}
FIELD = list(FT)


def pick(r, xs):
    return r.choice(xs)


def ip(r):
    return "%d.%d.%d.%d" % (pick(r, [10, 172, 192, 34, 52, 185, 203]), r.randint(0, 255), r.randint(0, 255), r.randint(1, 254))


def hx(r, n):
    return "%0*x" % (n, r.getrandbits(4 * n))


def hms(s):
    s %= 86400
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def tm(r, i, step=11):
    return hms(r.base + i * step + r.randint(0, step - 1))


def who(r):
    return pick(r, FIRST) + "." + pick(r, LAST)


def camel(*p):
    return "".join(x.capitalize() for x in "_".join(p).split("_"))


def lower_camel(s):
    c = camel(s)
    return c[0].lower() + c[1:]


def uniq(i):
    return "%s-%s" % (SVC[i % 16], ["api", "worker", "cron", "admin", "edge"][i // 16 % 5])


def jobj(d):
    return textwrap.indent(json.dumps(d, indent=2), "  ")


def mk(seed):
    r = random.Random(seed)
    r.base = r.randint(3600, 70000)
    mon = r.randint(1, 10)
    dom = r.randint(1, 28)
    name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"][mon - 1]
    r.day = "2026-%02d-%02d" % (mon, dom)
    r.mon = "%s %2d" % (name, dom)
    r.nday = "%02d/%s/2026" % (dom, name)
    r.host = "%s-%s-%02d" % (pick(r, ["web", "db", "edge", "batch", "bastion"]), pick(r, CITY), r.randint(1, 20))
    return r


def urlp(r):
    return pick(r, [
        "/api/v2/%ss/%d" % (pick(r, NOUN), r.randint(1, 99999)),
        "/api/v2/%ss?page=%d" % (pick(r, NOUN), r.randint(1, 40)),
        "/static/app.%s.js" % hx(r, 6),
        "/healthz",
        "/login",
        "/search?q=%s" % pick(r, CITY),
        "/v1/%s/%s/%s" % (pick(r, SVC), pick(r, VERB), hx(r, 8)),
    ])


def k_nginx(r, i):
    return '%s - - [%s:%s +0000] "%s %s HTTP/1.1" %d %d "-" "%s" %.3f' % (
        ip(r), r.nday, tm(r, i), pick(r, ["GET", "GET", "GET", "POST", "PUT", "DELETE"]), urlp(r),
        pick(r, [200, 200, 200, 200, 204, 301, 304, 400, 401, 404, 499, 500]), r.randint(90, 90000), pick(r, UA), r.random() * 1.8)


def k_syslog(r, i):
    p = pick(r, ["sshd", "systemd", "kernel", "cron", "dockerd", "kubelet", "chronyd", "postfix/smtp"])
    m = pick(r, [
        "Accepted publickey for %s from %s port %d ssh2" % (pick(r, FIRST), ip(r), r.randint(30000, 65000)),
        "Started %s-%s.service." % (pick(r, SVC), pick(r, VERB)),
        "Failed password for invalid user %s from %s port %d ssh2" % (pick(r, FIRST), ip(r), r.randint(30000, 65000)),
        "(root) CMD (/opt/jobs/%s_%s.sh)" % (pick(r, VERB), pick(r, NOUN)),
        "container %s exited with status %d" % (hx(r, 12), pick(r, [0, 0, 1, 137, 143])),
        "Selected source %s" % ip(r),
        "connect to mx.%s.example[%s]:25: Connection timed out" % (pick(r, CITY), ip(r)),
        "%s-%s.service: Consumed %.3fs CPU time." % (pick(r, SVC), pick(r, VERB), r.uniform(0.1, 40)),
        "Reloading %s configuration" % pick(r, SVC),
    ])
    return "%s %s %s %s[%d]: %s" % (r.mon, tm(r, i), r.host, p, r.randint(300, 60000), m)


def k_applog(r, i):
    m = pick(r, [
        "%s %s id=%d took %dms" % (pick(r, VERB), pick(r, NOUN), r.randint(1, 99999), r.randint(2, 900)),
        "cache %s for key %s:%s" % (pick(r, ["hit", "miss", "evict"]), pick(r, NOUN), hx(r, 8)),
        "retry %d/5 calling %s-svc: %s" % (r.randint(1, 5), pick(r, SVC), pick(r, ["timeout", "connection reset", "503"])),
        "user %s %s %s %d" % (who(r), pick(r, PAST), pick(r, NOUN), r.randint(1, 9999)),
        "queue depth=%d workers=%d lag=%.1fs" % (r.randint(0, 4000), r.randint(2, 32), r.random() * 30),
        "pool checkout waited %dms (in_use=%d/20)" % (r.randint(0, 3000), r.randint(1, 20)),
    ])
    return "%s %s,%03d %-7s [%s.%s] %s" % (r.day, tm(r, i), r.randint(0, 999), pick(r, ["INFO", "INFO", "INFO", "DEBUG", "WARNING", "ERROR"]), pick(r, SVC), pick(r, VERB), m)


def k_jsonlog(r, i):
    return json.dumps({
        "ts": "%sT%s.%03dZ" % (r.day, tm(r, i), r.randint(0, 999)),
        "level": pick(r, ["info", "info", "info", "warn", "error"]),
        "service": pick(r, SVC),
        "msg": "%s %s" % (pick(r, VERB), pick(r, NOUN)),
        "trace_id": hx(r, 16),
        "duration_ms": r.randint(1, 2400),
        "region": pick(r, REGION),
        "status": pick(r, [200, 200, 201, 400, 404, 500, 503]),
    })


def k_json(r, i):
    return jobj({
        "id": 770000 + i * 13 + r.randint(0, 9),
        "customer": {"name": "%s %s" % (pick(r, FIRST).title(), pick(r, LAST).title()), "email": "%s@%s.example" % (who(r), pick(r, CITY))},
        "items": [{"sku": "%s-%04d" % (pick(r, ["BK", "EL", "HM", "TY"]), r.randint(1, 9999)), "qty": r.randint(1, 6), "price": round(r.uniform(2, 300), 2)} for _ in range(r.randint(1, 3))],
        "status": pick(r, ["paid", "shipped", "pending", "refunded"]),
        "shipping": {"city": pick(r, CITY).title(), "method": pick(r, ["standard", "express", "pickup"])},
    })


def k_k8sev(r, i):
    s = pick(r, SVC)
    typ, reason, msg = pick(r, [
        ("Normal", "Pulled", 'Container image "registry.internal/%s:%s" already present on machine' % (s, hx(r, 7))),
        ("Normal", "Scheduled", "Successfully assigned %s/%s to node-%s-%d" % (pick(r, ENVS), s, pick(r, CITY), r.randint(1, 9))),
        ("Warning", "BackOff", "Back-off restarting failed container %s" % s),
        ("Warning", "Unhealthy", "Readiness probe failed: HTTP probe failed with statuscode: %d" % pick(r, [500, 503])),
        ("Normal", "Killing", "Stopping container %s" % s),
        ("Warning", "FailedMount", 'MountVolume.SetUp failed for volume "%s-config" : configmap "%s-config" not found' % (s, s)),
        ("Normal", "Started", "Started container %s" % s),
    ])
    return "%-7s %-8s %-12s pod/%s-%s-%s   %s" % ("%dm" % (i * 2 + 1), typ, reason, s, hx(r, 9), hx(r, 5), msg)


def k_csv(r, i):
    return "%d,%s@%s.example,%s-%04d,%d,%.2f,%s,%s,%s,%sT%s" % (
        500000 + i * 3 + r.randint(0, 2), who(r), pick(r, CITY), pick(r, ["SKU", "BK", "EL", "HM"]), r.randint(1, 9999), r.randint(1, 12),
        r.uniform(2, 480), pick(r, ["EUR", "USD", "GBP", "JPY"]), pick(r, ["DE", "US", "NG", "JP", "BR", "IN", "NO"]),
        pick(r, ["paid", "paid", "shipped", "refunded", "pending", "cancelled"]), r.day, tm(r, i))


def k_sqlins(r, i):
    return "INSERT INTO customers (id, email, full_name, plan, city, credits, created_at) VALUES (%d, '%s@%s.example', '%s %s', '%s', '%s', %d, '%s %s');" % (
        9000 + i, who(r), pick(r, CITY), pick(r, FIRST).title(), pick(r, LAST).title(), pick(r, ["free", "pro", "team", "enterprise"]), pick(r, CITY).title(), r.randint(0, 5000), r.day, tm(r, i))


def k_gitlog(r, i):
    return "%s %s %-17s %s(%s): %s %s %s" % (
        hx(r, 7), r.day, who(r), pick(r, ["fix", "feat", "chore", "refactor", "test", "docs"]), pick(r, SVC), pick(r, VERB), pick(r, NOUN),
        pick(r, ["handler", "cache", "migration", "retry path", "validation", "serializer", "on timeout", "for bulk import"]))


def k_gosum(r, i):
    m = "github.com/%s/%s-%s v%d.%d.%d" % (pick(r, LAST), pick(r, NOUN), pick(r, ["go", "kit", "client", "util", "sdk"]), r.randint(0, 4), r.randint(0, 30), r.randint(0, 12))
    return "%s%s h1:%s" % (m, pick(r, ["", "/go.mod"]), base64.b64encode(r.getrandbits(256).to_bytes(32, "big")).decode())


def k_prom(r, i):
    return '%s{service="%s",method="%s",code="%d",instance="%s:9100"} %d' % (
        pick(r, ["http_requests_total", "http_request_duration_seconds_count", "grpc_server_handled_total", "db_pool_connections_in_use", "queue_messages_processed_total"]),
        pick(r, SVC), pick(r, ["GET", "POST", "PUT"]), pick(r, [200, 201, 400, 404, 500, 503]), ip(r), r.randint(0, 900000))


def k_ps(r, i):
    cmd = pick(r, [
        "/usr/bin/python3 /srv/%s/manage.py runworker" % pick(r, SVC),
        "nginx: worker process",
        "postgres: %s %s_db %s(%d) idle" % (pick(r, FIRST), pick(r, SVC), ip(r), r.randint(30000, 65000)),
        "/usr/local/bin/node /srv/%s/dist/server.js" % pick(r, SVC),
        "/usr/bin/redis-server 127.0.0.1:%d" % r.randint(6379, 6390),
        "java -Xmx%dg -jar /opt/%s/%s.jar" % (r.randint(1, 8), pick(r, SVC), pick(r, SVC)),
        "sshd: %s@pts/%d" % (pick(r, FIRST), r.randint(0, 9)),
    ])
    return "%-9s %6d %4.1f %4.1f %8d %7d ?        %-4s %s %3d:%02d %s" % (
        pick(r, ["root", "www-data", "postgres", "deploy", "redis", "nobody"]), 400 + i * 37 + r.randint(0, 30), r.uniform(0, 40), r.uniform(0, 12),
        r.randint(9000, 4000000), r.randint(800, 900000), pick(r, ["Ss", "S", "Sl", "R", "Ssl"]), tm(r, i)[:5], r.randint(0, 300), r.randint(0, 59), cmd)


def k_pytest(r, i):
    return "tests/%s/test_%s.py::test_%s_%s_%s %s" % (
        pick(r, SVC), pick(r, NOUN), pick(r, VERB), pick(r, NOUN), pick(r, ["ok", "empty", "duplicate", "unauthorized", "timeout", "unicode", "bulk"]),
        pick(r, ["PASSED"] * 9 + ["SKIPPED", "XFAIL"]))


def k_gotest(r, i):
    n = "Test%s%s/%s_%s" % (camel(pick(r, VERB)), camel(pick(r, NOUN)), pick(r, ["empty", "single", "duplicate", "expired", "concurrent", "large"]), pick(r, NOUN))
    return "=== RUN   %s\n--- PASS: %s (%.2fs)" % (n, n, r.random() * 0.4)


def k_tsc(r, i):
    f = "src/%s/%s%s.ts" % (pick(r, SVC), pick(r, NOUN), pick(r, ["Service", "Controller", "Repo", ".dto", ".test"]))
    c, m = pick(r, [
        (2322, "Type 'string' is not assignable to type 'number'."),
        (2339, "Property '%s' does not exist on type '%s'." % (lower_camel(pick(r, FIELD)), camel(pick(r, NOUN)))),
        (7006, "Parameter '%s' implicitly has an 'any' type." % pick(r, NOUN)),
        (2345, "Argument of type '%s | undefined' is not assignable to parameter of type '%s'." % ((camel(pick(r, NOUN)),) * 2)),
        (18048, "'%s' is possibly 'undefined'." % pick(r, NOUN)),
        (2304, "Cannot find name '%s'." % camel(pick(r, VERB), pick(r, NOUN))),
    ])
    return "%s(%d,%d): error TS%d: %s" % (f, r.randint(3, 400), r.randint(1, 80), c, m)


def k_cargo(r, i):
    v = pick(r, NOUN)
    ln = r.randint(8, 600)
    return "warning: unused variable: `%s`\n   --> crates/%s/src/%s.rs:%d:9\n    |\n%3d |     let %s = %s_%s(&ctx)?;\n    |         %s help: if this is intentional, prefix it with an underscore: `_%s`\n" % (
        v, pick(r, SVC), pick(r, VERB), ln, ln, v, pick(r, VERB), pick(r, NOUN), "^" * len(v), v)


def k_javatrace(r, i):
    c = camel(pick(r, NOUN)) + pick(r, ["Service", "Controller", "Repository", "Filter", "Interceptor", "Handler"])
    pk = pick(r, ["com.acme.%s" % pick(r, SVC), "org.springframework.web.servlet", "org.springframework.aop.framework", "org.apache.catalina.core", "com.zaxxer.hikari.pool", "org.hibernate.engine.jdbc"])
    return "\tat %s.%s.%s%s(%s.java:%d)" % (pk, c, pick(r, VERB), camel(pick(r, NOUN)), c, r.randint(20, 900))


def k_compose(r, i):
    s = uniq(i)
    return '  %s:\n    image: registry.internal/%s:%d.%d.%d\n    restart: unless-stopped\n    environment:\n      LOG_LEVEL: %s\n      %s_URL: http://%s:%d\n    ports:\n      - "%d:%d"\n    mem_limit: %dm\n' % (
        s, s, r.randint(0, 9), r.randint(0, 40), r.randint(0, 20), pick(r, ["info", "debug", "warn"]), pick(r, SVC).upper(), uniq(r.randint(0, 60)), pick(r, PORTS),
        8000 + i, pick(r, [8080, 3000, 5000, 9000]), pick(r, [128, 256, 512, 1024]))


def k_k8s(r, i):
    s = uniq(i)
    return ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: %s\n  namespace: %s\nspec:\n  replicas: %d\n  selector:\n    matchLabels:\n      app: %s\n"
            "  template:\n    metadata:\n      labels:\n        app: %s\n    spec:\n      containers:\n        - name: %s\n          image: registry.internal/%s:%s\n"
            "          ports:\n            - containerPort: %d\n          resources:\n            limits:\n              cpu: %dm\n              memory: %dMi") % (
        s, pick(r, ENVS), r.randint(1, 8), s, s, s, s, hx(r, 7), pick(r, [8080, 3000, 5000, 9000]), pick(r, [250, 500, 1000, 2000]), pick(r, [256, 512, 1024, 2048]))


def k_tf(r, i):
    s = uniq(i).replace("-", "_")
    return 'resource "aws_instance" "%s" {\n  ami           = "ami-%s"\n  instance_type = "%s"\n  subnet_id     = aws_subnet.%s_%s.id\n\n  tags = {\n    Name  = "%s"\n    Owner = "%s"\n    Env   = "%s"\n  }\n}\n' % (
        s, hx(r, 17), pick(r, ["t3.micro", "t3.small", "m6i.large", "c7g.xlarge", "r6g.large"]), pick(r, ["private", "public"]), pick(r, "abc"), s.replace("_", "-"), who(r), pick(r, ENVS))


def k_tfplan(r, i):
    s = uniq(i).replace("-", "_")
    act, y = pick(r, [("created", "+"), ("created", "+"), ("destroyed", "-"), ("updated in-place", "~")])
    return '  # aws_security_group_rule.%s will be %s\n  %s resource "aws_security_group_rule" "%s" {\n      %s cidr_blocks       = ["%s/%d"]\n      %s from_port         = %d\n      %s protocol          = "tcp"\n      %s security_group_id = "sg-%s"\n      %s type              = "%s"\n    }\n' % (
        s, act, y, s, y, ip(r), pick(r, [16, 24, 32]), y, pick(r, PORTS), y, y, hx(r, 17), y, pick(r, ["ingress", "egress"]))


def k_env(r, i):
    suf, val = pick(r, [
        ("TTL", str(r.randint(30, 86400))),
        ("URL", "https://%s.internal:%d" % (pick(r, SVC), r.randint(3000, 9999))),
        ("TIMEOUT_MS", str(r.randint(100, 30000))),
        ("MAX_RETRIES", str(r.randint(0, 8))),
        ("ENABLED", pick(r, ["true", "false"])),
        ("POOL_SIZE", str(r.randint(2, 64))),
        ("REGION", pick(r, REGION)),
        ("BUCKET", "%s-%s-%s" % (pick(r, SVC), pick(r, ENVS), hx(r, 6))),
    ])
    return "%s_%s_%s=%s" % (SVC[i % 16].upper(), pick(r, NOUN).upper(), suf, val)


def k_ini(r, i):
    return "[%s]\nhost = %s.internal\nport = %d\nworkers = %d\ntimeout = %d\nlog_level = %s\nfeature_%s_%s = %s\n" % (
        uniq(i), uniq(i), r.randint(3000, 9999), r.randint(1, 32), r.randint(5, 120), pick(r, ["info", "debug", "warning"]), pick(r, VERB), pick(r, NOUN), pick(r, ["on", "off"]))


def k_java(r, i):
    ty = pick(r, ["String", "long", "int", "boolean", "BigDecimal", "Instant", "List<String>"])
    f = NOUN[i % 20] + ATTR[(i // 20 + i) % 10]
    c = f[0].upper() + f[1:]
    return "    private %s %s;\n\n    public %s get%s() {\n        return %s;\n    }\n\n    public void set%s(%s %s) {\n        this.%s = %s;\n    }\n" % (ty, f, ty, c, f, c, ty, f, f, f)


def k_ts(r, i):
    fs = r.sample(FIELD, r.randint(4, 8))
    return "export interface %s%s {\n%s\n}\n" % (
        camel(NOUN[i % 20]), ["Dto", "Row", "Input", "Event", "Patch"][i // 20 % 5],
        "\n".join("  %s%s: %s;" % (lower_camel(f), pick(r, ["", "", "?"]), FT[f][1]) for f in fs))


def k_go(r, i):
    fs = r.sample(FIELD, r.randint(4, 8))
    return "type %s%s struct {\n%s\n}\n" % (
        camel(NOUN[i % 20]), ["", "Row", "Request", "Event", "Patch"][i // 20 % 5],
        "\n".join("\t%s %s `json:\"%s\"`" % (camel(f), FT[f][2], f) for f in fs))


def k_ch(r, i):
    return "#define %s%d_%s_%s 0x%04XU" % (pick(r, ["UART", "SPI", "I2C", "TIM", "ADC", "GPIO"]), i % 7, pick(r, ["CTRL", "STAT", "DATA", "MASK", "IRQ", "CLK"]), ["EN", "DIV", "RST", "SEL", "TH", "CFG", "LVL"][i // 7 % 7] + str(i // 49), r.getrandbits(16))


def k_proto(r, i):
    fs = r.sample(FIELD, r.randint(4, 8))
    return "message %s%s {\n%s\n}\n" % (
        camel(NOUN[i % 20]), ["", "Request", "Response", "Event", "Patch"][i // 20 % 5],
        "\n".join("  %s %s = %d;" % (FT[f][3], f, k + 1) for k, f in enumerate(fs)))


def k_ddl(r, i):
    t = "%s_%ss" % (SVC[i % 16], NOUN[(i * 7 + i // 16) % 20])
    cols = r.sample(FIELD[1:], r.randint(4, 8))
    body = ["  id bigserial PRIMARY KEY"] + ["  %s %s%s" % (c, FT[c][0], pick(r, ["", " NOT NULL", " NOT NULL"])) for c in cols]
    return "CREATE TABLE %s (\n%s\n);\nCREATE INDEX %s_%s_idx ON %s (%s);\n" % (t, ",\n".join(body), t, cols[0], t, cols[0])


def k_html(r, i):
    return "<tr><td>%s-%04d</td><td>%s %s</td><td class=\"num\">%d</td><td>%s</td><td><a href=\"/stock/%d\">%s</a></td></tr>" % (
        pick(r, ["BK", "EL", "HM", "TY"]), r.randint(1, 9999), pick(r, CITY).title(), pick(r, NOUN), r.randint(0, 4000), pick(r, ["ok", "low", "backorder"]), 3000 + i, pick(r, ["details", "history", "reorder"]))


def k_changelog(r, i):
    return "## [3.%d.%d] - %s\n### %s\n- %s %s %s in the %s service (#%d)\n- %s %s %s for %s tenants (#%d)\n" % (
        40 - i // 3, 2 - i % 3, r.day, pick(r, ["Fixed", "Added", "Changed"]), pick(r, ["Fixed", "Reworked", "Dropped"]), pick(r, NOUN), pick(r, ["validation", "pagination", "caching", "retries"]),
        pick(r, SVC), r.randint(1000, 9999), pick(r, ["Added", "Enabled", "Tuned"]), pick(r, NOUN), pick(r, ["export", "webhooks", "rate limits", "audit trail"]), pick(r, REGION), r.randint(1000, 9999))


def k_openapi(r, i):
    n = NOUN[i % 20]
    v = ["get", "post", "delete", "patch"][i // 20 % 4]
    return "  /v%d/%ss/{id}%s:\n    %s:\n      operationId: %s%s%d\n      tags: [%s]\n      parameters:\n        - name: id\n          in: path\n          required: true\n          schema:\n            type: %s\n      responses:\n        \"%d\":\n          description: %s %s\n" % (
        i // 80 + 1, n, pick(r, ["", "/archive", "/history", "/lock"]), v, v, camel(n), i, pick(r, SVC), pick(r, ["string", "integer"]), pick(r, [200, 201, 204]), pick(r, PAST).title(), n)


def k_gha(r, i):
    lang, cmd = pick(r, [("node", "npm ci && npm run %s" % pick(r, ["lint", "test", "build", "typecheck"])), ("python", "pip install -r requirements.txt && pytest tests/%s -q" % pick(r, SVC)), ("go", "go test ./%s/..." % pick(r, SVC))])
    return "  %s:\n    runs-on: ubuntu-%s\n    timeout-minutes: %d\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-%s@v5\n      - run: %s\n        env:\n          %s_ENV: %s\n" % (
        uniq(i), pick(r, ["22.04", "24.04", "latest"]), r.randint(5, 45), lang, cmd, pick(r, SVC).upper(), pick(r, ENVS))


def k_cron(r, i):
    return "%s %s * * %s /opt/jobs/%s_%s.sh >> /var/log/jobs/%s.log 2>&1" % (
        pick(r, ["*/5", "*/15", "0", "30", str(r.randint(1, 59))]), pick(r, ["*", "*", str(r.randint(0, 23)), "*/6"]), pick(r, ["*", "*", "1-5", "0"]), pick(r, VERB), pick(r, NOUN) + "s", SVC[i % 16] + str(i // 16))


def k_pglog(r, i):
    pid = r.randint(2000, 64000)
    m = pick(r, [
        "LOG:  duration: %.3f ms  statement: SELECT %s FROM %ss WHERE %s = %d" % (r.uniform(200, 9000), ", ".join(r.sample(FIELD, 3)), pick(r, NOUN), pick(r, FIELD), r.randint(1, 99999)),
        "LOG:  checkpoint complete: wrote %d buffers (%.1f%%); write=%.3f s, sync=%.3f s" % (r.randint(100, 90000), r.uniform(0.1, 30), r.uniform(1, 270), r.uniform(0.001, 2)),
        "LOG:  connection authorized: user=%s_app database=%s_db application_name=%s" % (pick(r, SVC), pick(r, SVC), pick(r, ["psql", "pgbouncer", "sqlalchemy", "HikariPool-1"])),
        'ERROR:  duplicate key value violates unique constraint "%ss_%s_key"' % (pick(r, NOUN), pick(r, FIELD)),
        'LOG:  automatic analyze of table "%s_db.public.%ss" system usage: CPU: user: %.2f s' % (pick(r, SVC), pick(r, NOUN), r.uniform(0, 9)),
        'LOG:  temporary file: path "base/pgsql_tmp/pgsql_tmp%d.%d", size %d' % (pid, r.randint(0, 9), r.randint(10 ** 6, 10 ** 9)),
    ])
    return "%s %s.%03d UTC [%d] %s" % (r.day, tm(r, i), r.randint(0, 999), pid, m)


def k_redis(r, i):
    k = "%s:%s" % (pick(r, NOUN), hx(r, 10))
    c = pick(r, [
        '"GET" "%s"' % k,
        '"SET" "%s" "%s" "EX" "%d"' % (k, hx(r, 16), r.randint(30, 3600)),
        '"HINCRBY" "stats:%s" "%s" "1"' % (pick(r, SVC), pick(r, VERB)),
        '"EXPIRE" "%s" "%d"' % (k, r.randint(30, 900)),
        '"LPUSH" "queue:%s" "%s"' % (pick(r, SVC), hx(r, 12)),
        '"DEL" "%s"' % k,
        '"ZADD" "leaderboard:%s" "%d" "%s"' % (pick(r, CITY), r.randint(1, 99999), who(r)),
    ])
    return "%d.%06d [0 %s:%d] %s" % (1790000000 + r.base + i * 3, r.randint(0, 999999), ip(r), r.randint(30000, 65000), c)


def k_kafka(r, i):
    s = SVC[i // 12 % 16]
    end = r.randint(10 ** 5, 10 ** 8)
    lag = pick(r, [0, 0, 0, r.randint(1, 40), r.randint(100, 5000)])
    return "%-18s %-18s %-3d %-10d %-10d %-6d consumer-%s-%d-%s /%s" % (s + "-consumer", s + ".events", i % 12, end - lag, end, lag, s, i % 4 + 1, hx(r, 8), ip(r))


def k_ipt(r, i):
    return pick(r, [
        "-A INPUT -s %s/%d -p tcp -m tcp --dport %d -j ACCEPT" % (ip(r), pick(r, [16, 24, 32]), pick(r, PORTS)),
        '-A INPUT -p tcp -m tcp --dport %d -m comment --comment "%s %s" -j ACCEPT' % (pick(r, PORTS), pick(r, SVC), pick(r, ENVS)),
        "-A INPUT -s %s/32 -j DROP" % ip(r),
        "-A FORWARD -i eth%d -o docker0 -p tcp -m tcp --dport %d -j ACCEPT" % (r.randint(0, 2), pick(r, PORTS)),
        "-A INPUT -s %s/24 -p udp -m udp --dport %d -j ACCEPT" % (ip(r), pick(r, [53, 123, 514, 4789])),
        '-A INPUT -p tcp -m tcp --dport %d -m limit --limit %d/min -j LOG --log-prefix "%s-drop: "' % (pick(r, PORTS), r.randint(1, 30), pick(r, SVC)),
    ])


def k_docker(r, i):
    c = pick(r, [
        "RUN apt-get install -y --no-install-recommends %s" % " ".join(r.sample(PKG, 3)),
        "COPY %s/ /app/%s/" % ((pick(r, SVC),) * 2),
        "RUN pip install --no-cache-dir -r requirements/%s.txt" % pick(r, ENVS),
        "RUN npm run build:%s" % pick(r, SVC),
        "RUN go build -o /out/%s ./cmd/%s" % ((pick(r, SVC),) * 2),
    ])
    return "#%d [%s %d/%d] %s\n#%d %s" % (i + 5, pick(r, ["builder", "runtime", "deps"]), i % 14 + 1, 14, c, i + 5, pick(r, ["CACHED", "DONE %.1fs" % r.uniform(0.1, 90)]))


def k_mvn(r, i):
    g = pick(r, ["org.apache.commons", "com.fasterxml.jackson.core", "io.netty", "org.springframework", "org.slf4j", "com.google.guava", "org.hibernate.orm"])
    a = "%s-%s" % (pick(r, NOUN), pick(r, ["core", "api", "codec", "util", "annotations"]))
    v = "%d.%d.%d" % (r.randint(1, 6), r.randint(0, 30), r.randint(0, 9))
    return "[INFO] Downloaded from central: https://repo.maven.apache.org/maven2/%s/%s/%s/%s-%s.jar (%d kB at %d kB/s)" % (g.replace(".", "/"), a, v, a, v, r.randint(8, 4000), r.randint(90, 9000))


def k_nginxconf(r, i):
    s = uniq(i)
    return "server {\n    listen 443 ssl;\n    server_name %s.%s.example;\n    ssl_certificate /etc/ssl/%s.pem;\n    client_max_body_size %dm;\n\n    location / {\n        proxy_pass http://%s_upstream;\n        proxy_read_timeout %ds;\n        proxy_set_header Host $host;\n    }\n}\n" % (
        s, pick(r, CITY), s, pick(r, [1, 8, 20, 100]), s.replace("-", "_"), pick(r, [15, 30, 60, 120]))


def k_strace(r, i):
    fd = r.randint(3, 40)
    c = pick(r, [
        'openat(AT_FDCWD, "/srv/%s/%s.%s", O_RDONLY|O_CLOEXEC) = %d' % (pick(r, SVC), pick(r, NOUN), pick(r, ["conf", "so", "json", "pem"]), fd),
        'read(%d, "%s"..., 4096) = %d' % (fd, hx(r, 24), r.randint(1, 4096)),
        "epoll_wait(4, [{events=EPOLLIN, data={u32=%d, u64=%d}}], 64, %d) = 1" % (fd, fd, pick(r, [-1, 100, 1000])),
        'write(%d, "%s %s %d\\n", %d) = %d' % (fd, pick(r, VERB), pick(r, NOUN), r.randint(1, 9999), 24, 24),
        "recvfrom(%d, 0x%s, 8192, 0, NULL, NULL) = -1 EAGAIN (Resource temporarily unavailable)" % (fd, hx(r, 12)),
        "close(%d) = 0" % fd,
        "futex(0x%s, FUTEX_WAKE_PRIVATE, 1) = %d" % (hx(r, 12), r.randint(0, 1)),
        'stat("/etc/%s/%s.conf", {st_mode=S_IFREG|0644, st_size=%d, ...}) = 0' % (pick(r, SVC), pick(r, NOUN), r.randint(40, 9000)),
    ])
    return "%d %s.%06d %s" % (r.base % 30000 + 1200, tm(r, i, 2), r.randint(0, 999999), c)


def k_mddoc(r, i):
    n = NOUN[i % 20]
    v = ["GET", "POST", "DELETE", "PATCH"][i // 20 % 4]
    f = r.sample(FIELD, 2)
    return "### %s /v1/%ss/{id}%s\n\n%s a %s. Requires the `%s:%s` scope. Rate limit: %d requests per minute.\n\n| Parameter | Type | Description |\n|---|---|---|\n| `%s` | %s | %s the %s by %s |\n| `%s` | %s | Optional %s override |\n" % (
        v, n, pick(r, ["", "/archive", "/history"]), pick(r, ["Returns", "Modifies", "Removes", "Locks"]), n, n, pick(r, ["read", "write", "admin"]), pick(r, [30, 60, 120, 600]),
        f[0], FT[f[0]][1], pick(r, ["Filters", "Sorts", "Scopes"]), n, f[0].replace("_", " "), f[1], FT[f[1]][1], f[1].replace("_", " "))


def k_promrule(r, i):
    s = SVC[i % 16]
    a, e = pick(r, [
        ("HighLatency", 'histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket{service="%s"}[5m]))) > %.1f' % (s, r.uniform(0.3, 3))),
        ("PoolSaturated", 'db_pool_connections_in_use{service="%s"} / db_pool_connections_max{service="%s"} > 0.%d' % (s, s, r.randint(7, 9))),
        ("QueueBacklog", 'sum(queue_depth{service="%s"}) > %d' % (s, r.randint(500, 9000))),
        ("RestartLoop", 'increase(kube_pod_container_status_restarts_total{container="%s"}[15m]) > %d' % (s, r.randint(2, 6))),
    ])
    return "      - alert: %s%s%d\n        expr: %s\n        for: %dm\n        labels:\n          severity: %s\n          team: %s\n        annotations:\n          summary: \"%s %s on {{ $labels.instance }}\"\n" % (
        camel(s), a, i // 16, e, pick(r, [2, 5, 10, 15]), pick(r, ["page", "ticket", "info"]), pick(r, CITY), s, a)


def k_dmesg(r, i):
    m = pick(r, [
        "usb %d-%d: new high-speed USB device number %d using xhci_hcd" % (r.randint(1, 4), r.randint(1, 8), r.randint(2, 40)),
        "EXT4-fs (nvme0n1p%d): mounted filesystem with ordered data mode. Quota mode: none." % r.randint(1, 6),
        "eth%d: link up, %d Mbps, full duplex" % (r.randint(0, 3), pick(r, [100, 1000, 10000])),
        "audit: type=1400 audit(%d.%03d:%d): apparmor=\"DENIED\" operation=\"open\" profile=\"%s\" pid=%d" % (1790000000 + i, r.randint(0, 999), r.randint(100, 9000), pick(r, SVC), r.randint(900, 60000)),
        "nvme nvme0: I/O %d QID %d timeout, completion polled" % (r.randint(1, 900), r.randint(1, 32)),
        "TCP: request_sock_TCP: Possible SYN flooding on port %d. Sending cookies." % pick(r, PORTS),
        "perf: interrupt took too long (%d > %d), lowering kernel.perf_event_max_sample_rate to %d" % (r.randint(2600, 9000), r.randint(2000, 2500), r.randint(20000, 79000)),
        "br-%s: port %d(veth%s) entered forwarding state" % (hx(r, 12), r.randint(1, 30), hx(r, 7)),
    ])
    return "[%12.6f] %s" % (r.base / 10 + i * 3.7 + r.random(), m)


def k_zone(r, i):
    ty, v = pick(r, [("A", ip(r)), ("A", ip(r)), ("CNAME", "%s.%s.example." % (pick(r, SVC), pick(r, CITY))), ("TXT", '"v=%s%s"' % (pick(r, ["spf1 include:_spf.", "verify-"]), hx(r, 12))), ("MX", "%d mx%d.%s.example." % (pick(r, [10, 20]), r.randint(1, 4), pick(r, CITY)))])
    return "%-26s %-5d IN %-5s %s" % ("%s-%s%d" % (SVC[i % 16], pick(r, CITY), i // 16), pick(r, [60, 300, 3600]), ty, v)


def k_props(r, i):
    v, n = pick(r, VERB), NOUN[i % 20]
    return "%s.%s.%s_%s%s=%s %s%s" % (SVC[i // 20 % 16], pick(r, ["button", "title", "tooltip", "menu"]), v, n, "" if i < 320 else str(i), v.title(), n, pick(r, ["", "", "...", " now"]))


K = {
    "nginx": (k_nginx, "", "", "\n"),
    "syslog": (k_syslog, "", "", "\n"),
    "applog": (k_applog, "", "", "\n"),
    "jsonlog": (k_jsonlog, "", "", "\n"),
    "json": (k_json, "[\n", "\n]", ",\n"),
    "k8sev": (k_k8sev, "LAST SEEN   TYPE      REASON        OBJECT    MESSAGE\n", "", "\n"),
    "csv": (k_csv, "order_id,customer_email,sku,qty,unit_price,currency,country,status,placed_at\n", "", "\n"),
    "sqlins": (k_sqlins, "", "", "\n"),
    "gitlog": (k_gitlog, "", "", "\n"),
    "gosum": (k_gosum, "", "", "\n"),
    "prom": (k_prom, "", "", "\n"),
    "ps": (k_ps, "USER         PID %CPU %MEM      VSZ     RSS TTY      STAT START   TIME COMMAND\n", "", "\n"),
    "pytest": (k_pytest, "", "", "\n"),
    "gotest": (k_gotest, "", "", "\n"),
    "tsc": (k_tsc, "", "", "\n"),
    "cargo": (k_cargo, "", "", "\n"),
    "javatrace": (k_javatrace, "org.springframework.dao.DataAccessResourceFailureException: Unable to acquire JDBC Connection; nested exception is java.sql.SQLTransientConnectionException: HikariPool-1 - Connection is not available, request timed out after 30000ms.\n", "", "\n"),
    "compose": (k_compose, "services:\n", "", "\n"),
    "k8s": (k_k8s, "", "", "\n---\n"),
    "tf": (k_tf, "", "", "\n"),
    "tfplan": (k_tfplan, "Terraform will perform the following actions:\n\n", "", "\n"),
    "env": (k_env, "", "", "\n"),
    "ini": (k_ini, "", "", "\n"),
    "java": (k_java, "package com.acme.model;\n\nimport java.math.BigDecimal;\nimport java.time.Instant;\nimport java.util.List;\n\npublic class CustomerRecord {\n\n", "}", "\n"),
    "ts": (k_ts, "", "", "\n"),
    "go": (k_go, "package model\n\nimport \"time\"\n\n", "", "\n"),
    "ch": (k_ch, "#ifndef BOARD_REGS_H\n#define BOARD_REGS_H\n\n#include <stdint.h>\n\n", "\n\n#endif", "\n"),
    "proto": (k_proto, "syntax = \"proto3\";\n\npackage acme.fulfilment.v1;\n\n", "", "\n"),
    "ddl": (k_ddl, "", "", "\n"),
    "html": (k_html, "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>Warehouse stock report</title>\n</head>\n<body>\n<table>\n<tr><th>SKU</th><th>Item</th><th>On hand</th><th>State</th><th></th></tr>\n", "\n</table>\n</body>\n</html>", "\n"),
    "changelog": (k_changelog, "# Changelog\n\n", "", "\n"),
    "openapi": (k_openapi, "openapi: 3.0.3\ninfo:\n  title: Fulfilment API\n  version: 4.2.0\npaths:\n", "", ""),
    "gha": (k_gha, "name: ci\non:\n  push:\n    branches: [main]\n  pull_request:\njobs:\n", "", ""),
    "cron": (k_cron, "SHELL=/bin/bash\nMAILTO=ops@acme.example\n", "", "\n"),
    "pglog": (k_pglog, "", "", "\n"),
    "redis": (k_redis, "", "", "\n"),
    "kafka": (k_kafka, "GROUP              TOPIC              PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG    CONSUMER-ID    HOST\n", "", "\n"),
    "ipt": (k_ipt, "*filter\n:INPUT DROP [0:0]\n:FORWARD DROP [0:0]\n:OUTPUT ACCEPT [18231:2291822]\n-A INPUT -i lo -j ACCEPT\n", "\nCOMMIT", "\n"),
    "docker": (k_docker, "", "", "\n"),
    "mvn": (k_mvn, "", "", "\n"),
    "nginxconf": (k_nginxconf, "", "", "\n"),
    "strace": (k_strace, "", "", "\n"),
    "mddoc": (k_mddoc, "", "", "\n"),
    "promrule": (k_promrule, "groups:\n  - name: service-alerts\n    rules:\n", "", ""),
    "dmesg": (k_dmesg, "", "", "\n"),
    "zone": (k_zone, "$ORIGIN acme.example.\n$TTL 300\n", "", "\n"),
    "props": (k_props, "", "", "\n"),
}


def build(kind, seed, n, *needles, head=None, tail=None):
    fn, h, t, sep = K[kind]
    h = h if head is None else head
    t = t if tail is None else tail
    r = mk(seed)
    n = max(n - len(h) - len(t), 200)
    marks = sorted(r.uniform(0.2, 0.8) * n for _ in needles)
    out, size, i, k = [], 0, 0, 0
    while size < n or k < len(needles):
        if k < len(needles) and size >= marks[k]:
            it = needles[k](r, i) if callable(needles[k]) else needles[k]
            k += 1
        else:
            it = fn(r, i)
        out.append(it)
        size += len(it) + len(sep)
        i += 1
    return h + sep.join(out) + t
