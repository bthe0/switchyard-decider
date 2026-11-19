import numpy as np

from switchyard_decider.serve import MAX_LEN, TASK_CHARS, encode, hardness, head_tail, p_solve, request_text

rng = np.random.default_rng(0)
VAL = rng.normal(size=(500, 7)).astype(np.float32)


def make_calib(z, out=None):
    e = np.exp(z - z.max(1, keepdims=True))
    pa, pj = e[:, :3] / e[:, :3].sum(1, keepdims=True), e[:, 3:6] / e[:, 3:6].sum(1, keepdims=True)
    pr = (pa + pj) / 2
    aj, g = np.sort(pr[:, 2]), np.sort(-z[:, 6])
    h = (np.searchsorted(aj, pr[:, 2]) + np.searchsorted(g, -z[:, 6])) / 2 / len(aj)
    return {"aj": aj, "g": g, "h": np.sort(h), "out": np.linspace(0, 1, len(h)) if out is None else np.sort(out)}


CALIB = make_calib(VAL)


def logits(hard, gold):
    return np.array([0, 0, hard, 0, 0, hard, gold], dtype=np.float32)


def test_hardness_monotonic_in_hard_heads():
    hs = [hardness(logits(x, 0.0), CALIB) for x in np.linspace(-4, 4, 9)]
    assert all(a <= b for a, b in zip(hs, hs[1:])) and hs[0] < hs[-1]


def test_hardness_decreases_with_gold_logit():
    hs = [hardness(logits(0.0, x), CALIB) for x in np.linspace(-4, 4, 9)]
    assert all(a >= b for a, b in zip(hs, hs[1:])) and hs[0] > hs[-1]


def test_hardness_range():
    assert hardness(logits(-50, 50), CALIB) == 0.0
    assert hardness(logits(50, -50), CALIB) == 1.0


def test_p_solve_range_and_monotonic():
    ps = [p_solve(h, CALIB) for h in np.linspace(0, 1, 21)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(a >= b for a, b in zip(ps, ps[1:])) and ps[0] > ps[-1]


def test_p_solve_follows_target_distribution():
    calib = make_calib(VAL, out=rng.uniform(0.2, 0.6, size=500))
    ps = [p_solve(h, calib) for h in np.linspace(0, 1, 21)]
    assert min(ps) >= calib["out"].min() and max(ps) <= calib["out"].max()


def test_request_text():
    assert request_text({"opening_task": "fix bug", "latest_follow_up": "also tests"}) == "fix bug\n\nalso tests"
    assert request_text({"opening_task": "fix bug"}) == "fix bug"
    assert request_text({"latest_follow_up": "more"}) == "more"
    assert request_text({}) == ""


def test_request_text_truncates_opening_task_only():
    t = request_text({"opening_task": "a" * (TASK_CHARS + 5000), "latest_follow_up": "b" * 5000})
    assert t == "a" * TASK_CHARS + "\n\n" + "b" * 5000


def test_head_tail_keeps_first_and_last_halves():
    ids = list(range(2000))
    cut = head_tail(ids)
    assert MAX_LEN == 768 and len(cut) == 768
    assert cut[:384] == ids[:384] and cut[384:] == ids[-384:]


def test_head_tail_short_unchanged():
    for n in (0, 1, 767, 768):
        assert head_tail(list(range(n))) == list(range(n))


def test_encode_cuts_chars_then_tokens():
    seen = []

    def tok(text):
        seen.append(len(text))
        return {"input_ids": [ord(ch) for ch in text]}

    ids = encode(tok, "a" * 384 + "x" * (TASK_CHARS * 2) + "z" * 384)
    assert seen == [TASK_CHARS] and len(ids) == 768
    assert ids[:384] == [ord("a")] * 384 and ids[384:] == [ord("x")] * 384


def test_ridge_probe_blend():
    rs = np.linspace(-1, 1, 101).astype(np.float32)
    calib = dict(CALIB, ra=np.ones(4, np.float32), rb=np.float32(0), rs=rs, rw=np.float32(4))
    lg = logits(0.0, 0.0)
    assert hardness(lg, calib) == hardness(lg, CALIB)
    hs = [hardness(lg, calib, np.full(4, x / 4, np.float32)) for x in np.linspace(-1.5, 1.5, 7)]
    assert all(a <= b for a, b in zip(hs, hs[1:])) and 0.0 <= hs[0] < hs[-1] <= 1.0
