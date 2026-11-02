"""离线重放 n-gram 草稿源：只用已生成的 token 序列，量它的收益，不碰链路。

这是 `docs/SPECULATION.md` §6 要求的「先量收益再上」的实现。它回答两个问题，且只回答
这两个：

  1. **m_eff** —— 如果每轮用 n-gram 起草，平均每轮能前进几个 token。盈亏线
     m >= 2.5（`docs/SPECULATION.md:166`）。口径必须与 §6 的 2.56-3.37 相同：那是
     **每轮前进倍数**，不是草稿命中率——两个数字混用会得出相反结论。
  2. **回扫成本** —— 匹配本身要扫多少历史。这是将来运行时影子模式的开销上界。

算法是 `H:/llamacpp-c/common/ngram-map.cpp:49-113` 的 `common_ngram_simple_draft` 的
逐行翻译（`ngram_simple_draft` 与它一一对应，行号写在函数体注释里）。选它而不选
ngram-map-k / k4v / mod / cache 的理由见 `docs/SPECULATION.md` §6 与本脚本的
`PARAMS`：本项目的 k 是 2-5，而上游那几套的默认值是 m=24-64（`common/common.h:361-372`），
它们是给「一次验 50 行、batch 摊薄成本」的场景调的，而本项目 lm_head 按行线性收钱。

输入是 `checks/verify_equiv.py --mode chain --json-out ...` 的产物：`prompt_ids` 是
prompt 的 token，`ids` 是 greedy 生成出来的 token。greedy 下真实解码就是逐 token 产出
`ids`，所以「如果当时有 n-gram 草稿」可以完全离线复算：每轮的历史是 gold 的确定前缀，
不需要跑模型。

用法：
    python checks/probe_ngram_replay.py .qlh_tmp/ngram-gold-64.json .qlh_tmp/ngram-gold-128.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

# (size_n, size_m) 的扫描网格。size_n 是**含 anchor** 的 pattern 长度
# （`ngram-map.cpp:66-70` 把 sampled 也 push 进 pattern），size_m 是草稿上限。
#
# 上游默认 size_n=12 / size_m=48（`H:/llamacpp-c/common/common.h:369-370`）。这里扫
# 2-8 / 1-8，是因为本项目的 k 上限是 5：验证块每多一行就多付一次 lm_head
# （4.4-5.0 ms/行，`docs/SPECULATION.md:72`），m=48 要验 49 行，光是 lm_head 就 200 ms+。
N_GRID = (2, 3, 4, 5, 6, 8)
M_GRID = (1, 2, 3, 4, 5, 6, 7, 8)


def ngram_simple_draft(tokens: list[int], sampled: int, size_n: int, size_m: int) -> list[int]:
    """Return up to `size_m` draft tokens, translated from `ngram-map.cpp:49-113`.

    `tokens` is the token history *without* `sampled`; `sampled` is the token that was just
    produced and is the anchor the draft extends from. This mirrors the C++ signature
    `common_ngram_simple_draft(config, tokens, sampled)` exactly, including its two
    conservative guards, which look arbitrary but are load-bearing for the numbers:

      * `cur_len <= size_n + size_m + 1` -> no draft (`:62-65`)
      * `copy_max < size_n` -> no draft (`:96-98`): a match whose tail is shorter than the
        lookup n-gram is not trusted. With size_n=4 this means the match must have at least
        4 tokens after it even though only 3 are drafted.

    The scan walks backwards from `cur_len - size_n - 1` down to index 1 and takes the
    **nearest** match; index 0 is deliberately skipped (`:71-77`, "we ignore position 0,
    position 0 == no match"). Nearest-match is what makes code continuation work: the last
    time this identifier/indent block appeared is the best predictor of what follows.
    """
    cur_len = len(tokens)
    if cur_len <= size_n + size_m + 1:
        return []

    # pattern = the last size_n-1 history tokens + the sampled token (`:66-71`).
    pattern = tokens[cur_len - size_n + 1:] + [sampled]

    match_pos = 0
    for j in range(cur_len - size_n - 1, 0, -1):
        # Element-wise, like the C++ inner loop (`:73-80`); a slice compare would allocate
        # and would understate the scan cost we are trying to measure.
        matched = True
        for k in range(size_n):
            if tokens[j + k] != pattern[k]:
                matched = False
                break
        if matched:
            match_pos = j
            break
    if match_pos == 0:
        return []

    copy_max = min(size_m, cur_len - (match_pos + size_n))       # `:88-91`
    if copy_max < size_n:                                        # `:96-98`
        return []
    return tokens[match_pos + size_n: match_pos + size_n + copy_max]


def _selftest() -> None:
    """Pin the three behaviours the translation has to preserve.

    Every number in the report comes out of `ngram_simple_draft`, so a silent divergence
    here would invalidate the whole verdict. These are the three places the C++ has
    non-obvious behaviour; each is a hand-computed example, not a round-trip.
    """
    # (1) m < n can never draft: copy_max = min(m, avail) <= m < n trips `:96-98`.
    #     This is why n=4/m=3 measured 0.0% hit rate -- it is not "poor coverage".
    assert ngram_simple_draft([1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8], 4, 4, 3) == []

    # (2) A match with room drafts the tokens that followed it. Hand-computed: the only
    #     [20,30,40] that starts at index >= 1 is at 1, so the draft is tokens[4:8].
    assert ngram_simple_draft([50, 20, 30, 40, 60, 70, 80, 20, 30], 40, 3, 4) \
        == [60, 70, 80, 20]

    # (3) A match starting at index 0 is ignored (`:71-77`), so this returns nothing.
    assert ngram_simple_draft([20, 30, 40, 55, 66, 77, 88, 20, 30], 40, 3, 4) == []

    # (4) Not enough history to look back into (`:62-65`).
    assert ngram_simple_draft([1, 2, 3, 4], 5, 3, 4) == []


def replay(gold: list[int], prompt_len: int, size_n: int, size_m: int) -> dict:
    """Rolling ledger: simulate one greedy decode pass that used n-gram drafts at every round.

    Positions follow `chain_client.forward_batch`: a round starts with the anchor already
    fixed (the token plain decode would produce next) and submits `[anchor] + drafts` as one
    block. Accepted prefix length `accept` means the block legitimately wrote `1 + accept`
    tokens, and the row after the accepted prefix is free -- the same accounting as
    `DFlashDriver.step` (`chain_client.py:1038-1041`) and `speculate()` (`:894-895`).

    The gold sequence is fixed because greedy decoding is deterministic and a correct draft
    only ever reproduces it; that is what lets this run without a model.

    `m_eff = total_advance / rounds` is the "tokens advanced per round" multiple, the number
    `docs/SPECULATION.md:166` compares against 2.5. It is NOT an acceptance rate: 1.0 means
    "same as plain decode", and every round advances at least 1.
    """
    p = prompt_len
    rounds = total_advance = 0
    hit_rounds = draft_total = accept_total = 0
    per_round = []
    n_gold = len(gold)
    while p < n_gold:
        history = gold[:p]
        drafts = ngram_simple_draft(history, gold[p], size_n, size_m)
        if drafts:
            hit_rounds += 1
        accept = 0
        for j, d in enumerate(drafts):
            if p + 1 + j < n_gold and gold[p + 1 + j] == d:
                accept += 1
            else:
                break
        # 1 (anchor -> the free token) + accepted drafts. Truncated at the end of `gold`,
        # where the free token was never generated; the last round of a run is the only one
        # this can touch.
        step = min(1 + accept, n_gold - p)
        per_round.append(step)
        rounds += 1
        total_advance += step
        draft_total += len(drafts)
        accept_total += accept
        p += step
    return {
        "rounds": rounds,
        "advance": total_advance,
        "m_eff": total_advance / rounds if rounds else 0.0,
        "hit_rate": hit_rounds / rounds if rounds else 0.0,
        "drafts": draft_total,
        "accepted": accept_total,
        # Conditional acceptance: of the drafts actually proposed, how many survived.
        "accept_of_drafts": accept_total / draft_total if draft_total else 0.0,
        "mean_drafts_when_hit": draft_total / hit_rounds if hit_rounds else 0.0,
        "mean_advance_when_hit": (
            sum(s for s in per_round if s > 1) / hit_rounds if hit_rounds else 0.0),
    }


def scan(gold: list[int], prompt_len: int) -> list[dict]:
    rows = []
    for size_n in N_GRID:
        for size_m in M_GRID:
            r = replay(gold, prompt_len, size_n, size_m)
            r.update({"n": size_n, "m": size_m})
            rows.append(r)
    return rows


def cost_probe(gold: list[int], repeat: int = 200) -> dict:
    """How long one draft lookup takes, as a function of history length.

    Two readings are reported because "cost" means two different things here:

      * **per lookup at ctx length** -- the marginal cost added to every decode step once
        shadow mode is live. History is capped by ctx (7168 here, `config.ctx()`).
      * **100k lookups at 4096 history** -- the accumulated cost of 100k decode steps, which
        is what "10 万 token 的累计时间" asks for. It is NOT a 100k-token generation: a real
        generation grows history linearly, making the total O(N^2) and the number useless.
    """
    # A long synthetic history with the same repetition structure as the gold sequence: the
    # scan cost depends on history length, and near-matches early in the scan only make it
    # stop sooner. Repeating gold is the honest shape for code (repetition is the point).
    base = list(gold)
    long_hist: list[int] = []
    while len(long_hist) < 100_000:
        long_hist.extend(base)

    out: dict = {}
    for hist_len in (1024, 4096, 7168, 65536, 100_000):
        hist = long_hist[:hist_len]
        sampled = long_hist[hist_len]
        # Worst case for the scan: no match anywhere, so it walks every position. An anchor
        # that never appears guarantees it; with a repeating history the first match would
        # short-circuit and the measurement would say nothing about the upper bound.
        anchor = max(hist) + 1
        t0 = time.perf_counter()
        for _ in range(repeat):
            ngram_simple_draft(hist, anchor, 4, 3)
        dt = (time.perf_counter() - t0) / repeat * 1000.0
        out[f"ms_per_lookup_h{hist_len}"] = round(dt, 4)

    hist = long_hist[:4096]
    t0 = time.perf_counter()
    for i in range(100_000):
        ngram_simple_draft(hist, long_hist[4096 + (i & 0x3F)], 4, 3)
    out["s_for_100k_lookups_h4096"] = round(time.perf_counter() - t0, 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", nargs="+", help="verify_equiv.py --mode chain 的 json 产物")
    ap.add_argument("--top", type=int, default=6, help="每个序列打印前多少个 (n,m) 组合")
    ap.add_argument("--cost", action="store_true", help="附带测回扫成本")
    args = ap.parse_args()

    _selftest()

    for path_str in args.gold:
        path = Path(path_str)
        blob = json.loads(path.read_text(encoding="utf-8"))
        prompt = [int(t) for t in blob["prompt_ids"]]
        ids = [int(t) for t in blob["ids"]]
        gold = prompt + ids
        print(f"\n=== {path.name}: prompt {len(prompt)} + gen {len(ids)} = "
              f"{len(gold)} tokens, topology {blob.get('topology')} ===")
        print(f"    text tail: {blob.get('text', '')[-60:]!r}")

        rows = scan(gold, len(prompt))
        rows.sort(key=lambda r: r["m_eff"], reverse=True)
        print(f"    {'n':>2} {'m':>2} {'m_eff':>6} {'命中率':>7} {'接受/草稿':>9} "
              f"{'草稿数/轮':>9} {'轮数':>5}")
        for r in rows[: args.top]:
            print(f"    {r['n']:>2} {r['m']:>2} {r['m_eff']:>6.2f} "
                  f"{100*r['hit_rate']:>6.1f}% {100*r['accept_of_drafts']:>8.1f}% "
                  f"{r['mean_drafts_when_hit']:>8.2f} {r['rounds']:>5}")
        # The chosen default (n=4, m=3) must be reported explicitly even when it is not in
        # the top rows: it is the parameter the runtime path will use.
        chosen = [r for r in rows if r["n"] == 4 and r["m"] == 3][0]
        print(f"    选定 n=4 m=3: m_eff={chosen['m_eff']:.2f} 命中率 "
              f"{100*chosen['hit_rate']:.1f}% 接受/草稿 {100*chosen['accept_of_drafts']:.1f}% "
              f"轮数 {chosen['rounds']}  判决 m>=2.5: "
              f"{'PASS' if chosen['m_eff'] >= 2.5 else 'FAIL'}")

        if args.cost:
            print("    回扫成本:", json.dumps(cost_probe(gold), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
