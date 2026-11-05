"""多轮对话验收：KV 连续增长时，链路是否仍等价于「从头重放整段历史」。

为什么要这个脚本：`chain_client` 的 `n_past` 只在**单轮内部**被累加过，跨轮的落点是零：

  * `chain_client.main()` 每跑完一次 `--repeat` 就 `reset()`，prompt 永远从位置 0 起；
  * `server.py` 的模块 docstring 写明「There is no multi-turn memory」，每个请求 prefill
    自己那一份完整 messages；
  * `stream.generate_stream` 原先也固定 `forward(..., 0)`。

所以「KV 里已经有 N 个位置、这一轮从 N 接着写」这条路径在本仓库里**一次都没跑过**，而它
正是任何真多轮对话的地基。光看「没崩」不算证据：位置错一位、recurrent 状态没跟上、ubatch
边界上的快照没滚动，模型照样会吐出一段「看起来像话」的文本。

判据由弱到强，各自抓不同的失败：

  R1  n_past 单调     每一轮 prefill 的起点 == 上一轮生成结束的终点。
  R2  每轮都出 token  空输出说明 prefill 之后的位置已经被写坏，argmax 拿到的是垃圾。
  R3  记忆            第 2 轮问「刚给你的暗号是什么」，输出里必须出现第 1 轮才给过的串。
      反向对照（丢掉第 1 轮、只喂第 2 轮）必须**不**出现——否则这个串可能是模型自己
      编的，R3 就不构成证据。
  R4  等价（最强）      续用 KV 生成的 token 序列，与 reset 后把同一段历史从头 prefill
                      再生成的结果**逐位相同**。这直接证明「增量 KV」与「重放」是同一条
                      计算，不依赖模型是否听话。

失败代价说明：R4 比 R3 强，但 R4 会受「分批 prefill vs 逐 token 写入」的浮点累加顺序
影响。所以 R4 不一致时脚本不直接判死，而是报出首个分歧位置与两边的文本，由人判断那是
数值噪声还是真的语义分叉；R1/R2/R3 任一不过就是 FAIL。

用法（两段都在本机）：

    SA_TOPOLOGY=topology-local-32-8.toml python checks/verify_multiturn.py

退出码：0 全过；7 R1/R2/R3 有失败；8 R4 有分歧（数值或语义，需要人看报告）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# 仓库里本文件在 checks/ 下、模块在上一级；打进包里时它与 worker/ 同级。
# 判据用 protocol.py 落在哪，和 scripts/probe.py 用同一套，别再造第三份布局假设。
for _cand in (HERE.parent, HERE, HERE / "worker", HERE.parent / "worker"):
    if (_cand / "protocol.py").is_file():
        REPO = _cand
        break
sys.path.insert(0, str(REPO))

import config  # noqa: E402
import topology  # noqa: E402
import chain_client  # noqa: E402
from chain_client import StageClient, _terminate, bring_up, tail_rs_depth  # noqa: E402
from chat import apply_chatml, next_turn  # noqa: E402
from stream import generate_stream  # noqa: E402

# Worker 日志走一个本脚本专属的目录。`build/relay-records` 是共享路径，而本机可能同时
# 有另一个会话在跑别的拓扑（每段日志按 stage 索引固定命名），于是这次运行会覆盖那次的
# 日志、或者被那次覆盖——排障时看到的是别人的 20 层 2B，而不是自己这条 35B 链。
chain_client.LOG_DIR = REPO / "build" / "relay-records-multiturn"

#: 第 1 轮给、第 2 轮要复述的暗号。要选模型自己编不出来的东西才有证据力：常见的
#: 「苹果」「北京」这类词，模型在没上下文时也会张口就来，反例就失去意义。
SECRET = "靛蓝七号"

SYSTEM = "你是严谨的中文助手，回答尽量简短。"

#: (问题, 这一轮最多生成多少 token)。第一轮只要求「记住」，短预算够；第 2 轮模型要先
#: 想一段再回答，给得宽一点，否则 R3 会因为被截断而假阴性。
TURNS = [
    (f"请记住这个暗号：{SECRET}。等一下我会问你。", 96),
    ("我刚才让你记住的暗号是什么？直接说暗号。", 224),
    ("再确认一次，那个暗号是什么？", 160),
]


# --------------------------------------------------------------------------- 单轮驱动
def _turn(clients, delta_text: str, n_past: int, pending: int | None,
          max_tokens: int, spec_k: int) -> dict:
    """跑一轮：把 ``delta_text`` 接在 KV 已填到的位置后面，贪心生成至多 max_tokens。

    这里**不实现解码循环**：本仓库只允许一个（AGENTS §7），而 `stream.generate_stream`
    就是 TUI / server 共用的那一个。这个函数只负责把「增量 token + 起始位置」交给它，
    再把结果收回来——所以多轮能力是给既有循环加参数，不是新写一份。
    """
    tokens = clients[0].tokenize(delta_text)
    if pending is not None:
        # 上一轮最后 emit 出的那个 token 从没走过 forward（循环在 emit 之后就 break 了），
        # 但它的位置就是 n_past。把它接在增量最前面，KV 才与对话文本严格对齐。
        tokens = [pending] + tokens
    stats: dict = {}
    ids: list[int] = []
    text = ""
    for chunk in generate_stream(clients, "", prefix_ids=tokens, n_past=n_past,
                                 max_tokens=max_tokens, spec_k=spec_k, stats=stats):
        # 组装器收尾时会补一个 token=-1 的 chunk 专门送剩下的文本，它不是模型生成的 id。
        if chunk.token >= 0:
            ids.append(chunk.token)
        text += chunk.text
    tail_stats = stats.get("kv_tail") or []
    return {"prefill": tokens, "ids": ids, "text": text,
            "n_past_before": n_past, "n_past_after": stats["n_past"],
            "pending": stats["pending"], "orphans": stats.get("orphans") or [],
            "kv_tail": tail_stats,
            # 「KV 与对话文本对齐」：这一轮写进 KV 的生成位置，必须等于 emit 出去的 token
            # 数（pending 那个还没进 KV 时要减 1）加上 kv_tail（已进 KV 却没 emit 的）。
            # 投机块在块内被截断时，kv_tail 会非空——那是 llama.cpp 不允许回退造成的，
            # 不是本链路的错账，所以判据认它，但报告里单独列出来给人看。
            "kv_written": stats["n_past"] - n_past - len(tokens),
            "kv_expected": (len(ids) - (0 if stats["pending"] is None else 1)
                            + len(tail_stats)),
            "prefill_written": stats["n_past"] >= n_past + len(tokens)}


def _delta_text(index: int, question: str, pending_is_eog: bool) -> str:
    """把一问接成 ChatML 的下一个块。

    第 1 轮用 `chat.apply_chatml`，后续轮用 `chat.next_turn`——**唯一**的 ChatML 渲染器。
    这里原先自己拼 `<|im_start|>` 标记，是第二份渲染器（TUI 的多轮也要同一段文本，
    两边一定会漂）；现在两个入口调同一个函数。

    之所以不每轮都用 apply_chatml 渲染整段：那样就得把历史文本重新 token 化一遍，而生成
    出来的 token 序列经 文本→token 往返并不无损（字节级 BPE），KV 里留着的和重新编码出来
    的会悄悄错开。next_turn 只渲染**新增的部分**，历史原样留在 KV 里。

    `pending_is_eog`：上一轮若是自然停止，pending 就是 `<|im_end|>`，收尾已经写进 KV；
    被 max_tokens 截断时才需要补一个结束标记，否则模型会看到「回答直接被新的 user 块截断」。
    """
    if index == 0:
        return apply_chatml([{"role": "user", "content": question}], system=SYSTEM)
    return next_turn(question, assistant_closed=pending_is_eog)


def _decode(clients, ids: list[int]) -> str:
    return b"".join(clients[-1].piece(t) for t in ids).decode("utf-8", "replace")


# ------------------------------------------------------------------------------- 主流程
def run(args: argparse.Namespace) -> dict:
    stages = topology.load(config.topology_path())
    ctx = config.ctx()
    shim = Path(args.shim)
    if not shim.is_file():
        raise SystemExit(f"shim not found: {shim}")

    # ChatML 的 <|im_start|> 必须是 special token，否则 tokenize 会把它拆成普通片段，
    # 模型看到的就不是对话结构了。必须在 spawn 之前设：worker 继承启动时的环境。
    os.environ["SA_PARSE_SPECIAL"] = "1"
    os.environ.setdefault("SA_MEM_SAFETY", "0.88")

    procs: list = []
    clients = [StageClient(s) for s in stages]
    report: dict = {"topology": str(config.topology_path()), "model": str(stages[0].model),
                    "ctx": ctx, "spec_k": args.spec_k, "secret": SECRET, "rounds": []}
    try:
        bring_up(stages, shim, ctx, tail_rs_depth("", 0, 0), procs=procs)
        local = {stage.index: proc for proc, _log, stage in procs}
        for c in clients:
            c.connect(proc=local.get(c.stage.index))
        # 复用已有 worker 时（弱设备链上 tail 常年常驻）KV 里可能留着上一次的位置，
        # 不 reset 就会从那个 n_past 接着写、llama_decode 直接 rc=-1。本机自己 spawn
        # 的段是全新的所以看不出这个问题，跨机时才暴露。
        for c in clients:
            c.reset()
        # greedy：R4 要比 token 序列，任何采样随机性都会让两边"不一致"变得不可解释。
        for c in clients:
            c.set_sampling(temperature=0.0, top_p=1.0, top_k=0)

        size = clients[-1].n_embd
        print(f"链已就绪：{len(clients)} 段，hidden={size}，ctx={ctx}，"
              f"spec_k={args.spec_k}")
        print(f"模型：{stages[0].model.name}")

        n_past = 0
        pending: int | None = None
        # （上一轮的文本不再需要：ChatML 的下一个块只取决于 pending 是不是结束符。）
        #: KV 里当前实际有哪些 token，顺序即位置。R4 的「重放」组直接用它。
        kv_ids: list[int] = []

        for i, (question, budget) in enumerate(TURNS):
            pending_is_eog = pending is not None and clients[-1].is_eog(pending)
            delta = _delta_text(i, question, pending_is_eog)
            kv_before = list(kv_ids)
            t0 = time.perf_counter()
            res = _turn(clients, delta, n_past, pending, budget, args.spec_k)
            took = time.perf_counter() - t0

            # 已进 KV 的 = 本轮 prefill 的（含 replay 的 pending）+ 生成里除 pending 之外的
            # + 那些「在 KV 里但没 emit」的（投机块截断，llama.cpp 不允许回退）。
            written = res["ids"] if res["pending"] is None else res["ids"][:-1]
            kv_ids.extend(res["prefill"])
            kv_ids.extend(written)
            kv_ids.extend(res["kv_tail"])

            entry = {
                "round": i + 1,
                "question": question,
                "n_past_start": res["n_past_before"],
                "n_past_end": res["n_past_after"],
                "prefill_tokens": len(res["prefill"]),
                "pending_in": pending,
                "pending_out": res["pending"],
                "generated_tokens": len(res["ids"]),
                "seconds": round(took, 1),
                "prefill_written": res["prefill_written"],
                "kv_written": res["kv_written"],
                "kv_expected": res["kv_expected"],
                "orphans": res["orphans"],
                "kv_tail": res["kv_tail"],
                "text": res["text"],
                "ids": res["ids"],
                # R4 要重放的那一整段：本轮 prefill 之前的 KV + 本轮 prefill。
                "kv_before": kv_before,
                "kv_at_generate": kv_before + res["prefill"],
            }
            report["rounds"].append(entry)

            print(f"\n轮 {i + 1}  n_past {res['n_past_before']} -> {res['n_past_after']}"
                  f"  (prefill {len(res['prefill'])} + gen {len(res['ids'])}, {took:.1f}s)")
            print(f"  问: {question}")
            print(f"  答: {res['text'].strip()!r}")

            n_past = res["n_past_after"]
            pending = res["pending"]

        # ------------------------------------------------------------- R1 / R2 / R3
        checks: dict = {}
        bad_npast = [r["round"] for r in report["rounds"] if not r["prefill_written"]]
        starts = [r["n_past_start"] for r in report["rounds"]]
        ends = [r["n_past_end"] for r in report["rounds"]]
        # 起点必须等于上一轮终点；第 1 轮必须从 0 起。
        chained = all(starts[i] == ends[i - 1] for i in range(1, len(starts)))
        checks["R1_npast_monotonic"] = {
            "pass": starts[0] == 0 and chained and not bad_npast,
            "starts": starts, "ends": ends,
            "not_increasing": [(r["round"], r["n_past_start"], r["n_past_end"])
                               for r in report["rounds"]
                               if r["n_past_end"] <= r["n_past_start"]],
        }
        empty = [r["round"] for r in report["rounds"] if not r["ids"]]
        checks["R2_every_round_has_tokens"] = {"pass": not empty, "empty_rounds": empty}

        misaligned = [{"round": r["round"], "kv_written": r["kv_written"],
                       "text_tokens": r["kv_expected"], "orphans": r["orphans"]}
                      for r in report["rounds"] if r["kv_written"] != r["kv_expected"]]
        checks["R5_kv_matches_transcript"] = {"pass": not misaligned,
                                              "misaligned": misaligned}

        recalled = SECRET in report["rounds"][1]["text"]
        checks["R3_memory"] = {"pass": recalled, "secret": SECRET,
                               "found_in_round2": recalled}

        # ---------------------------------------------------- R4：续用 KV vs 从头重放
        target = report["rounds"][1]          # 用第 2 轮做对照：它跨越了第 1 轮的全部 KV
        for c in clients:
            c.reset()
        # 重放组的输入就是「第 2 轮生成之前的完整 KV」，从位置 0 一次性 prefill。
        stats: dict = {}
        ids_replay: list[int] = []
        text_replay = ""
        for chunk in generate_stream(clients, "", prefix_ids=target["kv_at_generate"],
                                     n_past=0, max_tokens=TURNS[1][1],
                                     spec_k=args.spec_k, stats=stats):
            if chunk.token >= 0:
                ids_replay.append(chunk.token)
            text_replay += chunk.text
        a, b = target["ids"], ids_replay
        first_diff = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
        checks["R4_incremental_equals_replay"] = {
            "pass": first_diff is None and len(a) == len(b),
            "round": 2,
            "incremental_tokens": len(a), "replay_tokens": len(b),
            "first_diff_index": first_diff,
            "incremental_text": target["text"],
            "replay_text": text_replay,
        }

        # --------------------------------------- R3 的反向对照：没有第 1 轮时还知道暗号吗
        for c in clients:
            c.reset()
        control = _turn(clients, _delta_text(1, TURNS[1][0], False), 0, None,
                        TURNS[1][1], args.spec_k)
        checks["R3_control_without_round1"] = {
            # 对照组的正确结果是**不知道**；知道反而说明 R3 的证据是假的。
            "pass": SECRET not in control["text"],
            "text": control["text"],
        }

        report["checks"] = checks
        hard = [k for k in ("R1_npast_monotonic", "R2_every_round_has_tokens",
                            "R5_kv_matches_transcript",
                            "R3_memory", "R3_control_without_round1")
                if not checks[k]["pass"]]
        report["verdict"] = ("PASS" if not hard
                             else "FAIL: " + ", ".join(hard))
        report["r4_verdict"] = ("PASS" if checks["R4_incremental_equals_replay"]["pass"]
                                else "DIVERGED")
        return report
    finally:
        for c in clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001 - keep closing the rest
                pass
        _terminate(procs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", default="", help="覆盖 SA_SHIM（见 config.find_shim）")
    ap.add_argument("--topology", default="topology-local-32-8.toml")
    ap.add_argument("--spec-k", type=int, default=0,
                    help="0 = 纯 decode（默认，隔离 KV 变量）；>0 顺带验投机路径跨轮")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    # 找 DLL 只有一条规则（--shim > SA_SHIM > 本仓库 build），见 config.find_shim。
    args.shim = str(config.find_shim(args.shim))
    os.environ[config.TOPOLOGY_ENV] = args.topology

    report = run(args)
    blob = json.dumps(report, ensure_ascii=False, indent=1)
    print("\n" + "=" * 78)
    for name, res in report["checks"].items():
        print(f"  {'PASS' if res['pass'] else 'FAIL'}  {name}")
    print(f"\nverdict: {report['verdict']}")
    print(f"R4 (增量 KV == 从头重放): {report['r4_verdict']}")
    if args.json_out:
        Path(args.json_out).write_text(blob, encoding="utf-8")
        print(f"详细报告: {args.json_out}")

    if report["verdict"] != "PASS":
        return 7
    return 0 if report["r4_verdict"] == "PASS" else 8


if __name__ == "__main__":
    raise SystemExit(main())
