"""#135：段复用判据必须依据本次清点，而不是 capabilities 快照。

背景：master 有一处优化 —— 策略已判成 `local_cut` 后，若发现 worker 手上已有身份
完全一致的段，就改判 `distribute` 让接收端直接复用，避免重裁。该判断原只依据 worker
上报的 **capabilities 快照**，而快照只在 hello / 能力变更时刷新。

于是运维清掉段缓存后（磁盘满、迁移、手工回收 —— 都是正常操作），worker 在下一次重扫
前仍持有旧清单 ⇒ master 误判"可复用" ⇒ 改判 `distribute` ⇒ 接收端找不到文件 ⇒ 退化成
真下载，比直接 `local_cut` 更差（2026-10-27 实测卡在 `bytes_received=11534336/1983372689`
超过 7 分钟；重启 worker 让它重扫后同一轮回落到 `local_cut`，58.7 s 完成）。

修复：worker 的 `inventory()` 现在返回 `artifacts`（只有段文件真实存在且通过
`_verify_files` 的尺寸 + 摘要校验时才列出），master 的 `reuse_candidates()` 优先采信它，
且**即使为空也不回退快照**。
"""
import pytest

from model_supply import reuse_artifact_matches, reuse_candidates


def artifact(**overrides):
    value = {
        "layer_range": [28, 32],
        "artifact_sha256": "a" * 64,
        "source_model_id": "qwen3-5-9b",
        "source_model_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
    }
    value.update(overrides)
    return value


def test_empty_inventory_overrides_stale_capabilities():
    """段文件被删后，清点报空必须压过快照里残留的旧清单。

    这是本修复的核心：若此处回退到 `capabilities`，就会重现"删缓存后误判复用"。
    """
    stale_capabilities = {"layer_artifacts": [artifact()]}

    candidates, evidence = reuse_candidates({"artifacts": []}, stale_capabilities)

    assert candidates == [], "空清点不得回退到快照，否则误判复用"
    assert evidence == "inventory"


def test_present_inventory_is_authoritative():
    fresh = [artifact()]
    candidates, evidence = reuse_candidates(
        {"artifacts": fresh}, {"layer_artifacts": []}
    )
    assert candidates == fresh
    assert evidence == "inventory"


def test_inventory_without_artifacts_falls_back_to_capabilities():
    """旧 worker 不返回 `artifacts` 键时保持向后兼容。"""
    reported = {"sources": [], "local_cut": True}
    capabilities = {"layer_artifacts": [artifact()]}

    candidates, evidence = reuse_candidates(reported, capabilities)

    assert candidates == [artifact()]
    assert evidence == "capabilities"


@pytest.mark.parametrize("inventory, capabilities", [(None, None), ({}, None), (None, {})])
def test_missing_both_sides_yields_no_candidates(inventory, capabilities):
    candidates, evidence = reuse_candidates(inventory, capabilities)
    assert candidates == []
    assert evidence == "capabilities"


def test_artifacts_key_of_wrong_type_is_not_trusted():
    """非 list 的 `artifacts` 视为缺失，退回快照 —— 避免异常结构让判据静默失效。"""
    candidates, evidence = reuse_candidates(
        {"artifacts": "not-a-list"}, {"layer_artifacts": [artifact()]}
    )
    assert candidates == [artifact()]
    assert evidence == "capabilities"


def test_identity_match_accepts_exact_artifact():
    assert reuse_artifact_matches(artifact(), artifact())


@pytest.mark.parametrize("field, value", [
    ("layer_range", [0, 4]),
    ("artifact_sha256", "d" * 64),
    ("source_model_id", "some-other-model"),
    ("source_model_sha256", "e" * 64),
    ("tokenizer_sha256", "f" * 64),
])
def test_identity_match_rejects_any_field_mismatch(field, value):
    manifest = artifact()
    assert not reuse_artifact_matches(artifact(**{field: value}), manifest)


@pytest.mark.parametrize("candidate", [None, [], "artifact", 7])
def test_identity_match_rejects_non_mapping(candidate):
    assert not reuse_artifact_matches(candidate, artifact())


def test_identity_match_treats_absent_optional_fields_as_equal():
    """老 manifest 可能没有 `source_model_id` / `tokenizer_sha256`，两侧都为 None 视为相同。"""
    manifest = {"layer_range": [0, 4], "artifact_sha256": "a" * 64,
                "source_model_sha256": "b" * 64}
    candidate = {"layer_range": [0, 4], "artifact_sha256": "a" * 64,
                 "source_model_sha256": "b" * 64}
    assert reuse_artifact_matches(candidate, manifest)
