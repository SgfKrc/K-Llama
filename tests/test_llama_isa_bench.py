"""ISA admission and evidence failures must never imply a faster production build."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('qlh_isa_test_driver', ROOT / 'scripts/llama_isa_bench.py')
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def host():
    return {'schema': 'qlh.isa_host.v1', 'ok': True, 'os_avx_state': True,
            'process_affinity_mask': '255', 'system_affinity_mask': '255',
            'features': {key: True for key in (*bench.FEATURES, 'avx_vnni')}}


@pytest.mark.parametrize('profile', ['avx2', 'avx2_vnni'])
def test_matching_cpu_os_and_explicit_mask_are_admitted(profile):
    bench._host_check(host(), profile, 0x55, 4)


@pytest.mark.parametrize('feature', [*bench.FEATURES, 'avx_vnni'])
@pytest.mark.parametrize('value', [False, 1, None])
def test_missing_or_nonboolean_cpu_feature_rejects_candidate(feature, value):
    data = host()
    data['features'][feature] = value
    with pytest.raises(bench.BenchError, match='CPU/OS'):
        bench._host_check(data, 'avx2_vnni', 0x55, 4)


def test_baseline_remains_usable_when_vnni_is_not_exposed():
    data = host()
    data['features']['avx_vnni'] = False
    bench._host_check(data, 'avx2', 0x55, 4)
    with pytest.raises(bench.BenchError, match='avx2_vnni'):
        bench._host_check(data, 'avx2_vnni', 0x55, 4)


@pytest.mark.parametrize('field,value', [('schema', 'old'), ('ok', 1), ('os_avx_state', False),
                                        ('os_avx_state', 1)])
def test_host_identity_or_missing_os_state_rejects(field, value):
    data = host()
    data[field] = value
    with pytest.raises(bench.BenchError):
        bench._host_check(data, 'avx2', 0x55, 4)


@pytest.mark.parametrize('mask,threads', [(0, 4), (-1, 4), (True, 1), (2**64, 4),
                                         (256, 1), (3, 4)])
def test_invalid_or_undersized_affinity_never_passes(mask, threads):
    with pytest.raises(bench.BenchError, match='affinity'):
        bench._host_check(host(), 'avx2', mask, threads)


def test_system_mask_is_checked_in_addition_to_process_mask():
    data = host()
    data['system_affinity_mask'] = '15'
    with pytest.raises(bench.BenchError, match='affinity'):
        bench._host_check(data, 'avx2', 0x55, 4)


def compile_inventory(tmp_path, *, profile='avx2', flags=None, source_file=None):
    source = tmp_path / 'vendor'
    build = tmp_path / 'build'
    build.mkdir()
    if flags is None:
        flags = ['-msse4.2', '-mf16c', '-mfma', '-mbmi2', '-mavx', '-mavx2']
        if profile == 'avx2_vnni':
            flags.append('-mavxvnni')
    file = source_file or source / 'ggml/src/ggml-cpu/arch/x86/repack.cpp'
    (build / 'compile_commands.json').write_text(json.dumps([
        {'file': str(file), 'arguments': ['g++', '-O3', *flags, '-c', str(file)]}
    ]), encoding='utf-8')
    return build, source


@pytest.mark.parametrize('profile', ['avx2', 'avx2_vnni'])
def test_actual_compile_inventory_matches_explicit_profile(tmp_path, profile):
    build, source = compile_inventory(tmp_path, profile=profile)
    path, commands = bench._compile_flags(build, profile, source)
    assert path.name == 'compile_commands.json' and len(commands) == 1


@pytest.mark.parametrize('flag', ['-march=native', '-march=alderlake', '-mavx512f', '-m32', '-mtune=native'])
def test_unregistered_compiler_isa_flags_are_rejected(tmp_path, flag):
    build, source = compile_inventory(tmp_path, flags=['-mavx2', flag])
    with pytest.raises(bench.BenchError, match='unapproved'):
        bench._compile_flags(build, 'avx2', source)


@pytest.mark.parametrize('profile,flags', [('avx2', ['-mavx2', '-mavxvnni']),
                                         ('avx2_vnni', ['-mavx2']), ('avx2', [])])
def test_wrong_actual_cpu_isa_is_rejected(tmp_path, profile, flags):
    build, source = compile_inventory(tmp_path, flags=flags)
    with pytest.raises(bench.BenchError, match='compiler'):
        bench._compile_flags(build, profile, source)


@pytest.mark.parametrize('outside', [True, False])
def test_source_escape_or_missing_cpu_compile_unit_rejects(tmp_path, outside):
    file = tmp_path / ('outside.cpp' if outside else 'vendor/src/llama.cpp')
    build, source = compile_inventory(tmp_path, source_file=file)
    with pytest.raises(bench.BenchError, match='escapes|no actual CPU'):
        bench._compile_flags(build, 'avx2', source)


def build_pair():
    receipt = {'cmake_definitions': {'GGML_AVX_VNNI': 'OFF', 'GGML_NATIVE': 'OFF'},
               'source_revision': 'a', 'source_tree': 'b', 'lock_sha256': 'c',
               'builder_sha256': 'd', 'header_sha256': 'e', 'shim_source_sha256': 'f',
               'shim_script_sha256': 'g', 'tools': {'g++': {'sha256': 'h'}}, 'patches': ['fixed']}
    candidate = deepcopy(receipt)
    candidate['cmake_definitions']['GGML_AVX_VNNI'] = 'ON'
    return {'receipt': receipt}, {'receipt': candidate}


def test_only_requested_isa_switch_may_differ():
    bench._same_build_policy(*build_pair())


@pytest.mark.parametrize('key', ['source_revision', 'source_tree', 'lock_sha256', 'builder_sha256',
                                'header_sha256', 'shim_source_sha256', 'shim_script_sha256', 'tools', 'patches'])
def test_build_provenance_differences_are_rejected(key):
    base, candidate = build_pair()
    candidate['receipt'][key] = 'different'
    with pytest.raises(bench.BenchError, match='provenance'):
        bench._same_build_policy(base, candidate)


def test_non_isa_build_configuration_difference_is_rejected():
    base, candidate = build_pair()
    candidate['receipt']['cmake_definitions']['GGML_NATIVE'] = 'ON'
    with pytest.raises(bench.BenchError, match='outside'):
        bench._same_build_policy(base, candidate)


def runs():
    result = {'n_ctx': 256, 'n_batch': 16, 'n_ubatch': 16, 'n_seq_max': 1,
              'n_threads': 4, 'n_threads_batch': 4,
              'fixed_argmax': [1, 2, 3], 'greedy_tokens': [4, 5],
              'prefill_seconds': [0.16] * 3, 'decode_seconds': [0.32] * 3}
    first = {'profile': 'avx2', 'result': result,
             'hidden': np.arange(6, dtype='f4').reshape(3, 2), 'repack_buffer_observed': True}
    second = deepcopy(first)
    second['profile'] = 'avx2_vnni'
    second['result']['decode_seconds'] = [0.16] * 3
    return [first, second, deepcopy(second), deepcopy(first)]


def compare(values):
    return bench._compare(values, 16, 32)


def test_counterbalanced_complete_same_precision_comparison_can_pass_local_gate():
    result = compare(runs())
    assert result['numerical']['passed']
    assert result['candidate_over_baseline']['decode'] == 2
    assert result['local_microbenchmark_gain_gate_passed']
    assert result['production_admitted'] is False
    assert result['measured_product_benefit_proven'] is False


@pytest.mark.parametrize('change,key', [('argmax', 'fixed_argmax_equal'), ('greedy', 'greedy_equal'),
                                      ('hidden', 'hidden_close')])
def test_one_candidate_run_divergence_blocks_numerical_and_gain(change, key):
    values = runs()
    if change == 'argmax':
        values[2]['result']['fixed_argmax'][0] = 7
    elif change == 'greedy':
        values[2]['result']['greedy_tokens'][0] = 7
    else:
        for row in values:
            if row['profile'] == 'avx2_vnni':
                row['hidden'][0, 0] += 1
    result = compare(values)
    assert result['numerical'][key] is False
    assert not result['numerical']['passed']
    assert not result['local_microbenchmark_gain_gate_passed']


def test_close_hidden_still_requires_same_profile_bitwise_repeatability():
    values = runs()
    values[2]['hidden'][0, 0] += 0.000001
    result = compare(values)
    assert result['numerical']['hidden_close']
    assert not result['numerical']['same_profile_bitwise_repeatable']
    assert not result['numerical']['passed']


@pytest.mark.parametrize('key', ['n_ctx', 'n_batch', 'n_ubatch', 'n_seq_max', 'n_threads', 'n_threads_batch'])
def test_different_effective_configurations_cannot_be_averaged(key):
    values = runs()
    values[2]['result'][key] += 1
    with pytest.raises(bench.BenchError, match='different effective'):
        compare(values)


@pytest.mark.parametrize('mode', ['missing_process', 'missing_profile', 'too_few_samples'])
def test_partial_timing_or_process_evidence_rejects(mode):
    values = runs()
    if mode == 'missing_process':
        values.pop()
    elif mode == 'missing_profile':
        for row in values:
            row['profile'] = 'avx2'
    else:
        for row in values:
            row['result']['decode_seconds'] = [0.1]
    with pytest.raises(bench.BenchError):
        compare(values)


@pytest.mark.parametrize('mode', ['no_gain', 'prefill_regression', 'unstable', 'missing_repack'])
def test_performance_or_buffer_failures_do_not_admit_candidate(mode):
    values = runs()
    for row in values:
        if row['profile'] == 'avx2_vnni':
            if mode == 'no_gain':
                row['result']['decode_seconds'] = [0.32] * 3
            elif mode == 'prefill_regression':
                row['result']['prefill_seconds'] = [0.2] * 3
            elif mode == 'unstable':
                row['result']['decode_seconds'] = [0.1, 0.2, 0.8]
            else:
                row['repack_buffer_observed'] = False
    result = compare(values)
    assert result['numerical']['passed']
    assert not result['local_microbenchmark_gain_gate_passed']
    assert not result['production_admitted']


def test_baseline_control_never_fabricates_candidate_ratio_or_gain():
    result = bench._compare([row for row in runs() if row['profile'] == 'avx2'], 16, 32, baseline_only=True)
    assert result['numerical']['passed']
    assert result['candidate_over_baseline'] is None
    assert not result['local_microbenchmark_gain_gate_passed']
    assert not result['production_admitted']


def cli(tmp_path):
    return ['--model', 'unused.gguf', '--baseline-build', 'unused-build', '--baseline-only',
            '--affinity-mask', '0x55', '--out-dir', str(tmp_path)]


def test_existing_output_is_preserved_without_source_check_or_native(monkeypatch, tmp_path):
    marker = tmp_path / 'report.json'
    marker.write_text('existing evidence')
    monkeypatch.setattr(bench, 'inspect_sync', lambda **kw: pytest.fail('output protection must precede source inspection'))
    assert bench.main(cli(tmp_path)) == 1
    assert marker.read_text() == 'existing evidence'


@pytest.mark.parametrize('flag,value', [('--ctx', '1'), ('--repeats', '2'), ('--rounds', '1'),
                                      ('--affinity-mask', '0'), ('--affinity-mask', 'text')])
def test_invalid_request_never_starts_source_or_native(monkeypatch, tmp_path, flag, value):
    monkeypatch.setattr(bench, 'inspect_sync', lambda **kw: pytest.fail('invalid input reached source inspection'))
    args = cli(tmp_path / 'fresh')
    if flag == '--affinity-mask':
        args[args.index(flag) + 1] = value
    else:
        args.extend([flag, value])
    assert bench.main(args) == 1
    assert not (tmp_path / 'fresh').exists()


@pytest.mark.parametrize('sync', [{'state': 'ready_to_apply', 'patches': []},
                                  {'state': 'applied', 'patches': [{'state': 'applied'}]}])
def test_uncommitted_source_never_reaches_build_or_model(monkeypatch, tmp_path, sync):
    monkeypatch.setattr(bench, 'inspect_sync', lambda **kw: sync)
    monkeypatch.setattr(bench, '_validated_build', lambda *a: pytest.fail('invalid source reached build'))
    assert bench.main(cli(tmp_path / 'fresh')) == 1
    assert not (tmp_path / 'fresh').exists()


def receipt_before_tools(tmp_path, sync):
    """Reach source-envelope validation with JSON only, before any native input."""
    build = tmp_path / 'receipt-only'
    build.mkdir()
    receipt = {'schema_version': 1, 'exit_code': 0, 'isa_profile': 'avx2',
               'production_default_profile': 'avx2', 'experimental_profile': False,
               'source_git_clean': True, 'source_revision': sync['revision'],
               'source': str(ROOT / 'android/app/src/main/cpp/llama.cpp'),
               'source_root': str(ROOT), 'build_dir': str(build),
               'source_sync': {phase: {'ok': True, **sync} for phase in ('before', 'after')},
               'builder_sha256': 'deliberately-not-a-current-builder-hash'}
    return build, receipt


def test_builder_cli_success_envelope_matches_python_inspection_without_ok(tmp_path):
    sync = {'revision': 'a' * 40, 'state': 'applied', 'dry_run': True,
            'patches': [{'state': 'applied_committed'}]}
    build, receipt = receipt_before_tools(tmp_path, sync)
    (build / 'keep-head-cpu-build.json').write_text(json.dumps(receipt), encoding='utf-8')
    # Passing the full source contract reaches the deliberately wrong builder
    # digest. No cache, tool, DLL or model exists in this fixture.
    with pytest.raises(bench.BenchError, match='different builder'):
        bench._validated_build(build, 'avx2', sync)


@pytest.mark.parametrize('phase,mutation', [('before', 'missing_ok'), ('after', 'nonboolean_ok'),
                                           ('before', 'revision'), ('after', 'pending')])
def test_receipt_source_envelope_and_contract_drift_reject_before_tools(tmp_path, phase, mutation):
    sync = {'revision': 'a' * 40, 'state': 'applied', 'dry_run': True,
            'patches': [{'state': 'applied_committed'}]}
    build, receipt = receipt_before_tools(tmp_path, sync)
    captured = receipt['source_sync'][phase]
    if mutation == 'missing_ok':
        del captured['ok']
    elif mutation == 'nonboolean_ok':
        captured['ok'] = 1
    elif mutation == 'revision':
        captured['revision'] = 'b' * 40
    else:
        captured['patches'] = [{'state': 'ready_to_apply'}]
    (build / 'keep-head-cpu-build.json').write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(bench.BenchError, match='source receipt'):
        bench._validated_build(build, 'avx2', sync)


@pytest.mark.parametrize('field,value', [('isa_profile', 'avx2_vnni'),
                                        ('experimental_profile', 0),
                                        ('source_git_clean', 1)])
def test_receipt_profile_or_nonboolean_clean_claim_rejects_before_tools(tmp_path, field, value):
    sync = {'revision': 'a' * 40}
    build, receipt = receipt_before_tools(tmp_path, sync)
    receipt[field] = value
    (build / 'keep-head-cpu-build.json').write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(bench.BenchError, match='committed ISA profile'):
        bench._validated_build(build, 'avx2', sync)
