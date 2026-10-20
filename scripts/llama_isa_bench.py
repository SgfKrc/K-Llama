#!/usr/bin/env python3
"""Compare explicit CPU ISA builds against the current AVX2/repack-ON baseline.

This is a local native development gate. It never changes packaging defaults.
Fresh builder receipts, effective compile flags, CPU/OS support, loaded DLL paths,
fixed inputs, affinity, numerical parity and raw samples are retained separately.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from edge_bench import BenchError, _json, _protected_model, _sha256
from gguf_metadata_contract import read_gguf_header, normalize_gguf_metadata
from llama_kernel_bench import PROMPT, FIXED, _cache, _check_native, _command, _tokens, _write
from scripts.model_tools.sync_llama_cpp import inspect_sync

SCHEMA = 'qlh.llama_isa_bench.v1'
DLLS = ('qlh_keep_head.dll', 'libllama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll',
        'libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll', 'libgomp-1.dll')
PROFILES = {'avx2': False, 'avx2_vnni': True}
FEATURES = ('sse42', 'avx', 'avx2', 'bmi2', 'fma', 'f16c')
ATOL = RTOL = 1e-4
MIN_GAIN = 1.03
MIN_PREFILL = 0.97
MAX_CV = 0.10


def _host_check(host, profile, mask, threads):
    if host.get('schema') != 'qlh.isa_host.v1' or host.get('ok') is not True:
        raise BenchError('missing successful CPU/OS feature probe')
    required = FEATURES + (('avx_vnni',) if PROFILES[profile] else ())
    if host.get('os_avx_state') is not True or any(host.get('features', {}).get(k) is not True for k in required):
        raise BenchError(f'CPU/OS lacks required ISA features for {profile}')
    available = int(host['process_affinity_mask'])
    system = int(host['system_affinity_mask'])
    if type(mask) is not int or mask <= 0 or mask >= 2**64 or mask & available != mask or mask & system != mask:
        raise BenchError('affinity mask is outside the available Windows processor group')
    if mask.bit_count() < threads:
        raise BenchError('affinity mask has fewer logical CPUs than requested threads')


def _compile_flags(build, profile, source):
    path = build / 'compile_commands.json'
    if path.stat().st_size > 16 * 1024 * 1024:
        raise BenchError('compile command inventory is too large')
    entries = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(entries, list) or not entries or len(entries) > 10000:
        raise BenchError('invalid compile command inventory')
    cpu_commands = []
    allowed = {'-mavx', '-mavx2', '-msse4.2', '-mbmi2', '-mf16c', '-mfma'}
    if PROFILES[profile]:
        allowed.add('-mavxvnni')
    for entry in entries:
        file = Path(entry['file']).resolve()
        if not file.is_relative_to(source.resolve()):
            raise BenchError('compile command source escapes pinned vendor')
        tokens = entry.get('arguments') or shlex.split(entry['command'])
        isa = [token for token in tokens if token.startswith('-m')]
        if any(token not in allowed for token in isa):
            raise BenchError(f'unapproved actual compiler ISA flag: {isa}')
        if file.is_relative_to((source / 'ggml/src/ggml-cpu').resolve()):
            if '-mavx2' not in isa or ('-mavxvnni' in isa) != PROFILES[profile]:
                raise BenchError('actual CPU compiler flags differ from the selected ISA profile')
            cpu_commands.append({'file': str(file), 'isa_flags': isa})
    if not cpu_commands:
        raise BenchError('no actual CPU compile commands were found')
    return path, cpu_commands


def _validated_build(build, profile, sync):
    build = build.resolve(strict=True)
    source = ROOT / 'android/app/src/main/cpp/llama.cpp'
    receipt_path = build / 'keep-head-cpu-build.json'
    receipt = _json(receipt_path)
    if (receipt.get('schema_version') != 1 or receipt.get('exit_code') != 0
            or receipt.get('isa_profile') != profile or receipt.get('production_default_profile') != 'avx2'
            or receipt.get('experimental_profile') is not PROFILES[profile]
            or receipt.get('source_git_clean') is not True
            or receipt.get('source_revision') != sync['revision']
            or Path(receipt.get('source', '')).resolve() != source.resolve()
            or Path(receipt.get('source_root', '')).resolve() != ROOT
            or Path(receipt.get('build_dir', '')).resolve() != build):
        raise BenchError('build receipt does not bind the selected committed ISA profile')
    for phase in ('before', 'after'):
        captured = receipt.get('source_sync', {}).get(phase, {})
        if captured.get('ok') is not True or {k: v for k, v in captured.items() if k != 'ok'} != {k: v for k, v in sync.items() if k != 'ok'}:
            raise BenchError('build source receipt differs from the current shared source contract')
    if receipt.get('builder_sha256') != _sha256(ROOT / 'scripts/model_tools/build_keep_head_cpu.ps1'):
        raise BenchError('build receipt names a different builder')
    if receipt.get('lock_sha256') != _sha256(ROOT / 'scripts/model_tools/llama_quantize.lock.json'):
        raise BenchError('build receipt names a different lock')
    cache_path = build / 'CMakeCache.txt'
    cache = _cache(cache_path)
    definitions = receipt.get('cmake_definitions')
    if not isinstance(definitions, dict) or not definitions:
        raise BenchError('build receipt lacks explicit CMake definitions')
    if any(cache.get(k, '').replace('\\', '/') != str(v).replace('\\', '/') for k, v in definitions.items()):
        raise BenchError('actual CMake cache differs from its build receipt')
    required = {'GGML_NATIVE': 'OFF', 'GGML_CPU': 'ON', 'GGML_AVX': 'ON', 'GGML_AVX2': 'ON',
                'GGML_AVX_VNNI': 'ON' if PROFILES[profile] else 'OFF', 'GGML_CPU_REPACK': 'ON',
                'GGML_AVX512': 'OFF', 'GGML_CPU_ALL_VARIANTS': 'OFF', 'GGML_BACKEND_DL': 'OFF',
                'GGML_CUDA': 'OFF', 'GGML_VULKAN': 'OFF', 'CMAKE_EXPORT_COMPILE_COMMANDS': 'ON',
                'CMAKE_C_FLAGS': '', 'CMAKE_CXX_FLAGS': '', 'CMAKE_BUILD_TYPE': 'Release',
                'CMAKE_C_FLAGS_RELEASE': '-O3 -DNDEBUG', 'CMAKE_CXX_FLAGS_RELEASE': '-O3 -DNDEBUG'}
    if any(definitions.get(k) != v for k, v in required.items()):
        raise BenchError('build definitions differ from the fixed CPU/repack policy')
    if (cache.get('CMAKE_GENERATOR') != 'MinGW Makefiles'
            or Path(cache.get('CMAKE_HOME_DIRECTORY', '')).resolve() != source.resolve()):
        raise BenchError('actual CMake generator/source differs')
    if receipt.get('cache_sha256') != _sha256(cache_path):
        raise BenchError('CMake cache bytes changed after the build')
    inputs = receipt.get('input_sha256')
    if not isinstance(inputs, dict) or not inputs:
        raise BenchError('missing build input bindings')
    paths = [receipt_path, cache_path, build / 'src/libllama.dll.a']
    for name, digest in inputs.items():
        path = Path(name)
        if not path.is_file() or _sha256(path) != digest:
            raise BenchError('build input bytes changed after the build: ' + str(path))
        paths.append(path)
    tools = receipt.get('tools', {})
    for key in ('cmake', 'gcc', 'g++', 'make'):
        tool = tools.get(key, {})
        path = Path(tool.get('path', ''))
        if not path.is_file() or _sha256(path) != tool.get('sha256'):
            raise BenchError('build tool identity changed: ' + key)
        paths.append(path)
    out = Path(receipt['out_dir']).resolve(strict=True)
    artifacts = receipt.get('artifacts', [])
    if len(artifacts) != len(DLLS) or {item['name'] for item in artifacts} != set(DLLS):
        raise BenchError('build receipt lacks the nine-DLL closure')
    for artifact in artifacts:
        path = out / artifact['name']
        if Path(artifact['path']).resolve() != path or path.stat().st_size != artifact['size_bytes'] or _sha256(path) != artifact['sha256']:
            raise BenchError('native artifact changed after the build: ' + artifact['name'])
        paths.append(path)
    spec = importlib.util.spec_from_file_location('qlh_isa_pe', ROOT / 'packaging/packaging/keep_head_resources.py')
    pe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pe)
    pe.NATIVE_RELATIVE = out
    pe.collect_keep_head_binaries(ROOT)
    compile_path, commands = _compile_flags(build, profile, source)
    paths.append(compile_path)
    return {'profile': profile, 'build_root': str(build), 'out_dir': str(out),
            'receipt': receipt, 'cpu_compile_commands': commands, 'paths': paths}


def _same_build_policy(baseline, candidate):
    a, b = baseline['receipt'], candidate['receipt']
    skip = {'GGML_AVX_VNNI'}
    if ({k: v for k, v in a['cmake_definitions'].items() if k not in skip}
            != {k: v for k, v in b['cmake_definitions'].items() if k not in skip}):
        raise BenchError('baseline and candidate differ outside the requested ISA switch')
    for key in ('source_revision', 'source_tree', 'lock_sha256', 'builder_sha256',
                'header_sha256', 'shim_source_sha256', 'shim_script_sha256', 'tools', 'patches'):
        if a.get(key) != b.get(key):
            raise BenchError('baseline and candidate build provenance differs: ' + key)


def _compare(runs, prompt_count, fixed_count, *, baseline_only=False):
    import numpy as np
    profiles = ('avx2',) if baseline_only else tuple(PROFILES)
    minimum = 2 if baseline_only else 4
    if len(runs) < minimum or {r['profile'] for r in runs} != set(profiles):
        raise BenchError('complete counterbalanced baseline/candidate runs are required')
    reference = runs[0]
    config = ('n_ctx', 'n_batch', 'n_ubatch', 'n_seq_max', 'n_threads', 'n_threads_batch')
    if any(any(r['result'][k] != reference['result'][k] for k in config) for r in runs):
        raise BenchError('different effective native configurations cannot be compared')
    numerical = {'baseline': 'avx2_repack_on', 'atol': ATOL, 'rtol': RTOL,
                 'fixed_argmax_equal': True, 'greedy_equal': True, 'hidden_close': True,
                 'same_profile_bitwise_repeatable': True, 'hidden_max_absolute_difference': 0.0}
    for run in runs:
        numerical['fixed_argmax_equal'] &= run['result']['fixed_argmax'] == reference['result']['fixed_argmax']
        numerical['greedy_equal'] &= run['result']['greedy_tokens'] == reference['result']['greedy_tokens']
        numerical['hidden_close'] &= bool(np.allclose(run['hidden'], reference['hidden'], atol=ATOL, rtol=RTOL))
        numerical['hidden_max_absolute_difference'] = max(numerical['hidden_max_absolute_difference'],
            float(np.max(np.abs(run['hidden'].astype('f8') - reference['hidden'].astype('f8')))))
    performance = {}
    for profile in profiles:
        selected = [r for r in runs if r['profile'] == profile]
        if len(selected) < 2:
            raise BenchError('at least two independent runs per profile are required')
        first = selected[0]
        numerical['same_profile_bitwise_repeatable'] &= all(np.array_equal(r['hidden'], first['hidden'])
            and r['result']['fixed_argmax'] == first['result']['fixed_argmax']
            and r['result']['greedy_tokens'] == first['result']['greedy_tokens'] for r in selected)
        row = {}
        for phase, count in (('prefill', prompt_count), ('decode', fixed_count)):
            samples = [count / t for r in selected for t in r['result'][phase + '_seconds']]
            if len(samples) < 6 or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise BenchError('at least six finite timing samples per phase/profile are required')
            row[phase] = {'tokens_per_second': samples, 'median': statistics.median(samples),
                          'cv': statistics.pstdev(samples) / statistics.mean(samples),
                          'min': min(samples), 'max': max(samples), 'count': len(samples)}
        performance[profile] = row
    numerical['passed'] = all(numerical[k] for k in ('fixed_argmax_equal', 'greedy_equal', 'hidden_close', 'same_profile_bitwise_repeatable'))
    ratios = None if baseline_only else {phase: performance['avx2_vnni'][phase]['median'] / performance['avx2'][phase]['median']
              for phase in ('prefill', 'decode')}
    stable = all(row[phase]['cv'] <= MAX_CV for row in performance.values() for phase in ('prefill', 'decode'))
    buffer_gate = all(r['repack_buffer_observed'] for r in runs)
    eligible = numerical['passed'] and buffer_gate
    return {'numerical': numerical, 'repack_buffer_gate_passed': buffer_gate,
            'performance': performance, 'candidate_over_baseline': ratios,
            'predeclared_gain_gate': {'min_decode_ratio': MIN_GAIN, 'min_prefill_ratio': MIN_PREFILL, 'max_cv': MAX_CV,
                                     'stable': stable, 'eligible': eligible},
            'local_microbenchmark_gain_gate_passed': bool(not baseline_only and eligible and stable and ratios['decode'] >= MIN_GAIN and ratios['prefill'] >= MIN_PREFILL),
            'production_admitted': False, 'measured_product_benefit_proven': False}


def _model(path):
    raw, tensors = read_gguf_header(path)
    meta = normalize_gguf_metadata(raw)
    if any(k.startswith('split.') for k in raw) or not tensors or not any(t[3] == 12 for t in tensors.values()):
        raise BenchError('ISA study requires a single GGUF with actual Q4_K tensors')
    if meta['architecture'] not in {'qwen2', 'qwen35'} or meta['nextn_predict_layers'] not in (None, 0):
        raise BenchError('ISA development matrix currently covers qwen2 and qwen35 without MTP')
    tensor = tensors.get('output.weight', tensors.get('token_embd.weight'))
    if tensor is None or len(tensor[2]) != 2:
        raise BenchError('missing actual vocabulary tensor dimensions')
    dimensions = {'n_embd': meta['n_embd'], 'n_layer': meta['block_count'], 'n_vocab': tensor[2][1]}
    if any(type(n) is not int or n <= 0 for n in dimensions.values()):
        raise BenchError('invalid model dimensions')
    return meta, dimensions


def run(args, report):
    if os.name != 'nt' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        raise BenchError('ISA native study currently requires Windows x64')
    if not (1 <= args.threads <= 64 and 2 <= args.rounds <= 10 and 3 <= args.repeats <= 20
            and 1 <= args.greedy_steps <= 4096 and 1 <= args.timeout <= 1800):
        raise BenchError('threads/rounds/repeats/greedy/timeout outside supported bounds')
    prompt, fixed = _tokens(args.prompt_tokens), _tokens(args.fixed_tokens)
    if not len(prompt) + max(len(fixed), args.greedy_steps) <= args.ctx <= 32768:
        raise BenchError('context cannot hold the complete fixed and greedy sequences')
    try:
        mask = int(args.affinity_mask, 0)
    except ValueError as exc:
        raise BenchError('affinity mask must be an explicit integer') from exc
    if not 0 < mask < 2**64 or mask.bit_count() < args.threads:
        raise BenchError('invalid or undersized affinity mask')
    if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
        raise BenchError('choose a fresh empty output directory')
    sync = inspect_sync(android_root=ROOT / 'android', dry_run=True)
    if sync['state'] != 'applied' or any(p['state'] != 'applied_committed' for p in sync['patches']):
        raise BenchError('ISA study requires committed applied source')
    builds = [_validated_build(args.baseline_build, 'avx2', sync)]
    if not args.baseline_only:
        builds.append(_validated_build(args.candidate_build, 'avx2_vnni', sync))
        _same_build_policy(*builds)
    model = args.model.resolve(strict=True)
    meta, dims = _model(model)
    if any(n >= dims['n_vocab'] for n in prompt + fixed):
        raise BenchError('token ID exceeds actual vocabulary')
    paths = [model, Path(__file__).resolve(), ROOT / 'scripts/llama_kernel_bench.py',
             ROOT / 'scripts/edge_bench.py', ROOT / 'scripts/model_tools/llama_kernel_probe.cpp',
             ROOT / 'scripts/model_tools/llama_isa_probe.cpp', ROOT / 'scripts/model_tools/llama_isa_host.cpp',
             ROOT / 'scripts/model_tools/llama_isa_host.h', ROOT / 'packaging/packaging/keep_head_resources.py']
    paths += [path for build in builds for path in build.pop('paths')]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report['created_output'] = True
    report['mode'] = 'baseline_control' if args.baseline_only else 'isa_comparison'
    report['source_sync'] = sync
    report['builds'] = builds
    report['model'] = {'path': str(model), 'metadata': meta, 'dimensions': dims}
    report['invocation'] = {'threads': args.threads, 'ctx': args.ctx, 'prompt': prompt, 'fixed': fixed,
                            'rounds': args.rounds, 'repeats': args.repeats, 'greedy_steps': args.greedy_steps,
                            'affinity_mask': str(mask), 'order': 'repeated baseline control; repack ON' if args.baseline_only else 'AB/BA alternating rounds; repack ON throughout'}
    report['declared_gate'] = {'atol': ATOL, 'rtol': RTOL, 'min_decode_ratio': MIN_GAIN,
                               'min_prefill_ratio': MIN_PREFILL, 'max_cv': MAX_CV}
    with ExitStack() as protected:
        for path in dict.fromkeys(paths):
            protected.enter_context(_protected_model(path))
        bindings = {str(path.resolve()): _sha256(path) for path in dict.fromkeys(paths)}
        report['input_sha256'] = bindings
        # Recheck after protecting the bytes; preflight/locking drift cannot pass.
        fresh = [_validated_build(args.baseline_build, 'avx2', sync)]
        if not args.baseline_only:
            fresh.append(_validated_build(args.candidate_build, 'avx2_vnni', sync))
            _same_build_policy(*fresh)
        compiler = builds[0]['receipt']['tools']['g++']['path']
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
        env['PATH'] = str(Path(compiler).parent) + os.pathsep + env.get('PATH', '')
        host_probe = args.out_dir.resolve() / 'isa-host.exe'
        common = [compiler, '-std=c++17', '-O2', '-march=x86-64', '-mtune=generic']
        _command(common + [str(ROOT / 'scripts/model_tools/llama_isa_host.cpp'), '-o', str(host_probe)],
                 args.out_dir, 'compile-host', environment=env, timeout=args.timeout)
        host = json.loads(_command([str(host_probe)], args.out_dir, 'host', environment=env, timeout=args.timeout).stdout)
        report['host'] = host
        for build in builds:
            _host_check(host, build['profile'], mask, args.threads)
        report['host'] = host
        probes = {}
        source = ROOT / 'android/app/src/main/cpp/llama.cpp'
        for build in builds:
            profile = build['profile']
            probe = args.out_dir.resolve() / ('isa-' + profile + '.exe')
            _command(common + [str(ROOT / 'scripts/model_tools/llama_isa_probe.cpp'),
                '-I' + str(source / 'include'), '-I' + str(source / 'ggml/include'),
                '-L' + str(Path(build['build_root']) / 'src'), '-lllama', '-o', str(probe)],
                args.out_dir, 'compile-' + profile, environment=env, timeout=args.timeout)
            probes[profile] = probe
            protected.enter_context(_protected_model(probe))
        report['probe_sha256'] = {k: _sha256(v) for k, v in probes.items()}
        runs = []
        report['runs'] = []
        for round_index in range(args.rounds):
            for index in ((0,) if args.baseline_only else (0, 1) if round_index % 2 == 0 else (1, 0)):
                build = builds[index]
                profile = build['profile']
                name = f'round-{round_index + 1}-{profile}'
                hidden = args.out_dir.resolve() / (name + '.hidden.f32')
                env['PATH'] = build['out_dir'] + os.pathsep + str(Path(compiler).parent) + os.pathsep + os.environ.get('PATH', '')
                argv = [str(probes[profile]), str(model), '1', str(args.threads), str(args.ctx),
                        args.prompt_tokens, args.fixed_tokens, str(args.greedy_steps), str(args.repeats), str(hidden), str(mask)]
                print('[run] ' + name, flush=True)
                native = _command(argv, args.out_dir, name, environment=env, timeout=args.timeout)
                data = json.loads(native.stdout)
                values = _check_native(data, policy=1, rows=len(prompt) + len(fixed), greedy=args.greedy_steps,
                    repeats=args.repeats, dimensions=dims, hidden=hidden, request=report['invocation'])
                _host_check(data.get('isa_host', {}), profile, mask, args.threads)
                if int(data['isa_host']['process_affinity_mask']) != mask:
                    raise BenchError('native affinity readback differs')
                loaded = data.get('loaded_modules', {})
                if set(loaded) != set(DLLS[1:]) or any(Path(path).resolve() != Path(build['out_dir']) / key for key, path in loaded.items()):
                    raise BenchError('actual loaded DLLs differ from the selected isolated closure')
                observed = any('CPU_REPACK' in line and 'model buffer size' in line for line in native.stderr.splitlines())
                row = {'profile': profile, 'name': name, 'result': data,
                       'hidden_sha256': _sha256(hidden), 'repack_buffer_observed': observed}
                report['runs'].append(row)
                runs.append({**row, 'hidden': values})
                _write(args.out_dir / 'report.json', report)
        report['comparison'] = _compare(runs, len(prompt), len(fixed), baseline_only=args.baseline_only)
        if any(_sha256(Path(path)) != digest for path, digest in bindings.items()):
            raise BenchError('a bound experiment input changed')
        if inspect_sync(android_root=ROOT / 'android', dry_run=True) != sync:
            raise BenchError('source provenance changed during the study')
        passed = report['comparison']['numerical']['passed'] and report['comparison']['repack_buffer_gate_passed']
        report.update(ok=passed, status='numerical_passed' if passed else 'numerical_rejected', native_executed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--baseline-build', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--candidate-build', type=Path)
    mode.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--affinity-mask', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--ctx', type=int, default=256)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--greedy-steps', type=int, default=32)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--prompt-tokens', default=PROMPT)
    parser.add_argument('--fixed-tokens', default=FIXED)
    args = parser.parse_args(argv)
    report = {'schema': SCHEMA, 'ok': False, 'status': 'failed', 'native_executed': False,
              'production_admitted': False, 'measured_product_benefit_proven': False,
              'started_at': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform()}
    try:
        run(args, report)
    except (BenchError, OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        report['error'] = str(exc)
    if report.get('created_output'):
        _write(args.out_dir / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'runs', 'input_sha256', 'builds'}}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
