"""Static import/write gate for the refactor and retained pure legacy contracts.

This checks source syntax without importing production modules. It deliberately
checks direct writes and literal setattr/delattr/container mutation, not arbitrary
runtime aliasing. Owner behavior and import side effects also need runtime tests.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_RESOURCE_CALLS = frozenset({
    "open", "read_text", "read_bytes", "write_text", "write_bytes", "mkdir",
    "makedirs", "unlink", "rmdir", "rename", "connect",
    "socket", "bind", "listen", "accept", "send", "sendall", "recv", "urlopen",
    "Popen", "run", "start", "FileHandler", "basicConfig", "from_pretrained",
    "load_model", "load_state_dict", "CDLL", "WinDLL",
})
_QUALIFIED_RESOURCE_CALLS = frozenset({"os.remove", "os.replace", "shutil.move", "shutil.rmtree", "shutil.copy", "shutil.copy2", "pathlib.Path.replace", "pathlib.PosixPath.replace", "pathlib.WindowsPath.replace"})
_MUTATORS = frozenset({"clear", "update", "setdefault", "pop", "popitem", "append",
                       "extend", "insert", "remove", "add", "discard", "__setitem__",
                       "__delitem__"})
_DYNAMIC_CALLS = frozenset({"__import__", "import_module", "exec", "eval"})


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    code: str
    detail: str


def load_policy(path: Path = ROOT / "refactor-boundaries.json") -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8-sig"))
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported boundary policy version")
    return policy


def _same_owner(module: str, owner: str) -> bool:
    """Treat the design namespace and installed package as the same domain."""
    def canonical(value):
        return "qlh." + value[len("qlh_core."):] if value.startswith("qlh_core.") else value
    return canonical(module) == canonical(owner)


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _resolved_call(node: ast.Call, aliases: dict[str, str]) -> str:
    name = _name(node.func)
    root, separator, rest = name.partition(".")
    return aliases.get(root, root) + (separator + rest if separator else "")


def _imports(tree: ast.Module, module: str):
    package = module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                prefix = ".".join([*base, *([node.module] if node.module else [])])
            else:
                prefix = node.module or ""
            # Include imported names: `from qlh import adapters` must not bypass
            # the layer check even though the ImportFrom module is just `qlh`.
            for alias in node.names:
                yield node, f"{prefix}.{alias.name}" if prefix else alias.name


def _import_time_calls(node: ast.AST, *, annotations_eager: bool = True):
    if isinstance(node, ast.Module):
        annotations_eager = not any(
            isinstance(item, ast.ImportFrom) and item.module == "__future__"
            and any(alias.name == "annotations" for alias in item.names)
            for item in node.body
        )
    if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and _name(node.test.left) == "__name__" and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq) and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"):
        for statement in node.orelse:
            yield from _import_time_calls(statement, annotations_eager=annotations_eager)
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        expressions = [*getattr(node, "decorator_list", []), *node.args.defaults,
                       *(x for x in node.args.kw_defaults if x is not None)]
        if annotations_eager:
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            arguments += [x for x in (node.args.vararg, node.args.kwarg) if x is not None]
            expressions += [arg.annotation for arg in arguments if arg.annotation is not None]
            if getattr(node, "returns", None) is not None:
                expressions.append(node.returns)
        for expression in expressions:
            yield from _import_time_calls(expression, annotations_eager=annotations_eager)
        return
    if isinstance(node, ast.AnnAssign) and not annotations_eager:
        for expression in [node.target, *([node.value] if node.value is not None else [])]:
            yield from _import_time_calls(expression, annotations_eager=annotations_eager)
        return
    if isinstance(node, ast.Call):
        yield node
        if isinstance(node.func, ast.Lambda):
            yield from _import_time_calls(node.func.body, annotations_eager=annotations_eager)
    for child in ast.iter_child_nodes(node):
        yield from _import_time_calls(child, annotations_eager=annotations_eager)


def _written_fields(node: ast.AST):
    if isinstance(node, ast.Attribute):
        yield node.attr
        yield from _written_fields(node.value)
    elif isinstance(node, ast.Subscript):
        # Covers ordinary map writes and obj.__dict__["model_loaded"] = ... .
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            yield node.slice.value
        yield from _written_fields(node.value)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            yield from _written_fields(item)


def check_source(source: str, path: str, module: str, policy: dict[str, Any],
                 *, legacy_pure: bool = False) -> list[Violation]:
    result: list[Violation] = []

    def report(node: ast.AST, code: str, detail: str) -> None:
        result.append(Violation(path, getattr(node, "lineno", 1), code, detail))

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [Violation(path, exc.lineno or 1, "syntax_error", str(exc))]
    aliases = _import_aliases(tree)
    parts = module.split(".")
    layer = parts[1] if len(parts) > 1 and parts[1] != "__init__" else "contracts"
    rule = (dict(stdlib_only=True, pure=True, internal=[]) if legacy_pure
            else policy["layers"].get(layer))
    if rule is None:
        report(tree, "unclassified_layer", module)
        return result
    for node, dependency in _imports(tree, module):
        root = dependency.split(".")[0]
        if root == "src" and not legacy_pure:
            report(node, "noncanonical_import", dependency)
        if root in policy.get("package_roots", ["qlh"]) and not legacy_pure:
            target = dependency.split(".")[1] if "." in dependency else "contracts"
            if target not in {layer, *rule["internal"]}:
                report(node, "layer_import", f"{layer} -> {dependency}")
        elif root not in sys.stdlib_module_names and root != "__future__":
            pure_delegate = legacy_pure and any(
                dependency == allowed or dependency.startswith(allowed + ".")
                for allowed in policy.get("legacy_pure_imports", {}).get(module, [])
            )
            if rule["stdlib_only"] and not pure_delegate:
                report(node, "non_stdlib_import", dependency)
        if not legacy_pure and layer != "bootstrap":
            if root in policy["forbidden_core_imports"]:
                report(node, "retired_or_facade_import", dependency)
            if layer == "api" and root in policy["forbidden_api_imports"]:
                report(node, "api_engine_import", dependency)
    owners = policy["write_owners"] if not legacy_pure else {}
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            # Store/Del also cover for/with/comprehension targets.
            targets = [node]
        elif isinstance(node, ast.Call):
            call = _resolved_call(node, aliases).split(".")[-1]
            if not legacy_pure and layer not in {"adapters", "bootstrap"} and call in _DYNAMIC_CALLS:
                report(node, "dynamic_code", _name(node.func))
            if call in {"setattr", "delattr"} and len(node.args) >= 2:
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    owner = owners.get(key.value)
                    if owner and not _same_owner(module, owner):
                        report(node, "foreign_write", f"{key.value} belongs to {owner}")
                elif not legacy_pure and layer not in {"adapters", "bootstrap"}:
                    report(node, "dynamic_write", "computed setattr/delattr needs an explicit adapter")
            if isinstance(node.func, ast.Attribute) and call in _MUTATORS:
                targets = [node.func.value]
                if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "__dict__":
                    fields = set(keyword.arg for keyword in node.keywords if keyword.arg)
                    for argument in node.args:
                        if isinstance(argument, ast.Dict):
                            fields.update(key.value for key in argument.keys
                                          if isinstance(key, ast.Constant) and isinstance(key.value, str))
                        elif isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                            fields.add(argument.value)
                    for field in fields:
                        owner = owners.get(field)
                        if owner and not _same_owner(module, owner):
                            report(node, "foreign_write", f"{field} belongs to {owner}")
        for target in targets:
            for field in set(_written_fields(target)):
                owner = owners.get(field)
                if owner and not _same_owner(module, owner):
                    report(node, "foreign_write", f"{field} belongs to {owner}")
    calls = (n for n in ast.walk(tree) if isinstance(n, ast.Call)) if rule["pure"] else _import_time_calls(tree)
    for node in calls:
        name = _resolved_call(node, aliases)
        if name in policy.get("pure_call_exceptions", {}).get(module, []):
            continue
        if name.split(".")[-1] in _RESOURCE_CALLS or name in _QUALIFIED_RESOURCE_CALLS:
            report(node, "pure_io" if rule["pure"] else "import_resource", _name(node.func))
    return result


def check_repository(root: Path, policy: dict[str, Any]) -> tuple[list[Violation], list[str]]:
    violations: list[Violation] = []
    checked: list[str] = []
    required = [root / name for name in policy["legacy_pure_modules"]]
    candidates = [*required, *sorted(path for package in policy.get("package_roots", ["qlh"])
                                    for path in (root / "src" / package).rglob("*.py"))]
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            violations.append(Violation(relative, 1, "missing_required_module", relative))
            continue
        legacy = path in required
        module = path.relative_to(root / "src").with_suffix("").as_posix().replace("/", ".")
        checked.append(relative)
        violations.extend(check_source(path.read_text(encoding="utf-8-sig"), relative, module,
                                       policy, legacy_pure=legacy))
    return violations, checked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    policy = load_policy(args.root / "refactor-boundaries.json")
    violations, checked = check_repository(args.root, policy)
    payload = {"passed": not violations, "checked": checked,
               "violations": [asdict(item) for item in violations]}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for item in violations:
            print(f"{item.path}:{item.line}: {item.code}: {item.detail}")
        print(f"refactor boundaries: {len(checked)} modules, {len(violations)} violations")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
