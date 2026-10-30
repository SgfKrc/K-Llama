"""Tests for topology.py. Run from the repository root:

    python -m unittest test_topology -v

The models are empty placeholder files: validation only asks whether the artifact
exists, so the tests never need real GGUFs.
"""
from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import topology
from topology import Stage, TopologyError


def _p(path: Path) -> str:
    return str(path).replace("\\", "/")


def _block(
    model: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 9101,
    layers: int = 8,
    kind: str = "gpu",
    gpu_layers: int = 0,
    cpu_moe: int = 0,
    cpus: str = "",
    threads: int = 3,
    bool_flag: bool = False,
) -> str:
    gpu_value = "true" if bool_flag else str(gpu_layers)
    return (
        "\n[[stage]]\n"
        f'model      = "{_p(model)}"\n'
        f'host       = "{host}"\n'
        f"port       = {port}\n"
        f"layers     = {layers}\n"
        f'kind       = "{kind}"\n'
        f"gpu_layers = {gpu_value}\n"
        f"cpu_moe    = {cpu_moe}\n"
        f'cpus       = "{cpus}"\n'
        f"threads    = {threads}\n"
    )


class TopologyTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.models = []
        for i in range(3):
            p = self.dir / f"stage-{i}.gguf"
            p.write_bytes(b"GGUF")
            self.models.append(p)

    def _write(self, text: str, name: str = "topology.toml") -> Path:
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def _three(self, overrides: dict[int, dict] | None = None) -> str:
        """gpu 8 + cpu 16 + tail 16 = 40 layers, the only legal full-width shape."""
        blocks = [
            dict(port=9101, layers=8, kind="gpu", gpu_layers=99, cpu_moe=0, cpus="0,1,2,3", threads=3),
            dict(port=9102, layers=16, kind="cpu", gpu_layers=0, cpu_moe=16, cpus="2,3,4,5,6,7", threads=3),
            dict(port=9103, layers=16, kind="tail", gpu_layers=0, cpu_moe=0, cpus="", threads=3),
        ]
        for i, block in enumerate(blocks):
            block.update((overrides or {}).get(i, {}))
        text = "".join(_block(self.models[i], **block) for i, block in enumerate(blocks))
        return text.lstrip("\n")

    # --- happy path -------------------------------------------------------

    def test_three_stage_config_parses(self) -> None:
        path = self._write(self._three())
        stages = topology.load(path)

        self.assertEqual(len(stages), 3)
        self.assertEqual([s.index for s in stages], [0, 1, 2])
        self.assertEqual([s.kind for s in stages], ["gpu", "cpu", "tail"])
        self.assertEqual([s.port for s in stages], [9101, 9102, 9103])
        self.assertEqual([s.layers for s in stages], [8, 16, 16])

        first = stages[0]
        self.assertIsInstance(first, Stage)
        self.assertEqual(first.model, self.models[0])
        self.assertEqual(first.gpu_layers, 99)
        self.assertEqual(first.cpus, "0,1,2,3")
        self.assertEqual(first.threads, 3)
        self.assertTrue(first.is_local)

    def test_two_stage_config_parses(self) -> None:
        text = (
            _block(self.models[0], port=9101, layers=8, kind="gpu", gpu_layers=99)
            + _block(self.models[1], port=9102, layers=32, kind="tail")
        )
        stages = topology.load(self._write(text.lstrip("\n")))
        self.assertEqual([s.kind for s in stages], ["gpu", "tail"])
        self.assertEqual(topology.total_layers(stages), 40)

    def test_optional_fields_default(self) -> None:
        text = (
            '[[stage]]\n'
            f'model  = "{_p(self.models[0])}"\n'
            'host   = "127.0.0.1"\n'
            "port   = 9101\n"
            "layers = 8\n"
            'kind   = "gpu"\n'
            '[[stage]]\n'
            f'model  = "{_p(self.models[1])}"\n'
            'host   = "10.0.0.5"\n'
            "port   = 9102\n"
            "layers = 32\n"
            'kind   = "tail"\n'
        )
        stages = topology.load(self._write(text))
        self.assertEqual(stages[0].gpu_layers, 0)
        self.assertEqual(stages[1].cpu_moe, 0)
        self.assertEqual(stages[0].cpus, "")
        self.assertEqual(stages[0].threads, 3)

    # --- rule 1: file exists and parses -----------------------------------

    def test_missing_file(self) -> None:
        with self.assertRaises(TopologyError) as ctx:
            topology.load(self.dir / "nope.toml")
        self.assertIn("does not exist", str(ctx.exception))

    def test_invalid_toml(self) -> None:
        path = self._write('[[stage]\nmodel = "x"\n')
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("not valid TOML", str(ctx.exception))

    def test_missing_model_file(self) -> None:
        path = self._write(_block(self.dir / "gone.gguf", kind="gpu").lstrip("\n"))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("model does not point to a file", str(ctx.exception))

    # --- rule 2: non-empty stage array ------------------------------------

    def test_empty_stage_array(self) -> None:
        with self.assertRaises(TopologyError) as ctx:
            topology.load(self._write(""))
        self.assertIn("non-empty", str(ctx.exception))

    # --- rules 3 and 4: kind order and uniqueness -------------------------

    def test_first_kind_must_be_gpu(self) -> None:
        path = self._write(self._three({0: dict(kind="cpu")}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("stage[0] is 'cpu' but must be 'gpu'", str(ctx.exception))

    def test_last_kind_must_be_tail(self) -> None:
        path = self._write(self._three({2: dict(kind="cpu")}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("stage[2] is 'cpu' but must be 'tail'", str(ctx.exception))

    def test_middle_kind_must_be_cpu(self) -> None:
        path = self._write(self._three({1: dict(kind="tail")}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("stage[1] is 'tail' but must be 'cpu'", str(ctx.exception))

    def test_two_gpu_stages(self) -> None:
        path = self._write(self._three({1: dict(kind="gpu")}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("stage[1] is 'gpu' but must be 'cpu'", str(ctx.exception))

    def test_unknown_kind(self) -> None:
        path = self._write(self._three({1: dict(kind="npu")}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("kind must be one of", str(ctx.exception))

    # --- rule 5: 4-aligned cuts -------------------------------------------

    def test_layers_must_be_multiple_of_four(self) -> None:
        path = self._write(self._three({0: dict(layers=6)}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        message = str(ctx.exception)
        self.assertIn("multiple of 4", message)
        # The reason has to travel with the error or the next person retries it.
        self.assertIn("full_attention_interval", message)

    # --- rule 6: total budget ---------------------------------------------

    def test_total_layers_exceeds_model(self) -> None:
        path = self._write(self._three({1: dict(layers=20), 2: dict(layers=20)}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("but the model has 40", str(ctx.exception))

    def test_total_layers_leaves_gap(self) -> None:
        # A gap is worse than an overlap: the chain runs on the wrong layers and
        # produces plausible output instead of failing to load. cpu_moe must be
        # lowered too, otherwise the per-stage bound fires first and masks the gap.
        path = self._write(self._three({1: dict(layers=8, cpu_moe=8),
                                        2: dict(layers=8, cpu_moe=8)}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("must cover every layer exactly", str(ctx.exception))

    # --- rule 7: port range -----------------------------------------------

    def test_port_below_range(self) -> None:
        with self.assertRaises(TopologyError) as ctx:
            topology.load(self._write(self._three({1: dict(port=80)})))
        self.assertIn("stage[1].port must be in [1024, 65535]", str(ctx.exception))

    def test_port_above_range(self) -> None:
        with self.assertRaises(TopologyError) as ctx:
            topology.load(self._write(self._three({2: dict(port=70000)})))
        self.assertIn("stage[2].port must be in [1024, 65535]", str(ctx.exception))

    # --- rule 8: unique ports ---------------------------------------------

    def test_duplicate_ports(self) -> None:
        with self.assertRaises(TopologyError) as ctx:
            topology.load(self._write(self._three({2: dict(port=9101)})))
        self.assertIn("unique", str(ctx.exception))

    # --- rule 9: counters -------------------------------------------------

    def test_negative_gpu_layers(self) -> None:
        with self.assertRaises(TopologyError) as ctx:
            topology.load(self._write(self._three({0: dict(gpu_layers=-1)})))
        self.assertIn("gpu_layers must be >= 0", str(ctx.exception))

    def test_cpu_moe_exceeds_layers(self) -> None:
        with self.assertRaises(TopologyError) as ctx:
            topology.load(self._write(self._three({1: dict(cpu_moe=17)})))
        self.assertIn("cpu_moe must be in [0, layers=16]", str(ctx.exception))

    def test_bool_is_not_an_int(self) -> None:
        path = self._write(self._three({0: dict(bool_flag=True)}))
        with self.assertRaises(TopologyError) as ctx:
            topology.load(path)
        self.assertIn("must be an integer", str(ctx.exception))

    # --- rule 10: tail gpu_layers warns, does not fail --------------------

    def test_tail_with_gpu_layers_warns(self) -> None:
        path = self._write(self._three({2: dict(gpu_layers=4)}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stages = topology.load(path)
        self.assertEqual(len(stages), 3)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertIn("tail", str(caught[0].message))

    # --- helpers ----------------------------------------------------------

    def test_total_layers(self) -> None:
        stages = topology.load(self._write(self._three()))
        self.assertEqual(topology.total_layers(stages), 40)
        self.assertEqual(topology.total_layers(()), 0)

    def test_sum_by_kind(self) -> None:
        stages = topology.load(self._write(self._three()))
        self.assertEqual(topology.sum_by_kind(stages), {"gpu": 8, "cpu": 16, "tail": 16})

    def test_local_stages(self) -> None:
        path = self._write(self._three({1: dict(host="192.168.1.20")}))
        stages = topology.load(path)
        local = topology.local_stages(stages)
        self.assertEqual([s.port for s in local], [9101, 9103])
        self.assertFalse(stages[1].is_local)
        self.assertTrue(stages[2].is_local)

    def test_localhost_and_ipv6_are_local(self) -> None:
        self.assertTrue(Stage(0, Path("m"), "localhost", 9101, 8, "gpu", 0, 0, "", 3).is_local)
        self.assertTrue(Stage(0, Path("m"), "::1", 9101, 8, "gpu", 0, 0, "", 3).is_local)
        self.assertFalse(Stage(0, Path("m"), "0.0.0.0", 9101, 8, "gpu", 0, 0, "", 3).is_local)


if __name__ == "__main__":
    unittest.main()
