"""SEG-Y 输入输出测试。

覆盖：
- 单炮写出/读回的往返一致性（数据、采样间隔、几何）；
- 多炮合并文件按 ``SourceX`` 分组；
- 深度域图像的 SEG-Y 往返；
- 用 SEG-Y 作为输入的完整流水线（不跑正演，直接读资料 → 成像 → 写 SEG-Y）。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ertm import segy  # noqa: E402
from ertm.config import InputConfig, load_config  # noqa: E402
from ertm.diagnostics import _tiny_config  # noqa: E402
from ertm.models import LayeredModelBuilder  # noqa: E402
from ertm.pipeline import ERTMPipeline, _build_receivers, _build_source  # noqa: E402
from ertm.solver import build_solver  # noqa: E402
from ertm.types import GridSpec, ImageResult, ShotRecord  # noqa: E402


def make_shot(source_x: float = 100.0, nrec: int = 11, nt: int = 64) -> ShotRecord:
    from ertm.types import ReceiverSpec, SourceSpec

    x = np.linspace(0.0, 200.0, nrec)
    data = np.random.default_rng(int(source_x)).normal(size=(nt, nrec)).astype(np.float32)
    return ShotRecord(
        data=data,
        dt=4.0e-4,
        t=np.arange(nt, dtype=np.float64) * 4.0e-4,
        receivers=ReceiverSpec(x=x, z=np.full(nrec, 20.0), component="p"),
        source=SourceSpec(x=source_x, z=20.0, f0=20.0),
        components={"p": data},
    )


class TestSegyRoundTrip(unittest.TestCase):
    def test_shot_roundtrip(self) -> None:
        shot = make_shot(120.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot.sgy"
            segy.write_segy_shot(shot, path)
            loaded = segy.read_segy_shots(path)
        self.assertEqual(len(loaded), 1)
        record = loaded[0]
        self.assertEqual(record.receivers.nrec, shot.receivers.nrec)
        self.assertAlmostEqual(record.dt, shot.dt, places=9)
        self.assertAlmostEqual(record.source.x, shot.source.x, places=3)
        self.assertTrue(np.allclose(record.data, shot.data, atol=2e-3))
        self.assertTrue(np.allclose(record.receivers.x, shot.receivers.x, atol=1e-6))

    def test_multi_shot_grouping(self) -> None:
        shots = [make_shot(100.0), make_shot(300.0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.sgy"
            segy.write_segy_shots(shots, path)
            grouped = segy.read_segy_shots(path, segy.SegYReadOptions(group_by="source"))
            single = segy.read_segy_shots(path, segy.SegYReadOptions(group_by="file"))
        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0].receivers.nrec, 2 * shots[0].receivers.nrec)
        sources = sorted(record.source.x for record in grouped)
        self.assertAlmostEqual(sources[0], 100.0, places=3)
        self.assertAlmostEqual(sources[1], 300.0, places=3)
        for record in grouped:
            self.assertEqual(record.receivers.nrec, shots[0].receivers.nrec)

    def test_image_roundtrip(self) -> None:
        grid = GridSpec(nx=13, nz=17, dx=10.0, dz=5.0)
        image = ImageResult(
            image=np.random.default_rng(3).normal(size=grid.shape).astype(np.float32),
            grid=grid,
            mode="pp",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.sgy"
            segy.write_segy_image(image, path)
            back, dz, x = segy.read_segy_image(path)
        self.assertEqual(back.shape, grid.shape)
        self.assertAlmostEqual(dz, grid.dz, places=6)
        self.assertTrue(np.allclose(x, np.arange(grid.nx) * grid.dx, atol=1e-6))
        self.assertTrue(np.allclose(back, image.image, atol=1e-6))


class TestSegyPipeline(unittest.TestCase):
    def test_pipeline_runs_from_segy_input(self) -> None:
        config = _tiny_config(load_config())
        model = LayeredModelBuilder().build(config)
        solver = build_solver(config.solver, config.q)
        shot = solver.forward(
            model,
            _build_source(config),
            _build_receivers(config),
            components=("p",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "input.sgy"
            segy.write_segy_shot(shot, data_path)
            config = replace(
                config,
                input=InputConfig(
                    format="segy",
                    segy_path=str(data_path),
                    component="p",
                    group_by="source",
                    receiver_depth=20.0,
                    source_depth=20.0,
                ),
                output=replace(
                    config.output,
                    root=str(tmp_path / "output"),
                    save_figures=False,
                    save_data=True,
                ),
            )
            result = ERTMPipeline(config).run()
            self.assertEqual(result.shot.meta.get("input"), "segy")
            self.assertTrue(np.all(np.isfinite(result.image.image)))
            self.assertGreater(float(np.max(np.abs(result.image.image))), 0.0)
            report = result.report_path.read_text(encoding="utf-8")
            self.assertIn('"format": "segy"', report)
            self.assertTrue(
                (tmp_path / "output" / "images" / "image_pp.sgy").exists()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
