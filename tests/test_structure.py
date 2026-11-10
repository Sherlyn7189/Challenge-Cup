"""项目结构与模块契约测试。

运行方式：
    python -m unittest tests.test_structure -v
    python tests/test_structure.py
    python main.py --check

测试只验证“模块能组装、数据形状一致、读写往返正确”，不验证数值精度。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ertm.config import (  # noqa: E402
    AppConfig,
    ConfigError,
    GridConfig,
    ReceiverConfig,
    SolverConfig,
    load_config,
)
from ertm.data import NumpyDataStore  # noqa: E402
from ertm.diagnostics import format_checks, run_checks  # noqa: E402
from ertm.evaluation import BasicEvaluator  # noqa: E402
from ertm.imaging import ERTMImager  # noqa: E402
from ertm.models import LayeredModelBuilder  # noqa: E402
from ertm.pipeline import ERTMPipeline, _build_receivers, _build_source  # noqa: E402
from ertm.solver import PlaceholderForwardSolver  # noqa: E402


def tiny_config() -> AppConfig:
    """21x16 网格、20 步的微型配置。"""
    config = load_config()
    anomaly = config.model.anomaly
    if anomaly is not None:
        anomaly = replace(anomaly, x=100.0, z=80.0, rx=10.0, rz=8.0)
    return replace(
        config,
        grid=GridConfig(nx=21, nz=16, dx=10.0, dz=10.0, x0=0.0, z0=0.0),
        model=replace(config.model, anomaly=anomaly),
        source=replace(config.source, x=100.0, z=5.0),
        receivers=ReceiverConfig(count=5, x_start=0.0, x_end=200.0, z=5.0),
        solver=SolverConfig(nt=20, dt=0.001, order=8, boundary="cpml", dtype="float32"),
    )


class TestProjectStructure(unittest.TestCase):
    def test_all_structure_checks(self) -> None:
        results = run_checks(PROJECT_ROOT)
        failures = [item for item in results if not item.ok]
        self.assertFalse(failures, msg="\n" + format_checks(results))

    def test_config_rejects_invalid_grid(self) -> None:
        with self.assertRaises(ConfigError):
            GridConfig(nx=1, nz=10).validate()

    def test_component_contracts(self) -> None:
        config = tiny_config()
        model = LayeredModelBuilder().build(config)
        self.assertEqual(model.shape, config.grid.shape)
        self.assertGreater(float(model.vp.min()), 0.0)
        self.assertTrue(bool((model.vs < model.vp).all()))

        source = _build_source(config)
        receivers = _build_receivers(config)
        shot = PlaceholderForwardSolver(config.solver).forward(model, source, receivers)
        self.assertEqual(shot.data.shape, (config.solver.nt, receivers.nrec))
        self.assertTrue(shot.meta.get("placeholder", False))

        image = ERTMImager(config.imaging).migrate(shot, model)
        self.assertEqual(image.image.shape, model.shape)
        self.assertTrue(image.meta.get("placeholder", False))

        metrics = BasicEvaluator().evaluate(image, model, shot)
        self.assertTrue(metrics.values)
        self.assertIn("shot_rms", metrics.values)

    def test_data_roundtrip(self) -> None:
        config = tiny_config()
        model = LayeredModelBuilder().build(config)
        source = _build_source(config)
        receivers = _build_receivers(config)
        shot = PlaceholderForwardSolver(config.solver).forward(model, source, receivers)
        image = ERTMImager(config.imaging).migrate(shot, model)

        with tempfile.TemporaryDirectory() as tmp:
            store = NumpyDataStore(tmp)
            model_path = store.save_model(model, "unit_model")
            shot_path = store.save_shot(shot, "unit_shot")
            image_path = store.save_image(image, "unit_image")

            loaded_model = store.load_model(model_path)
            loaded_shot = store.load_shot(shot_path)
            loaded_image = store.load_image(image_path)

            self.assertTrue((loaded_model.vp == model.vp).all())
            self.assertTrue((loaded_shot.data == shot.data).all())
            self.assertTrue((loaded_image.image == image.image).all())

    def test_pipeline_dry_run(self) -> None:
        config = tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                config,
                output=replace(
                    config.output,
                    root=tmp,
                    save_figures=False,
                    save_data=False,
                ),
            )
            result = ERTMPipeline(config).run()
            self.assertIsNotNone(result.report_path)
            self.assertTrue(Path(result.report_path).exists())
            self.assertEqual(result.image.image.shape, result.model.shape)
            self.assertGreater(len(result.artifacts), 0)

    def test_cli_check(self) -> None:
        from main import main

        self.assertEqual(main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
