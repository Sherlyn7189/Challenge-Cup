"""项目结构与模块契约测试。

运行方式：
    python -m unittest tests.test_structure -v
    python tests/test_structure.py
    python main.py --check

本文件只验证“模块能组装、数据形状一致、读写往返正确”；数值精度与成像
质量由 ``tests/test_rtm.py`` 负责。
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
from ertm.diagnostics import _tiny_config, format_checks, run_checks  # noqa: E402
from ertm.evaluation import BasicEvaluator  # noqa: E402
from ertm.imaging import build_imager  # noqa: E402
from ertm.models import LayeredModelBuilder  # noqa: E402
from ertm.pipeline import ERTMPipeline, _build_receivers, _build_source  # noqa: E402
from ertm.solver import build_solver  # noqa: E402


def tiny_config() -> AppConfig:
    """41x31 网格、120 步的微型配置（与 ``--check`` 保持一致）。"""
    return _tiny_config(load_config())


class TestProjectStructure(unittest.TestCase):
    def test_all_structure_checks(self) -> None:
        results = run_checks(PROJECT_ROOT)
        failures = [item for item in results if not item.ok]
        self.assertFalse(failures, msg="\n" + format_checks(results))

    def test_config_rejects_invalid_grid(self) -> None:
        with self.assertRaises(ConfigError):
            GridConfig(nx=1, nz=10).validate()

    def test_config_rejects_unsupported_engine(self) -> None:
        with self.assertRaises(ConfigError):
            SolverConfig(engine="spectral").validate()

    def test_component_contracts(self) -> None:
        config = tiny_config()
        model = LayeredModelBuilder().build(config)
        self.assertEqual(model.shape, config.grid.shape)
        self.assertGreater(float(model.vp.min()), 0.0)
        self.assertTrue(bool((model.vs < model.vp).all()))

        source = _build_source(config)
        receivers = _build_receivers(config)
        solver = build_solver(config.solver)
        shot = solver.forward(model, source, receivers)
        self.assertEqual(shot.data.shape, (config.solver.nt, receivers.nrec))
        self.assertFalse(shot.meta.get("placeholder", True))
        self.assertTrue(np_all_finite(shot.data))

        imager = build_imager(config.imaging, config.solver)
        image = imager.migrate(shot, model)
        self.assertEqual(image.image.shape, model.shape)
        self.assertFalse(image.meta.get("placeholder", True))
        self.assertTrue(np_all_finite(image.image))

        metrics = BasicEvaluator().evaluate(image, model, shot)
        self.assertTrue(metrics.values)
        self.assertIn("shot_rms", metrics.values)
        self.assertIn("image_energy", metrics.values)

    def test_data_roundtrip(self) -> None:
        config = tiny_config()
        model = LayeredModelBuilder().build(config)
        source = _build_source(config)
        receivers = _build_receivers(config)
        solver = build_solver(config.solver)
        shot = solver.forward(model, source, receivers)
        image = build_imager(config.imaging, config.solver).migrate(shot, model)

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
            # 多分量记录必须完整往返
            self.assertEqual(loaded_shot.component_names, shot.component_names)
            for name in shot.components:
                self.assertTrue((loaded_shot.components[name] == shot.components[name]).all())

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


def np_all_finite(array) -> bool:
    import numpy as np

    return bool(np.all(np.isfinite(array)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
