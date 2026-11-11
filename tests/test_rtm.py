"""RTM 数值正确性与成像质量测试。

与 ``test_structure.py`` 的分工
-------------------------------
- ``test_structure.py``：接口/契约/IO/流水线能不能跑通；
- 本文件：数值算法本身是否正确——差分精度、CFL、检查点可复现性、
  绕射/反射归位、陡倾角成像、P/S 分离、Q 补偿、采集预处理。

所有用例都用小网格和短记录，保证整套测试在十几秒内跑完。
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

from ertm import numerics as num  # noqa: E402
from ertm.config import ImagingConfig, QConfig, SolverConfig, load_config  # noqa: E402
from ertm.imaging import RTMImager, build_imager, zero_lag_condition  # noqa: E402
from ertm.preprocess import (  # noqa: E402
    mute_direct_wave,
    prepare_shot,
    rotate_components,
    resample_shot,
)
from ertm.propagation import (  # noqa: E402
    AcousticPropagator,
    ElasticPropagator,
    PropagationSpec,
    build_propagator,
    check_stability,
    forward_shot,
    source_wavelet,
)
from ertm.qcomp import QCompensator, attenuate_record  # noqa: E402
from ertm.viscoelastic import (  # noqa: E402
    estimate_q_spectral_ratio,
    fit_gsls,
)
from ertm.propagation import QOptions, ViscoacousticPropagator, source_wavelet  # noqa: E402
from ertm.solver import build_solver  # noqa: E402
from ertm.types import ElasticModel, GridSpec, ReceiverSpec, ShotRecord, SourceSpec  # noqa: E402


# ----------------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------------
def make_model(
    nx: int = 121,
    nz: int = 91,
    dx: float = 10.0,
    dz: float = 10.0,
    vp: float = 3000.0,
    vs: float = 1700.0,
    rho: float = 2400.0,
) -> ElasticModel:
    grid = GridSpec(nx=nx, nz=nz, dx=dx, dz=dz)
    shape = grid.shape
    return ElasticModel(
        grid=grid,
        vp=np.full(shape, vp, dtype=np.float32),
        vs=np.full(shape, vs, dtype=np.float32),
        rho=np.full(shape, rho, dtype=np.float32),
        qp=np.full(shape, 100.0, dtype=np.float32),
        qs=np.full(shape, 50.0, dtype=np.float32),
    )


def layer_model(z_interface: float, vp1: float = 3000.0, vp2: float = 4000.0) -> ElasticModel:
    model = make_model(nx=141, nz=101)
    z = np.arange(model.grid.nz) * model.grid.dz
    mask = z >= z_interface
    vp = model.vp.copy()
    vp[mask, :] = vp2
    vp[~mask, :] = vp1
    return replace(model, vp=vp)


def dip_model(dip_deg: float = 70.0) -> ElasticModel:
    """陡倾角界面：过 (600, 300)，倾角 ``dip_deg``（相对水平）。"""
    model = make_model(nx=201, nz=151)
    x = np.arange(model.grid.nx) * model.grid.dx
    z = np.arange(model.grid.nz) * model.grid.dz
    xx, zz = np.meshgrid(x, z)
    shift = (zz - 300.0) * np.tan(np.deg2rad(90.0 - dip_deg))
    interface = (xx <= 600.0 + shift) & (zz >= 300.0)
    vp = np.where(interface, 4000.0, 3000.0).astype(np.float32)
    return replace(model, vp=vp)


def surface_receivers(model: ElasticModel, count: int, z: float = 20.0) -> ReceiverSpec:
    x = np.linspace(float(model.grid.x_min), float(model.grid.x_max), count)
    return ReceiverSpec(x=x, z=np.full_like(x, z), component="p")


def spec_for(model: ElasticModel, nt: int, dt: float = 5e-4, npml: int = 15) -> PropagationSpec:
    return PropagationSpec(
        dt=dt, nt=nt, order=8, npml=npml, dtype="float32", boundary="free"
    )


def source_at(x: float, z: float = 20.0, f0: float = 20.0, delay: float = 0.06) -> SourceSpec:
    return SourceSpec(x=x, z=z, f0=f0, delay=delay)


def muted_shot(
    model: ElasticModel,
    spec: PropagationSpec,
    source: SourceSpec,
    receivers: ReceiverSpec,
    *,
    engine: str = "acoustic",
    components: tuple[str, ...] = ("p",),
) -> ShotRecord:
    shot = forward_shot(model, source, receivers, spec, engine=engine, components=components)
    return prepare_shot(shot, model=model, mute=True, v_reference=float(np.percentile(model.vp, 25)))


# ----------------------------------------------------------------------------
# 数值层
# ----------------------------------------------------------------------------
class TestNumerics(unittest.TestCase):
    def test_staggered_derivative_accuracy(self) -> None:
        nx, nz, h = 401, 6, 1.0
        k = 2.0 * np.pi * 0.15
        x = np.arange(nx) * h
        field = np.tile(np.sin(k * x), (nz, 1))
        interior = slice(10, -10)
        analytic = k * np.cos(k * (x + h / 2))

        eighth = num.dplus(field, 1, num.staggered_coefficients(8), h)
        fourth = num.dplus(field, 1, num.staggered_coefficients(4), h)
        err8 = np.max(np.abs(eighth[0, interior] - analytic[interior])) / np.max(
            np.abs(analytic[interior])
        )
        err4 = np.max(np.abs(fourth[0, interior] - analytic[interior])) / np.max(
            np.abs(analytic[interior])
        )
        self.assertLess(err8, 1e-3)
        self.assertLess(err4, 1e-2)
        self.assertGreater(err4, err8)  # 高阶必须更准

    def test_sinc_resample_preserves_waveform(self) -> None:
        dt_in, dt_out = 4.0e-4, 1.0e-4
        t_in = np.arange(500) * dt_in
        signal = np.sin(2 * np.pi * 35.0 * t_in)
        resampled = num.resample_sinc(signal, dt_in, dt_out)
        t_out = np.arange(resampled.size) * dt_out
        analytic = np.sin(2 * np.pi * 35.0 * t_out)
        error = np.max(np.abs(resampled[:-8] - analytic[:-8]))
        self.assertLess(error, 2e-2)

    def test_cfl_rejects_unstable_step(self) -> None:
        model = make_model(vp=4000.0)
        bad = PropagationSpec(dt=2.0e-3, nt=10, order=8, npml=8, boundary="cpml")
        with self.assertRaises(Exception):
            check_stability(model, bad)
        good = PropagationSpec(dt=4.0e-4, nt=10, order=8, npml=8, boundary="cpml")
        check_stability(model, good)  # 不应抛错

    def test_cpml_absorbs_outgoing_wave(self) -> None:
        model = make_model(nx=101, nz=101)
        spec = PropagationSpec(dt=5e-4, nt=700, order=8, npml=12, dtype="float32", boundary="cpml")
        prop = AcousticPropagator(model, spec)
        prop.set_source(source_at(500.0, 500.0, f0=20.0, delay=0.02))
        wave = source_wavelet(prop._source, spec.dt, spec.nt)
        for step in range(spec.nt):
            prop.step(float(wave[step]))
        # 波传到边界被吸收后，全场能量应显著下降（否则会全反射回来）
        self.assertLess(prop.max_amplitude, 0.05 * float(np.abs(wave).max()) * 50.0)

    def test_checkpoint_replay_is_exact(self) -> None:
        model = make_model(nx=61, nz=51)
        spec = PropagationSpec(dt=5e-4, nt=80, order=8, npml=8, dtype="float32", boundary="cpml")
        source = source_at(300.0, 250.0, delay=0.03)
        wavelet = source_wavelet(source, spec.dt, spec.nt)

        reference = AcousticPropagator(model, spec)
        reference.set_source(source)
        for step in range(60):
            reference.step(float(wavelet[step]))
        expected = reference.p.copy()

        replay = AcousticPropagator(model, spec)
        replay.set_source(source)
        replay.step(float(wavelet[0]))
        state = replay.state()
        replay.load_state(state)
        for step in range(1, 60):
            replay.step(float(wavelet[step]))
        self.assertTrue(np.allclose(replay.p, expected, atol=1e-6))

        # 检查点大小必须等于“场 + PML 记忆”的字节数（否则内存预算会算错）
        self.assertGreater(replay.state_bytes(), 0)
        expected_bytes = sum(
            getattr(replay, name).nbytes for name in ("p", "vx", "vz")
        ) + replay.pml.memory_bytes()
        self.assertEqual(replay.state_bytes(), expected_bytes)


# ----------------------------------------------------------------------------
# 正演
# ----------------------------------------------------------------------------
class TestForwardModelling(unittest.TestCase):
    def test_moveout_matches_constant_velocity(self) -> None:
        model = make_model(nx=161, nz=121)
        spec = PropagationSpec(dt=4e-4, nt=700, order=8, npml=15, dtype="float32", boundary="cpml")
        source = source_at(800.0, 600.0, f0=20.0, delay=0.06)
        receivers = ReceiverSpec(x=np.array([1000.0, 1300.0]), z=np.array([600.0, 600.0]))
        shot = forward_shot(model, source, receivers, spec, engine="acoustic", components=("p",))
        wavelet = source_wavelet(source, spec.dt, spec.nt)

        def lag(trace: np.ndarray) -> float:
            correlation = np.correlate(trace, wavelet, mode="full")
            return (int(np.argmax(np.abs(correlation))) - (wavelet.size - 1)) * spec.dt

        measured = lag(shot.data[:, 1]) - lag(shot.data[:, 0])
        expected = (500.0 - 200.0) / 3000.0
        self.assertAlmostEqual(measured, expected, delta=0.01)

    def test_pure_p_source_has_no_s_energy(self) -> None:
        model = make_model(nx=101, nz=101)
        spec = PropagationSpec(dt=4e-4, nt=350, order=8, npml=12, dtype="float32", boundary="cpml")

        def energies(mechanism: str) -> tuple[float, float]:
            prop = ElasticPropagator(model, spec)
            prop.set_source(
                SourceSpec(x=500.0, z=500.0, f0=20.0, delay=0.04, mechanism=mechanism)
            )
            wave = source_wavelet(prop._source, spec.dt, spec.nt)
            for step in range(spec.nt):
                prop.step(float(wave[step]))
            fields = prop.image_fields("divcurl")
            # 去掉 PML 区后统计能量
            pad = prop.pad
            interior = (slice(pad + 12, -pad - 12), slice(pad + 12, -pad - 12))
            div = np.asarray(fields["P"])[interior]
            curl = np.asarray(fields["S"])[interior]
            return float(np.sum(div**2)), float(np.sum(curl**2))

        p_div, p_curl = energies("explosive")
        s_div, s_curl = energies("shear")
        self.assertLess(p_curl / p_div, 0.05, "爆炸源不应产生可观的 S 能量")
        self.assertGreater(s_curl / (s_div + 1e-30), 0.5, "剪切源应产生主导的 S 能量")


# ----------------------------------------------------------------------------
# 成像
# ----------------------------------------------------------------------------
class TestRTMImaging(unittest.TestCase):
    def test_image_of_layer_at_correct_depth(self) -> None:
        interface = 500.0
        model = layer_model(interface, vp1=3000.0, vp2=4000.0)
        spec = spec_for(model, nt=900)
        source = source_at(700.0)
        receivers = surface_receivers(model, 101)
        shot = muted_shot(model, spec, source, receivers)

        imager = RTMImager(
            ImagingConfig(mode="pp", condition="normalized", laplacian=True, lowpass_fraction=0.85),
            SolverConfig(nt=spec.nt, dt=spec.dt, order=8, boundary="free", engine="acoustic", npml=15),
        )
        image = imager.migrate(shot, model)
        magnitude = np.abs(image.image)
        column = magnitude[:, magnitude.shape[1] // 2]
        peak_depth = float(np.argmax(column)) * model.grid.dz
        self.assertLess(abs(peak_depth - interface), 100.0)
        self.assertTrue(np.all(np.isfinite(image.image)))
        self.assertFalse(image.meta["placeholder"])

    def test_zero_lag_condition_is_pointwise_product(self) -> None:
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        b = np.ones((3, 4))
        self.assertTrue(np.allclose(zero_lag_condition(a, b), a))

    def test_steep_dip_is_imaged(self) -> None:
        model = dip_model(dip_deg=70.0)
        spec = spec_for(model, nt=1400, npml=20)
        source = source_at(1000.0)
        receivers = surface_receivers(model, 201)
        shot = muted_shot(model, spec, source, receivers)

        imager = RTMImager(
            ImagingConfig(mode="pp", condition="normalized", laplacian=True, lowpass_fraction=0.85),
            SolverConfig(nt=spec.nt, dt=spec.dt, order=8, boundary="free", engine="acoustic", npml=20),
        )
        image = imager.migrate(shot, model)
        magnitude = np.abs(image.image)

        nz = magnitude.shape[0]
        rows = np.arange(30, nz - 5)
        dip_x = 600.0 + (rows * model.grid.dz - 300.0) * np.tan(np.deg2rad(20.0))
        valid = (dip_x > 20) & (dip_x < model.grid.x_max - 20)
        rows, dip_x = rows[valid], dip_x[valid]
        ix = np.round(dip_x / model.grid.dx).astype(int)
        on_dip = np.zeros_like(magnitude, dtype=bool)
        off_dip = np.zeros_like(magnitude, dtype=bool)
        for row, col in zip(rows, ix):
            on_dip[row, max(col - 1, 0) : col + 2] = True
            shifted = col + 30
            if shifted + 1 < magnitude.shape[1]:
                off_dip[row, shifted - 1 : shifted + 2] = True
        ratio = float(np.sum(magnitude[on_dip] ** 2) / np.sum(magnitude[off_dip] ** 2))
        self.assertGreater(ratio, 5.0, f"陡倾角界面成像能量比只有 {ratio:.2f}")

    def test_elastic_multicomponent_imaging(self) -> None:
        model = layer_model(400.0, vp1=3000.0, vp2=3600.0)
        spec = spec_for(model, nt=700)
        source = source_at(700.0)
        receivers = surface_receivers(model, 61)
        shot = forward_shot(
            model, source, receivers, spec, engine="elastic", components=("p", "vx", "vz")
        )
        shot = prepare_shot(shot, model=model, mute=True, v_reference=3000.0)

        for mode in ("pp", "ps"):
            imager = RTMImager(
                ImagingConfig(
                    mode=mode, condition="normalized", separation="divcurl", laplacian=True
                ),
                SolverConfig(
                    nt=spec.nt, dt=spec.dt, order=8, boundary="free", engine="elastic", npml=15
                ),
            )
            image = imager.migrate(shot, model)
            self.assertEqual(image.image.shape, model.shape)
            self.assertTrue(np.all(np.isfinite(image.image)))
            self.assertGreater(float(np.max(np.abs(image.image))), 0.0)


# ----------------------------------------------------------------------------
# Q 补偿与预处理
# ----------------------------------------------------------------------------
class TestQCompensation(unittest.TestCase):
    def test_attenuation_then_compensation_restores_amplitude(self) -> None:
        dt = 2.5e-4
        t = np.arange(1600) * dt
        pulse = num.ricker(t, 25.0, 0.12).astype(np.float64)[:, None]
        attenuated = attenuate_record(pulse, dt, q=40.0, f_ref=30.0)
        self.assertLess(float(np.abs(attenuated).max()), float(np.abs(pulse).max()))

        compensator = QCompensator(
            config=QConfig(enabled=True, alpha_c=0.9, f_ref=30.0, gain_cap=50.0),
            q_value=40.0,
        )
        restored = compensator(attenuated, dt)
        self.assertTrue(np.all(np.isfinite(restored)))
        ratio = float(np.abs(restored).max() / np.abs(pulse).max())
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 3.0)

    def test_compensation_disabled_is_identity(self) -> None:
        data = np.random.default_rng(0).normal(size=(128, 3))
        compensator = QCompensator(config=QConfig(enabled=False), q_value=50.0)
        self.assertTrue(np.allclose(compensator(data, 1e-3), data))


class TestGSLSViscoacoustic(unittest.TestCase):
    """GSLS 记忆变量黏声正演与反传补偿。"""

    def test_constant_q_fit_over_band(self) -> None:
        params = fit_gsls(40.0, 25.0, n_mechanisms=3, band_ratio=1.5)
        freqs = np.geomspace(25.0 / 1.5, 25.0 * 1.5, 9)
        quality = params.quality(freqs)
        deviation = np.max(np.abs(quality / 40.0 - 1.0))
        self.assertLess(deviation, 0.02, f"常 Q 拟合偏差 {deviation:.3f}")
        self.assertGreater(params.k_unrelaxed, 1.0)
        self.assertLess(params.k_unrelaxed, 1.2)

    def test_single_mechanism_peaks_at_reference_frequency(self) -> None:
        params = fit_gsls(40.0, 25.0, n_mechanisms=1)
        self.assertAlmostEqual(float(params.quality([25.0])[0]), 40.0, delta=0.5)
        self.assertGreater(float(params.quality([10.0])[0]), 40.0)

    def test_viscoacoustic_measures_target_q(self) -> None:
        grid = GridSpec(nx=141, nz=121, dx=10.0, dz=10.0)
        shape = grid.shape
        model = ElasticModel(
            grid=grid,
            vp=np.full(shape, 3000.0, dtype=np.float32),
            vs=np.full(shape, 1700.0, dtype=np.float32),
            rho=np.full(shape, 2400.0, dtype=np.float32),
            qp=np.full(shape, 40.0, dtype=np.float32),
            qs=np.full(shape, 50.0, dtype=np.float32),
        )
        spec = PropagationSpec(dt=4e-4, nt=900, order=8, npml=12, dtype="float64", boundary="cpml")
        source = SourceSpec(x=600.0, z=600.0, f0=25.0, delay=0.03)
        receivers = ReceiverSpec(x=np.array([800.0, 1000.0]), z=np.array([600.0, 600.0]))
        prop = ViscoacousticPropagator(
            model, spec, q_options=QOptions(enabled=True, f_ref=25.0)
        )
        prop.set_source(source)
        prop.set_receivers(receivers)
        wave = source_wavelet(source, spec.dt, spec.nt)
        data = np.zeros((spec.nt, receivers.nrec))
        for step in range(spec.nt):
            prop.step(float(wave[step]))
            data[step] = prop.sample(("p",))["p"]

        def windowed_spectrum(trace: np.ndarray, offset: float) -> np.ndarray:
            start = max(int((source.delay + offset / 3000.0) / spec.dt) - 100, 0)
            segment = trace[start : start + 400] * np.hanning(400)
            return np.abs(np.fft.rfft(segment))

        freqs = np.fft.rfftfreq(400, spec.dt)
        s1 = windowed_spectrum(data[:, 0], 200.0)
        s2 = windowed_spectrum(data[:, 1], 400.0)
        band = (freqs > 5) & (freqs < 60)
        slope = np.polyfit(
            freqs[band], np.log(s1[band] / s2[band]) + 0.5 * np.log(2.0), 1
        )[0]
        measured = float(np.pi * (400.0 - 200.0) / 3000.0 / max(slope, 1e-12))
        self.assertLess(abs(measured / 40.0 - 1.0), 0.2, f"测得 Q={measured:.1f}，目标 40")

    def test_compensation_recovers_deep_amplitude(self) -> None:
        def migrate(alpha_c: float) -> float:
            grid = GridSpec(nx=121, nz=101, dx=10.0, dz=10.0)
            shape = grid.shape
            z = np.arange(grid.nz) * grid.dz
            vp = np.where(z[:, None] < 600.0, 3000.0, 4200.0) + 0.0 * np.arange(grid.nx)[None, :]
            qp_node = np.where(z[:, None] < 600.0, 60.0, 15.0) + 0.0 * np.arange(grid.nx)[None, :]
            model = ElasticModel(
                grid=grid,
                vp=vp.astype(np.float32),
                vs=np.full(shape, 1700.0, dtype=np.float32),
                rho=np.full(shape, 2400.0, dtype=np.float32),
                qp=qp_node.astype(np.float32),
                qs=np.full(shape, 50.0, dtype=np.float32),
            )
            solver = SolverConfig(
                # 记录长度必须覆盖 600 m 界面的双程走时，否则测的是浅部假象
                nt=1500, dt=4e-4, order=8, boundary="free",
                dtype="float32", engine="viscoacoustic", npml=12,
            )
            spec = PropagationSpec.from_config(solver, model)
            source = SourceSpec(x=600.0, z=20.0, f0=20.0, delay=0.06)
            x = np.linspace(0.0, 1200.0, 81)
            receivers = ReceiverSpec(x=x, z=np.full_like(x, 20.0), component="p")
            shot = forward_shot(model, source, receivers, spec, engine="viscoacoustic")
            shot = prepare_shot(shot, model=model, mute=True, v_reference=3000.0)
            q = QConfig(enabled=alpha_c > 0.0, model="gsls", alpha_c=max(alpha_c, 0.0))
            image = build_imager(
                ImagingConfig(
                    condition="normalized", laplacian=True,
                    lowpass_fraction=0.85, checkpoint_mb=64.0,
                ),
                solver,
                q,
            ).migrate(shot, model)
            magnitude = np.abs(image.image)
            # 600 m 界面对应的能量窗口（±50 m）
            return float(np.sqrt((magnitude[55:65, :] ** 2).mean()))

        baseline = migrate(0.0)
        compensated = migrate(1.0)
        self.assertGreater(compensated, 1.15 * baseline,
                           f"补偿未提升深部振幅: {baseline:.3e} -> {compensated:.3e}")


class TestPreprocessing(unittest.TestCase):
    def test_four_component_rotation_preserves_energy(self) -> None:
        rng = np.random.default_rng(1)
        vx = rng.normal(size=(64, 5)).astype(np.float32)
        vz = rng.normal(size=(64, 5)).astype(np.float32)
        rotated = rotate_components({"p": vx + vz, "vx": vx, "vz": vz}, 30.0)
        before = np.sum(vx**2) + np.sum(vz**2)
        after = np.sum(np.asarray(rotated["vx"]) ** 2) + np.sum(np.asarray(rotated["vz"]) ** 2)
        self.assertAlmostEqual(float(before), float(after), places=2)

    def test_rotation_by_90_degrees_swaps_components(self) -> None:
        vx = np.array([[1.0], [2.0]], dtype=np.float32)
        vz = np.array([[3.0], [4.0]], dtype=np.float32)
        rotated = rotate_components({"vx": vx, "vz": vz}, 90.0)
        self.assertTrue(np.allclose(rotated["vx"], vz, atol=1e-6))
        self.assertTrue(np.allclose(rotated["vz"], -vx, atol=1e-6))

    def test_record_resample_roundtrip(self) -> None:
        model = make_model(nx=61, nz=51)
        spec = PropagationSpec(dt=5e-4, nt=200, order=8, npml=8, boundary="cpml")
        source = source_at(300.0, 20.0)
        receivers = surface_receivers(model, 11)
        shot = forward_shot(model, source, receivers, spec, engine="acoustic", components=("p",))

        coarse = resample_shot(shot, 1.0e-3)
        self.assertAlmostEqual(coarse.dt, 1.0e-3)
        fine = resample_shot(coarse, 5e-4)
        self.assertAlmostEqual(fine.dt, 5e-4)
        self.assertEqual(fine.data.shape[1], shot.data.shape[1])
        self.assertTrue(np.all(np.isfinite(fine.data)))

    def test_direct_mute_removes_early_energy(self) -> None:
        model = make_model(nx=61, nz=51)
        spec = PropagationSpec(dt=5e-4, nt=400, order=8, npml=8, boundary="cpml")
        source = source_at(300.0, 20.0)
        receivers = surface_receivers(model, 21)
        shot = forward_shot(model, source, receivers, spec, engine="acoustic", components=("p",))
        muted = mute_direct_wave(shot, v_reference=3000.0)
        early = shot.t < 0.08
        self.assertGreater(
            float(np.abs(shot.data[early]).max()), float(np.abs(muted.data[early]).max())
        )
        self.assertLess(float(np.abs(muted.data[early]).max()), 0.3 * float(np.abs(shot.data).max()))


# ----------------------------------------------------------------------------
# 端到端
# ----------------------------------------------------------------------------
class TestPipelineEndToEnd(unittest.TestCase):
    def test_pipeline_runs_and_produces_image(self) -> None:
        config = load_config()
        anomaly = config.model.anomaly
        if anomaly is not None:
            anomaly = replace(anomaly, x=200.0, z=150.0, rx=25.0, rz=15.0)
        config = replace(
            config,
            grid=replace(config.grid, nx=41, nz=31, dx=10.0, dz=10.0),
            model=replace(config.model, anomaly=anomaly),
            source=replace(config.source, x=200.0, z=20.0, delay=0.05),
            receivers=replace(config.receivers, count=9, x_start=0.0, x_end=400.0, z=20.0),
            solver=SolverConfig(
                nt=200, dt=2.5e-4, order=8, boundary="free", engine="acoustic", npml=6
            ),
            imaging=replace(config.imaging, checkpoint_mb=64.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                config,
                output=replace(config.output, root=tmp, save_data=False, save_figures=False),
            )
            from ertm.pipeline import ERTMPipeline

            result = ERTMPipeline(config).run()
            self.assertIsNotNone(result.report_path)
            self.assertTrue(Path(result.report_path).exists())
            self.assertEqual(result.image.image.shape, result.model.shape)
            self.assertTrue(np.all(np.isfinite(result.image.image)))
            self.assertIn("image_energy", result.metrics.values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
