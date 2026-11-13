"""ERTM 求解器工作台 · Qt/QML 与 SeismicProjectERTM 的桥接层。

职责
----
- 读取 ERTM 项目的 ``config/default.json`` 与 ``output/`` 运行报告；
- 把模型、炮记录、成像结果渲染成适合深色界面的 PNG 预览；
- 枚举 ``ertm/*.py`` 源码模块，供界面查看；
- 通过 QProcess 调用项目虚拟环境执行 ``main.py``，并把日志回传 QML。

界面本身只负责展示和交互，算法全部来自 ERTM 项目。
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    Property,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QDesktopServices, QImage
from PySide6.QtCore import Qt

# ────────────────────────────────────────────────────────────────
# 视觉配色（与 QML 主题保持一致）
# ────────────────────────────────────────────────────────────────
DARK_BASE = (6, 18, 30)
MODEL_STOPS = [
    (0.00, (7, 24, 40)),
    (0.18, (12, 50, 78)),
    (0.38, (20, 98, 130)),
    (0.58, (30, 155, 150)),
    (0.78, (128, 208, 140)),
    (1.00, (245, 211, 94)),
]
POS_STOPS = [
    (0.00, (7, 20, 32)),
    (0.45, (16, 52, 82)),
    (0.72, (160, 70, 76)),
    (1.00, (244, 112, 112)),
]
NEG_STOPS = [
    (0.00, (7, 20, 32)),
    (0.45, (14, 48, 84)),
    (0.72, (44, 112, 186)),
    (1.00, (100, 180, 255)),
]


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * t)),
        int(round(c1[1] + (c2[1] - c1[1]) * t)),
        int(round(c1[2] + (c2[2] - c1[2]) * t)),
    )


def _colormap(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """把 0..1 的数组映射为 RGB uint8 图像数组。"""
    flat = values.astype(np.float64).reshape(-1)
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    out = np.empty((flat.size, 3), dtype=np.uint8)
    for channel in range(3):
        colors = np.array([s[1][channel] for s in stops], dtype=np.float64)
        out[:, channel] = np.clip(np.interp(flat, xs, colors), 0, 255).astype(np.uint8)
    return out.reshape(values.shape + (3,))


def _normalize(data: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
    hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _seismic_rgb(data: np.ndarray, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    peak = float(np.nanmax(np.abs(arr))) if vmax is None or vmax <= 0 else float(vmax)
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0
    norm = np.clip(arr / peak, -1.0, 1.0)
    pos = _colormap(np.clip(norm, 0.0, 1.0), POS_STOPS)
    neg = _colormap(np.clip(-norm, 0.0, 1.0), NEG_STOPS)
    return np.where((norm >= 0)[..., None], pos, neg).astype(np.uint8)


def _qimage(rgb: np.ndarray, size: tuple[int, int], keep_aspect: bool = True) -> QImage:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    return image.scaled(
        size[0],
        size[1],
        Qt.AspectRatioMode.KeepAspectRatio if keep_aspect else Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class ErtmBridge(QObject):
    """QML 可直接绑定的 ERTM 项目数据源。"""

    cfgChanged = Signal()
    checksChanged = Signal()
    modulesChanged = Signal()
    comparisonsChanged = Signal()
    metricsChanged = Signal()
    artifactsChanged = Signal()
    previewsChanged = Signal()
    codeChanged = Signal()
    logChanged = Signal()
    runningChanged = Signal()
    progressChanged = Signal()
    runStateChanged = Signal()
    projectChanged = Signal()
    environmentChanged = Signal()
    lastRunChanged = Signal()
    realDataChanged = Signal()
    shotInfoChanged = Signal()
    qChanged = Signal()
    selectionChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._root: Path | None = self._detect_root()
        self._cfg: dict[str, Any] = {}
        self._checks: list[dict[str, Any]] = []
        self._modules: list[dict[str, Any]] = []
        self._comparisons: list[dict[str, Any]] = []
        self._metrics: list[dict[str, Any]] = []
        self._artifacts: list[dict[str, Any]] = []
        self._previews: dict[str, str] = {}
        self._environment: dict[str, Any] = {}
        self._real_data: dict[str, Any] = {}
        self._shot_info: dict[str, Any] = {}
        self._q: dict[str, Any] = {}
        self._q_previews: dict[str, str] = {}
        self._log = ""
        self._code = ""
        self._code_file = ""
        self._code_title = "尚未选择模块"
        self._code_role = ""
        self._code_status = ""
        self._code_lines = 0
        self._running = False
        self._progress = 0.0
        self._run_state = "空闲"
        self._last_run = "—"
        self._version = int(time.time())
        self._cache_dir = Path(__file__).resolve().parent / ".ui_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._process: QProcess | None = None
        self._ticker = QTimer(self)
        self._ticker.setInterval(160)
        self._ticker.timeout.connect(self._tick)
        self._connected_logged = False

        self._append_log("[UI] ERTM 求解器工作台启动。")
        if self._root is None:
            self._append_log("[UI] 未找到 ERTM 项目，请设置 ERTM_PROJECT_ROOT。")
        self.reload()

    # ── 路径与项目发现 ──────────────────────────────────────────
    @staticmethod
    def _detect_root() -> Path | None:
        candidates: list[Path] = []
        env_value = os.environ.get("ERTM_PROJECT_ROOT", "").strip()
        if env_value:
            candidates.append(Path(env_value))
        here = Path(__file__).resolve().parent
        candidates.extend(
            [
                # 作为 Challenge-Cup 仓库的 workbench/ 子目录运行时，
                # ERTM 代码位于上一级仓库根目录。
                here,
                here.parent,
                here.parent.parent,
                here.parent / "SeismicProjectERTM",
                here.parent.parent / "SeismicProjectERTM",
                here.parent.parent / "PycharmProjects" / "SeismicProjectERTM",
                # 独立工作台在本机的默认项目位置。
                Path(r"D:\PycharmProjects\SeismicProjectERTM"),
                Path(r"C:\Users\30967\PycharmProjects\SeismicProjectERTM"),
            ]
        )
        for candidate in candidates:
            if (candidate / "main.py").is_file() and (candidate / "config" / "default.json").is_file():
                return candidate.resolve()
        return None

    @Property(str, notify=projectChanged)
    def projectRoot(self) -> str:
        return "" if self._root is None else str(self._root)

    @Property(bool, notify=projectChanged)
    def projectOk(self) -> bool:
        return self._root is not None

    @Property(str, notify=projectChanged)
    def configPath(self) -> str:
        return "" if self._root is None else str(self._root / "config" / "default.json")

    @Property(str, notify=projectChanged)
    def outputRoot(self) -> str:
        if self._root is None:
            return ""
        root = self._cfg.get("output", {}).get("root", "output") if self._cfg else "output"
        path = Path(root)
        return str(path if path.is_absolute() else self._root / path)

    @Property(str, notify=projectChanged)
    def projectName(self) -> str:
        return str(self._cfg.get("project", {}).get("name", "ERTM"))

    @Property(str, notify=projectChanged)
    def projectVersion(self) -> str:
        return str(self._cfg.get("project", {}).get("version", "—"))

    # ── 配置与派生校验 ──────────────────────────────────────────
    @Property("QVariantMap", notify=cfgChanged)
    def cfg(self) -> dict[str, Any]:
        return self._cfg

    @Property("QVariantList", notify=checksChanged)
    def checks(self) -> list[dict[str, Any]]:
        return self._checks

    @Property("QVariantList", notify=modulesChanged)
    def modules(self) -> list[dict[str, Any]]:
        return self._modules

    @Property("QVariantList", notify=comparisonsChanged)
    def comparisons(self) -> list[dict[str, Any]]:
        return self._comparisons

    @Property("QVariantList", notify=metricsChanged)
    def metrics(self) -> list[dict[str, Any]]:
        return self._metrics

    @Property("QVariantList", notify=artifactsChanged)
    def artifacts(self) -> list[dict[str, Any]]:
        return self._artifacts

    @Property("QVariantMap", notify=previewsChanged)
    def previews(self) -> dict[str, str]:
        return self._previews

    @Property("QVariantMap", notify=environmentChanged)
    def environment(self) -> dict[str, Any]:
        return self._environment

    @Property("QVariantMap", notify=realDataChanged)
    def realData(self) -> dict[str, Any]:
        return self._real_data

    @Property("QVariantMap", notify=shotInfoChanged)
    def shotInfo(self) -> dict[str, Any]:
        return self._shot_info

    @Property("QVariantMap", notify=qChanged)
    def q(self) -> dict[str, Any]:
        return self._q

    @Property("QVariantMap", notify=qChanged)
    def qPreviews(self) -> dict[str, str]:
        return self._q_previews

    @Property(str, notify=codeChanged)
    def code(self) -> str:
        return self._code

    @Property(str, notify=codeChanged)
    def codeFile(self) -> str:
        return self._code_file

    @Property(str, notify=codeChanged)
    def codeTitle(self) -> str:
        return self._code_title

    @Property(str, notify=codeChanged)
    def codeRole(self) -> str:
        return self._code_role

    @Property(str, notify=codeChanged)
    def codeStatus(self) -> str:
        return self._code_status

    @Property(int, notify=codeChanged)
    def codeLines(self) -> int:
        return self._code_lines

    @Property(str, notify=logChanged)
    def log(self) -> str:
        return self._log

    @Property(bool, notify=runningChanged)
    def running(self) -> bool:
        return self._running

    @Property(float, notify=progressChanged)
    def progress(self) -> float:
        return self._progress

    @Property(str, notify=runStateChanged)
    def runState(self) -> str:
        return self._run_state

    @Property(str, notify=lastRunChanged)
    def lastRun(self) -> str:
        return self._last_run

    @Property(float, notify=lastRunChanged)
    def elapsed(self) -> float:
        report = self._read_json(self._root / "output" / "reports" / "run_report.json") if self._root else None
        try:
            return float(report.get("elapsed_s", 0.0)) if report else 0.0
        except (TypeError, ValueError):
            return 0.0

    @Property(bool, notify=lastRunChanged)
    def placeholder(self) -> bool:
        report = self._read_json(self._root / "output" / "reports" / "run_report.json") if self._root else None
        return bool(report.get("placeholder", True)) if report else True

    # ── 公共槽 ──────────────────────────────────────────────────
    @Slot()
    def reload(self) -> None:
        """重新读取配置、报告、模块和预览图。"""
        self._version = int(time.time())
        if self._root is not None:
            self._load_config()
            self._load_modules()
            self._load_metrics_and_artifacts()
            self._load_environment()
            self._load_real_data()
            self._load_q()
            self._generate_previews()
            if not self._code:
                self.loadSource("ertm/solver.py")
            if not self._connected_logged:
                self._append_log(
                    f"[UI] 已连接项目: {self.projectName} v{self.projectVersion}"
                )
                self._append_log(f"[UI] 项目目录: {self._root}")
                self._append_log(
                    f"[UI] 网格 {self._cfg.get('grid', {}).get('nx', '—')} × "
                    f"{self._cfg.get('grid', {}).get('nz', '—')} · "
                    f"配置 config/default.json"
                )
                self._append_log("[UI] 预览: 速度模型 / 炮记录 / PP 成像占位已加载")
                self._connected_logged = True
        self.projectChanged.emit()
        self.cfgChanged.emit()
        self.checksChanged.emit()
        self.modulesChanged.emit()
        self.comparisonsChanged.emit()
        self.metricsChanged.emit()
        self.artifactsChanged.emit()
        self.previewsChanged.emit()
        self.environmentChanged.emit()
        self.lastRunChanged.emit()
        self.realDataChanged.emit()
        self.shotInfoChanged.emit()
        self.qChanged.emit()

    @Slot(str)
    def loadSource(self, relative: str) -> None:
        """读取 ERTM 项目中的源码文件并推送到 QML。"""
        self._code_file = relative
        meta = next((m for m in self._modules if m.get("path") == relative), None)
        self._code_title = relative or "尚未选择模块"
        self._code_role = str(meta.get("role", "")) if meta else ""
        self._code_status = str(meta.get("status", "")) if meta else ""
        root = self._root
        if root is None:
            self._code = "# 未找到 ERTM 项目"
        else:
            path = root / relative
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    text = f"# 读取失败: {exc}"
                self._code = text
            else:
                self._code = (
                    f"# 待算法接入: {relative}\n"
                    "# 该模块由算法分工同学实现后接入本界面。\n"
                    f"# 期望职责: {self._code_role or '核心算法模块'}\n"
                )
        self._code_lines = self._code.count("\n") + 1
        self.codeChanged.emit()
        self.selectionChanged.emit()

    @Slot()
    def runPipeline(self) -> None:
        """运行默认 ERTM 流水线。"""
        self._start_pipeline("config/default.json", "ERTM 流水线")

    @Slot()
    def runQComp(self) -> None:
        """运行含 Q 衰减、补偿与 QC 对比的黏声 RTM 配置。"""
        self._start_pipeline("config/viscoacoustic.json", "Q 补偿")

    def _start_pipeline(self, config_path: str, label: str) -> None:
        if self._root is None:
            self._append_log("[UI] 未找到 ERTM 项目，无法运行。")
            return
        if self._running:
            return

        python = self._root / ".venv" / "Scripts" / "python.exe"
        if not python.is_file():
            python = Path(sys.executable)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.setWorkingDirectory(str(self._root))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self._process.setProcessEnvironment(env)
        self._process.readyRead.connect(self._read_process)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_process_error)

        self._set_running(True)
        self._progress = 0.0
        self.progressChanged.emit()
        self._run_state = f"运行中 · {label}"
        self.runStateChanged.emit()
        self._append_log(f"[UI] 启动 {label}: main.py --config {config_path}")
        self._process.start(
            str(python), ["main.py", "--config", config_path, "--log-level", "INFO"]
        )
        self._ticker.start()

    @Slot()
    def openProject(self) -> None:
        if self._root is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._root)))

    @Slot()
    def openOutput(self) -> None:
        path = self.outputRoot
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    @Slot()
    def openConfig(self) -> None:
        path = self.configPath
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    @Slot()
    def openReport(self) -> None:
        if self._root is not None:
            path = self._root / "output" / "reports" / "run_report.json"
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @Slot()
    def clearLog(self) -> None:
        self._log = ""
        self.logChanged.emit()

    # ── 内部：配置 / 校验 ───────────────────────────────────────
    def _load_config(self) -> None:
        assert self._root is not None
        path = self._root / "config" / "default.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._append_log(f"[UI] 配置读取失败: {exc}")
            data = {}
        self._cfg = data
        self._checks = self._build_checks(data)

    def _build_checks(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        grid = cfg.get("grid", {})
        solver = cfg.get("solver", {})
        model = cfg.get("model", {})
        nx = int(grid.get("nx", 0) or 0)
        nz = int(grid.get("nz", 0) or 0)
        dx = float(grid.get("dx", 1.0) or 1.0)
        dt = float(solver.get("dt", 0.0) or 0.0)
        nt = int(solver.get("nt", 0) or 0)
        f0 = float(cfg.get("source", {}).get("f0", 0.0) or 0.0)

        velocities = [float(model.get("background_vp", 3000.0))]
        for layer in model.get("layers", []) or []:
            velocities.append(float(layer.get("vp", 0.0)))
        anomaly = model.get("anomaly") or {}
        if anomaly.get("vp") is not None:
            velocities.append(float(anomaly["vp"]))
        vmax = max(velocities) if velocities else 0.0
        vmin = min(v for v in velocities if v > 0) if velocities else 0.0
        cfl = (vmax * dt / dx) if dx > 0 else 0.0
        ppw = ((vmin / (2.5 * f0)) / dx) if f0 > 0 and dx > 0 else 0.0

        return [
            {
                "label": "网格规模",
                "value": f"{nx} × {nz}",
                "note": f"{nx * nz:,} 个网格点",
                "state": "ok" if nx > 1 and nz > 1 else "warn",
            },
            {
                "label": "时间步",
                "value": f"{nt} × {dt * 1000:.1f} ms",
                "note": f"{nt * dt:.3f} s 记录长度",
                "state": "ok" if nt > 0 and dt > 0 else "warn",
            },
            {
                "label": "CFL 估算",
                "value": f"{cfl:.2f}",
                "note": "vmax·dt/dx，参考值",
                "state": "ok" if 0 < cfl <= 0.8 else "warn",
            },
            {
                "label": "PPW 估算",
                "value": f"{ppw:.1f}",
                "note": "vmin/(2.5f0)/dx",
                "state": "ok" if ppw >= 4 else "warn",
            },
            {
                "label": "配置校验",
                "value": "通过",
                "note": "config/default.json",
                "state": "ok" if cfg else "warn",
            },
        ]

    # ── 内部：模块 ──────────────────────────────────────────────
    def _load_modules(self) -> None:
        assert self._root is not None
        known: dict[str, tuple[str, str]] = {
            "config.py": ("配置契约", "JSON 配置加载、范围校验与快照输出"),
            "types.py": ("数据契约", "GridSpec / ElasticModel / ShotRecord / ImageResult"),
            "interfaces.py": ("模块接口", "ModelBuilder / ForwardSolver / Imager / DataStore"),
            "exceptions.py": ("异常体系", "统一 ERTM 异常类型"),
            "logging_utils.py": ("日志", "统一日志初始化与文件轮转"),
            "models.py": ("地质模型", "层状背景 + 椭圆异常体（占位几何）"),
            "solver.py": ("正演求解器", "解耦一阶速度—应力方程（当前为占位直达波）"),
            "imaging.py": ("成像条件", "PP/PS 互相关、归一化、Laplacian（占位）"),
            "data.py": ("数据 I/O", "npz / JSON 读写与输出目录管理"),
            "visualization.py": ("QC 可视化", "模型、炮集、成像、指标静态图"),
            "evaluation.py": ("评价指标", "统计指标与图像误差对比"),
            "pipeline.py": ("流水线", "配置 → 模型 → 正演 → 成像 → 评价"),
            "diagnostics.py": ("结构自检", "文件、导入、配置与接口契约检查"),
        }
        modules: list[dict[str, Any]] = []
        for path in sorted((self._root / "ertm").glob("*.py")):
            if path.name == "__init__.py":
                continue
            role, summary = known.get(path.name, ("模块", ""))
            try:
                lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                lines = 0
            status = "ready"
            if path.name in {"solver.py", "imaging.py"}:
                status = "placeholder"
            modules.append(
                {
                    "name": path.name,
                    "path": f"ertm/{path.name}",
                    "role": role,
                    "summary": summary,
                    "lines": lines,
                    "status": status,
                }
            )
        planned = [
            ("staggered_grid.py", "ertm/solver/staggered_grid.py", "差分算子", "八阶交错网格速度—应力更新"),
            ("adjoint.py", "ertm/migration/adjoint.py", "反传延拓", "离散转置伴随与检查点"),
            ("separation.py", "ertm/migration/separation.py", "P/S 分离", "散度/旋度波场分离"),
            ("qcomp.py", "ertm/migration/qcomp.py", "Q 补偿", "GSLS 伴随增益、限幅与低通"),
        ]
        for name, rel, role, summary in planned:
            modules.append(
                {
                    "name": name,
                    "path": rel,
                    "role": role,
                    "summary": summary,
                    "lines": 0,
                    "status": "planned",
                }
            )
        self._modules = modules

    # ── 内部：指标 / 产物 / 环境 ────────────────────────────────
    def _load_metrics_and_artifacts(self) -> None:
        assert self._root is not None
        metrics_path = self._root / "output" / "reports" / "metrics.json"
        report_path = self._root / "output" / "reports" / "run_report.json"
        metrics = self._read_json(metrics_path)
        report = self._read_json(report_path)

        labels = {
            "model_vp_min": ("Vp 最小", "m/s"),
            "model_vp_mean": ("Vp 平均", "m/s"),
            "model_vp_max": ("Vp 最大", "m/s"),
            "model_vs_vp_ratio_mean": ("Vs/Vp 均值", ""),
            "shot_abs_max": ("炮记录峰值", ""),
            "shot_rms": ("炮记录 RMS", ""),
            "image_min": ("成像最小值", ""),
            "image_max": ("成像最大值", ""),
            "image_mean": ("成像均值", ""),
            "image_std": ("成像标准差", ""),
            "image_nonzero_fraction": ("非零像素占比", ""),
        }
        values = metrics.get("values", {}) if metrics else {}
        rows: list[dict[str, Any]] = []
        for key, (label, unit) in labels.items():
            if key not in values:
                continue
            value = float(values[key])
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "value": self._format_number(value),
                    "raw": value,
                    "unit": unit,
                    "state": "muted" if key.startswith("image_") else "ok",
                }
            )
        self._metrics = rows
        self._shot_info = self._build_shot_info(metrics, report)

        artifacts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in (report.get("artifacts", []) if report else []):
            path = Path(str(raw))
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))
            artifacts.append(self._artifact_entry(path, "output"))
        for pattern, kind in (
            ("output/images/*.png", "image"),
            ("output/shots/*.npz", "shot"),
            ("output/models/*.npz", "model"),
            ("output/reports/*.json", "report"),
        ):
            for path in sorted(self._root.glob(pattern)):
                if str(path) in seen:
                    continue
                seen.add(str(path))
                artifacts.append(self._artifact_entry(path, kind))
        self._artifacts = artifacts[:24]

        if report_path.is_file():
            stamp = datetime.fromtimestamp(report_path.stat().st_mtime)
            self._last_run = stamp.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_shot_info(self, metrics: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
        """汇总单炮记录的采集参数与占位正演元数据，供界面解释卡片使用。"""
        cfg = self._cfg or {}
        grid = cfg.get("grid", {}) or {}
        source = cfg.get("source", {}) or {}
        receivers = cfg.get("receivers", {}) or {}
        solver = cfg.get("solver", {}) or {}

        nx = int(self._as_float(grid.get("nx"), 0))
        dx = self._as_float(grid.get("dx"), 0.0)
        x0 = self._as_float(grid.get("x0"), 0.0)
        xmax = x0 + max(0, nx - 1) * dx
        nrec = int(self._as_float(receivers.get("count"), 0))
        nt = int(self._as_float(solver.get("nt"), 0))
        dt = self._as_float(solver.get("dt"), 0.0)
        shot_meta = report.get("shot_meta", {}) if report else {}
        if not isinstance(shot_meta, dict):
            shot_meta = {}
        values = metrics.get("values", {}) if metrics else {}
        if not isinstance(values, dict):
            values = {}
        available = False
        if self._root is not None:
            available = (self._root / "output" / "shots" / "shot.npz").is_file()

        return {
            "available": available,
            "placeholder": bool(shot_meta.get("placeholder", True)),
            "method": str(shot_meta.get("method", "—")),
            "vmean": self._as_float(shot_meta.get("v_mean"), 0.0),
            "nrec": nrec,
            "nt": nt,
            "dt": dt,
            "tmax": nt * dt,
            "f0": self._as_float(source.get("f0"), 0.0),
            "delay": self._as_float(source.get("delay"), 0.0),
            "source_kind": str(source.get("kind", "—")),
            "component": str(receivers.get("component", "—")),
            "sx": self._as_float(source.get("x"), 0.0),
            "sz": self._as_float(source.get("z"), 0.0),
            "rx0": self._as_float(receivers.get("x_start"), 0.0),
            "rx1": self._as_float(receivers.get("x_end"), 0.0),
            "rz": self._as_float(receivers.get("z"), 0.0),
            "x0": x0,
            "xmax": xmax,
            "rms": self._as_float(values.get("shot_rms"), 0.0),
            "absmax": self._as_float(values.get("shot_abs_max"), 0.0),
        }

    @staticmethod
    def _artifact_entry(path: Path, kind: str) -> dict[str, Any]:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return {
            "name": path.name,
            "path": str(path),
            "kind": kind,
            "size": _format_size(size),
            "exists": path.is_file(),
        }

    def _load_environment(self) -> None:
        assert self._root is not None
        self._environment = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(terse=True),
            "root": str(self._root),
            "venv": str(self._root / ".venv" / "Scripts" / "python.exe"),
        }

    def _load_real_data(self) -> None:
        assert self._root is not None
        data_dir = self._root / "data"
        files = []
        if data_dir.is_dir():
            for path in data_dir.rglob("*"):
                if path.is_file() and path.name != ".gitkeep":
                    files.append(path)
        self._real_data = {
            "available": bool(files),
            "count": len(files),
            "text": f"已发现 {len(files)} 个数据文件" if files else "实测资料未导入",
            "dir": str(data_dir),
        }

    # ── 内部：Q 补偿数据 ────────────────────────────────────────
    def _load_q(self) -> None:
        assert self._root is not None
        qcfg = self._read_json(self._root / "config" / "viscoacoustic.json").get("q", {})
        report = self._read_json(
            self._root / "output" / "viscoacoustic" / "reports" / "run_report.json"
        )
        q_report_path = self._root / "output" / "viscoacoustic" / "reports" / "run_report.json"
        q_last_run = "—"
        if q_report_path.is_file():
            q_last_run = datetime.fromtimestamp(q_report_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        q_params = report.get("q", {}) if isinstance(report, dict) else {}
        image_meta = report.get("image_meta", {}) if isinstance(report, dict) else {}
        q_meta = image_meta.get("q", {}) if isinstance(image_meta, dict) else {}
        comparison = report.get("q_comparison", {}) if isinstance(report, dict) else {}
        if not comparison:
            comparison = self._read_json(
                self._root / "output" / "viscoacoustic" / "images" / "q_compare.json"
            )
        q_model = self._load_npz(
            self._root / "output" / "viscoacoustic" / "models" / "q_model.npz"
        )
        qp = q_model.get("qp")
        qs = q_model.get("qs")

        previews: dict[str, str] = {}
        if qp is not None:
            previews["qp"] = self._save_preview("q_qp", self._render_model(qp))
        if qs is not None:
            previews["qs"] = self._save_preview("q_qs", self._render_model(qs))
        image = self._load_npz(
            self._root / "output" / "viscoacoustic" / "images" / "image_pp.npz"
        ).get("image")
        if image is not None:
            arr = np.asarray(image, dtype=np.float64)
            peak = float(np.percentile(np.abs(arr), 99.0)) or float(np.nanmax(np.abs(arr))) or 1.0
            previews["image"] = self._save_preview(
                "q_image", _qimage(_seismic_rgb(arr, peak), (640, 460))
            )
        compare_png = self._root / "output" / "viscoacoustic" / "images" / "q_compare.png"
        if compare_png.is_file():
            previews["compare"] = compare_png.as_uri()

        alphas = q_meta.get("alpha_mechanisms", q_params.get("alpha_mechanisms", []))
        if not isinstance(alphas, list):
            alphas = []
        self._q = {
            "available": bool(qp is not None or qs is not None),
            "enabled": bool(q_params.get("enabled", qcfg.get("enabled", False))),
            "model": str(q_params.get("model", qcfg.get("model", "gsls"))),
            "alpha_c": self._as_float(q_params.get("alpha_c", qcfg.get("alpha_c")), 0.0),
            "f_ref": self._as_float(q_params.get("f_ref", qcfg.get("f_ref")), 30.0),
            "gain_cap": self._as_float(q_params.get("gain_cap", qcfg.get("gain_cap")), 30.0),
            "n_mechanisms": int(self._as_float(q_params.get("n_mechanisms", qcfg.get("n_mechanisms")), 3)),
            "f_max": self._as_float(q_params.get("f_max", qcfg.get("f_max")), 60.0),
            "gain_budget_db": self._as_float(
                q_params.get("gain_budget_db", qcfg.get("gain_budget_db")), 40.0
            ),
            "compare": bool(q_params.get("compare", qcfg.get("compare", False))),
            "q_fit_error": self._as_float(q_meta.get("q_fit_error"), 0.0),
            "clipped_steps": int(self._as_float(q_meta.get("clipped_steps"), 0)),
            "lowpass_steps": int(self._as_float(q_meta.get("lowpass_steps"), 0)),
            "alpha_mechanisms": [self._as_float(v) for v in alphas],
            "band_gain": self._as_float(comparison.get("band_amplitude_gain"), 0.0),
            "band_top": self._as_float(comparison.get("band_top_m"), 0.0),
            "band_bottom": self._as_float(comparison.get("band_bottom_m"), 0.0),
            "baseline_rms": self._as_float(comparison.get("baseline_band_rms"), 0.0),
            "compensated_rms": self._as_float(comparison.get("compensated_band_rms"), 0.0),
            "baseline_ratio": self._as_float(comparison.get("baseline_deep_shallow_ratio"), 0.0),
            "compensated_ratio": self._as_float(comparison.get("compensated_deep_shallow_ratio"), 0.0),
            "difference_rms": self._as_float(comparison.get("difference_rms"), 0.0),
            "correlation": self._as_float(comparison.get("correlation"), 0.0),
            "last_run": q_last_run,
            "elapsed": self._as_float(report.get("elapsed_s"), 0.0),
            "qp_min": float(np.nanmin(qp)) if qp is not None else 0.0,
            "qp_max": float(np.nanmax(qp)) if qp is not None else 0.0,
            "qs_min": float(np.nanmin(qs)) if qs is not None else 0.0,
            "qs_max": float(np.nanmax(qs)) if qs is not None else 0.0,
        }
        self._q_previews = previews

    # ── 内部：预览图生成 ────────────────────────────────────────
    def _generate_previews(self) -> None:
        assert self._root is not None
        previews: dict[str, str] = {}
        model_path = self._root / "output" / "models" / "model.npz"
        shot_path = self._root / "output" / "shots" / "shot.npz"
        image_path = self._root / "output" / "images" / "image_pp.npz"

        vp = vs = rho = None
        model_data = self._load_npz(model_path)
        if model_data:
            vp = model_data.get("vp")
            vs = model_data.get("vs")
            rho = model_data.get("rho")
        if vp is None:
            vp = self._build_preset("cave")
            vs = None
            rho = None

        if vp is not None:
            previews["modelVp"] = self._save_preview("model_vp", self._render_model(vp))
            if vs is not None:
                previews["modelVs"] = self._save_preview("model_vs", self._render_model(vs))
            if rho is not None:
                previews["modelRho"] = self._save_preview("model_rho", self._render_model(rho))

        shot_data = self._load_npz(shot_path)
        shot = shot_data.get("data") if shot_data else None
        if shot is not None and np.size(shot) > 0:
            previews["shot"] = self._save_preview(
                "shot", _qimage(_seismic_rgb(np.asarray(shot)), (640, 360), keep_aspect=False)
            )
        image_data = self._load_npz(image_path)
        image = image_data.get("image") if image_data else None
        if image is not None:
            previews["image"] = self._save_preview("image", self._render_image(image))

        for key, preset, label in (
            ("cmpCave", "cave", "溶洞"),
            ("cmpFracture", "fracture", "倾斜裂缝"),
            ("cmpCross", "cross", "交叉裂缝"),
        ):
            field = self._build_preset(preset)
            previews[key] = self._save_preview(f"cmp_{preset}", self._render_model(field))

        self._previews = previews
        self._comparisons = [
            {
                "id": "cave",
                "name": "溶洞模型",
                "subtitle": "Cave · 层状背景 + 椭圆溶洞",
                "modelKey": "cmpCave",
                "note": "Vp 1800 m/s · 40 × 30 m",
                "status": "模型就绪",
            },
            {
                "id": "fracture",
                "name": "倾斜裂缝",
                "subtitle": "Fracture · 55° 倾斜缝",
                "modelKey": "cmpFracture",
                "note": "半长 150 m · 半宽 12 m",
                "status": "模型预设",
            },
            {
                "id": "cross",
                "name": "交叉裂缝",
                "subtitle": "Cross fracture · ±45° 交叉",
                "modelKey": "cmpCross",
                "note": "两组裂缝交叉于 500 m",
                "status": "模型预设",
            },
        ]

    def _render_model(self, data: np.ndarray) -> QImage:
        norm = _normalize(data)
        rgb = _colormap(norm, MODEL_STOPS)
        return _qimage(rgb, (640, 460))

    def _render_image(self, data: np.ndarray) -> QImage:
        arr = np.asarray(data, dtype=np.float64)
        if not np.any(np.abs(arr) > 1e-12):
            rgb = np.zeros(arr.shape + (3,), dtype=np.uint8)
            rgb[..., 0] = 6
            rgb[..., 1] = 20
            rgb[..., 2] = 34
            return _qimage(rgb, (640, 460))
        return _qimage(_seismic_rgb(arr), (640, 460))

    def _save_preview(self, key: str, image: QImage) -> str:
        path = self._cache_dir / f"{key}_{self._version}.png"
        image.save(str(path), "PNG")
        for old in self._cache_dir.glob(f"{key}_*.png"):
            if old != path:
                try:
                    old.unlink()
                except OSError:
                    pass
        return path.as_uri()

    def _build_preset(self, kind: str) -> np.ndarray:
        """按当前配置构建 UI 模型预设（仅用于界面示意）。"""
        cfg = self._cfg or {}
        grid = cfg.get("grid", {})
        nx = int(grid.get("nx", 201))
        nz = int(grid.get("nz", 151))
        dx = float(grid.get("dx", 5.0))
        dz = float(grid.get("dz", 5.0))
        x0 = float(grid.get("x0", 0.0))
        z0 = float(grid.get("z0", 0.0))
        model = cfg.get("model", {})
        vp = np.full((nz, nx), float(model.get("background_vp", 3000.0)), dtype=np.float64)
        z = z0 + np.arange(nz, dtype=float) * dz
        for layer in model.get("layers", []) or []:
            mask = z >= float(layer.get("z_top", 0.0))
            vp[mask, :] = float(layer.get("vp", 3000.0))
        x = x0 + np.arange(nx, dtype=float) * dx
        xx, zz = np.meshgrid(x, z)

        if kind == "cave":
            anomaly = model.get("anomaly") or {}
            cx = float(anomaly.get("x", 500.0))
            cz = float(anomaly.get("z", 350.0))
            rx = float(anomaly.get("rx", 40.0))
            rz = float(anomaly.get("rz", 30.0))
            value = float(anomaly.get("vp", 1800.0) or 1800.0)
            vp[((xx - cx) / max(rx, 1.0)) ** 2 + ((zz - cz) / max(rz, 1.0)) ** 2 <= 1.0] = value
        elif kind == "fracture":
            mask = self._rotated_ellipse(xx, zz, 500.0, 390.0, 150.0, 12.0, 55.0)
            vp[mask] = 2000.0
        elif kind == "cross":
            mask = self._rotated_ellipse(xx, zz, 500.0, 400.0, 130.0, 9.0, 42.0)
            mask |= self._rotated_ellipse(xx, zz, 500.0, 400.0, 130.0, 9.0, -42.0)
            vp[mask] = 2000.0
        return vp

    @staticmethod
    def _rotated_ellipse(
        xx: np.ndarray,
        zz: np.ndarray,
        cx: float,
        cz: float,
        half_length: float,
        half_width: float,
        angle_deg: float,
    ) -> np.ndarray:
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = (xx - cx) * cos_a + (zz - cz) * sin_a
        zr = -(xx - cx) * sin_a + (zz - cz) * cos_a
        return (xr / half_length) ** 2 + (zr / half_width) ** 2 <= 1.0

    @staticmethod
    def _load_npz(path: Path) -> dict[str, np.ndarray]:
        if not path.is_file():
            return {}
        try:
            with np.load(path, allow_pickle=False) as data:
                return {key: np.asarray(data[key]) for key in data.files if key != "meta"}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ── 内部：运行进程与日志 ────────────────────────────────────
    def _read_process(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAll()).decode("utf-8", "replace")
        if chunk:
            lines = chunk.splitlines()
            for line in lines:
                if line.strip():
                    self._append_log(line.rstrip())
        # 简单推进度条，真实进度由算法侧以后提供
        self._progress = min(0.9, self._progress + 0.02)
        self.progressChanged.emit()

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._ticker.stop()
        self._set_running(False)
        ok = exit_code == 0
        self._run_state = "完成" if ok else f"失败 ({exit_code})"
        self._progress = 1.0
        self._last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.progressChanged.emit()
        self.runStateChanged.emit()
        self.lastRunChanged.emit()
        self._append_log(f"[UI] 流水线结束，退出码 {exit_code}。")
        if ok:
            QTimer.singleShot(350, self.reload)
        self.lastRunChanged.emit()

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        self._ticker.stop()
        self._set_running(False)
        self._run_state = "启动失败"
        self.runStateChanged.emit()
        self._append_log(f"[UI] 进程错误: {error}")

    def _tick(self) -> None:
        if not self._running:
            return
        if self._progress < 0.9:
            self._progress = min(0.9, self._progress + 0.015)
            self.progressChanged.emit()

    def _set_running(self, value: bool) -> None:
        if self._running == value:
            return
        self._running = value
        self.runningChanged.emit()

    def _append_log(self, line: str) -> None:
        self._log += line.rstrip() + "\n"
        if len(self._log) > 24000:
            self._log = self._log[-18000:]
        self.logChanged.emit()

    # ── 格式化 ──────────────────────────────────────────────────
    @staticmethod
    def _format_number(value: float) -> str:
        if value == 0:
            return "0"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4g}"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"
