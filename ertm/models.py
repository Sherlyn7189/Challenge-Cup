"""地质模型构建模块。

首期职责
--------
- 把 :class:`~ertm.config.ModelConfig` 转换为 :class:`~ertm.types.ElasticModel`；
- 支持水平层状背景 + 一个椭圆形异常体；
- 只做参数赋值与几何占位，不做 Q 拟合、不做各向异性、不做复杂岩石物理。

设计说明
--------
异常体的三类 ``kind``（cave/fracture/fault）目前共用椭圆几何，这是**占位
实现**。正式的溶洞/裂缝带/断层几何、抗混叠和体积分数平均在后续任务中
实现，届时只需替换 ``_apply_anomaly``，不影响其他模块。
"""

from __future__ import annotations

import numpy as np

from .config import PROJECT_ROOT, AnomalyConfig, AppConfig, ModelConfig
from .exceptions import ModelError
from .logging_utils import get_logger
from .types import ElasticModel, GridSpec
import json
from pathlib import Path

logger = get_logger(__name__)


class LayeredModelBuilder:
    """把配置转换为二维层状模型（含一个椭圆异常体）。"""

    name = "layered-model-builder"

    def build(self, config: AppConfig) -> ElasticModel:
        """按配置构建模型并校验。"""
        grid = GridSpec(
            nx=config.grid.nx,
            nz=config.grid.nz,
            dx=config.grid.dx,
            dz=config.grid.dz,
            x0=config.grid.x0,
            z0=config.grid.z0,
        )
        model_cfg = config.model
        shape = grid.shape

        vp = np.full(shape, model_cfg.background_vp, dtype=np.float32)
        vs = np.full(shape, model_cfg.background_vs, dtype=np.float32)
        rho = np.full(shape, model_cfg.background_rho, dtype=np.float32)
        qp = np.full(shape, model_cfg.background_qp, dtype=np.float32)
        qs = np.full(shape, model_cfg.background_qs, dtype=np.float32)

        z = grid.z0 + np.arange(grid.nz, dtype=np.float64) * grid.dz
        for layer in model_cfg.layers:
            mask = z >= layer.z_top
            vp[mask, :] = np.float32(layer.vp)
            vs[mask, :] = np.float32(layer.vs)
            rho[mask, :] = np.float32(layer.rho)
            qp[mask, :] = np.float32(layer.qp)
            qs[mask, :] = np.float32(layer.qs)

        meta = {
            "builder": self.name,
            "placeholder": False,
            "layer_count": len(model_cfg.layers),
            "anomaly_kind": None if model_cfg.anomaly is None else model_cfg.anomaly.kind,
            "anomaly_dip": None if model_cfg.anomaly is None else float(model_cfg.anomaly.dip),
        }

        if model_cfg.anomaly is not None:
            self._apply_anomaly(vp, vs, rho, qp, qs, grid, model_cfg)

        model = ElasticModel(
            grid=grid,
            vp=vp,
            vs=vs,
            rho=rho,
            qp=qp,
            qs=qs,
            name=config.project.name,
            meta=meta,
        )
        try:
            model.validate()
        except ValueError as exc:
            raise ModelError(f"模型校验失败: {exc}") from exc

        logger.info(
            "模型构建完成: %s, 网格=%dx%d, 异常体=%s",
            model.name,
            grid.nx,
            grid.nz,
            meta["anomaly_kind"],
        )
        return model

    @staticmethod
    def _anomaly_mask(
        grid: GridSpec,
        anomaly: "AnomalyConfig",
    ) -> np.ndarray:
        """按 ``kind`` 生成异常体掩膜（真实几何，不再是统一椭圆）。

        - ``cave``：椭圆溶洞；
        - ``fracture``：按 ``dip`` 旋转的窄带（裂缝带），长轴 ``rx``、厚度 ``rz``；
        - ``fault``：按 ``dip`` 倾斜的断层带，带宽 ``2·rz``，两侧错断由 ``rx`` 控制。
        """
        x = grid.x0 + np.arange(grid.nx, dtype=np.float64) * grid.dx
        z = grid.z0 + np.arange(grid.nz, dtype=np.float64) * grid.dz
        xx, zz = np.meshgrid(x, z)
        dx = xx - float(anomaly.x)
        dz = zz - float(anomaly.z)
        angle = np.deg2rad(float(anomaly.dip))
        # 旋转到异常体局部坐标（u 沿走向，v 垂直走向）
        u = np.cos(angle) * dx + np.sin(angle) * dz
        v = -np.sin(angle) * dx + np.cos(angle) * dz
        if anomaly.kind == "cave":
            return (u / anomaly.rx) ** 2 + (v / anomaly.rz) ** 2 <= 1.0
        if anomaly.kind == "fracture":
            return (np.abs(u) <= anomaly.rx) & (np.abs(v) <= anomaly.rz)
        return np.abs(v) <= anomaly.rz

    @staticmethod
    def _apply_anomaly(
        vp: np.ndarray,
        vs: np.ndarray,
        rho: np.ndarray,
        qp: np.ndarray,
        qs: np.ndarray,
        grid: GridSpec,
        model_cfg: ModelConfig,
    ) -> None:
        """按异常体类型修改参数（未显式给出的量按局部背景比例缩放）。"""
        anomaly = model_cfg.anomaly
        if anomaly is None:
            return

        mask = LayeredModelBuilder._anomaly_mask(grid, anomaly)
        if not np.any(mask):
            logger.warning(
                "异常体未落在任何网格内: kind=%s, x=%.1f, z=%.1f, rx=%.1f, rz=%.1f",
                anomaly.kind,
                anomaly.x,
                anomaly.z,
                anomaly.rx,
                anomaly.rz,
            )
            return

        vp[mask] = np.float32(anomaly.vp if anomaly.vp is not None else vp[mask] * 0.60)
        vs[mask] = np.float32(anomaly.vs if anomaly.vs is not None else vs[mask] * 0.60)
        rho[mask] = np.float32(anomaly.rho if anomaly.rho is not None else rho[mask] * 0.85)
        qp[mask] = np.float32(anomaly.qp if anomaly.qp is not None else qp[mask] * 0.50)
        qs[mask] = np.float32(anomaly.qs if anomaly.qs is not None else qs[mask] * 0.50)


# ----------------------------------------------------------------------------
# 文件模型：从 npz / SEG-Y 读取速度模型
# ----------------------------------------------------------------------------
class FileModelBuilder:
    """从文件构建速度模型（真实资料项目的主用途径）。

    支持的格式：

    - ``.npz``：键 ``vp``（必需）、``vs/rho/qp/qs``（可选）、``dx/dz/x0/z0``
      或 ``meta``（与 :meth:`NumpyDataStore.save_model` 写出的格式兼容）；
    - ``.sgy/.segy``：按 :func:`ertm.segy.read_segy_image` 读取，道 = x、
      样点 = 深度，``Interval`` 解释为深度采样（米）。

    缺失的 ``vs/rho/Q`` 按实用经验关系补齐（见 :class:`~ertm.config.ModelConfig`），
    并在 ``meta`` 里记录补齐方式，避免"看起来是实测、其实是假设"。
    """

    name = "file-model-builder"

    def build(self, config: AppConfig) -> ElasticModel:
        cfg = config.model
        path = Path(cfg.file).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise ModelError(f"速度模型文件不存在: {path}")

        arrays, dx, dz, x0, z0 = self._load(path, config)
        vp = np.asarray(arrays["vp"], dtype=np.float32)
        if vp.ndim != 2:
            raise ModelError(f"速度模型必须是二维数组，当前 {vp.shape}")
        shape = vp.shape
        if not np.all(np.isfinite(vp)) or np.any(vp <= 0):
            raise ModelError("速度模型含 NaN/Inf 或非正值")

        ratio = float(cfg.vs_ratio) if cfg.vs_ratio is not None else float(
            cfg.background_vs / cfg.background_vp
        )
        filled: list[str] = []
        vs = arrays.get("vs")
        if vs is None:
            vs = vp * ratio
            filled.append("vs=vp*%.3f" % ratio)
        rho = arrays.get("rho")
        if rho is None:
            rho = 310.0 * np.power(vp, 0.25) if cfg.rho_from_gardner else np.full(
                shape, cfg.background_rho
            )
            filled.append("rho=Gardner" if cfg.rho_from_gardner else "rho=常数")
        qp = arrays.get("qp")
        if qp is None:
            qp = np.full(shape, cfg.background_qp)
            filled.append("qp=常数")
        qs = arrays.get("qs")
        if qs is None:
            qs = np.full(shape, cfg.background_qs)
            filled.append("qs=常数")

        grid = GridSpec(
            nx=int(shape[1]),
            nz=int(shape[0]),
            dx=float(dx),
            dz=float(dz),
            x0=float(x0),
            z0=float(z0),
        )
        model = ElasticModel(
            grid=grid,
            vp=vp,
            vs=np.asarray(vs, dtype=np.float32),
            rho=np.asarray(rho, dtype=np.float32),
            qp=np.asarray(qp, dtype=np.float32),
            qs=np.asarray(qs, dtype=np.float32),
            name=config.project.name,
            meta={
                "builder": self.name,
                "placeholder": False,
                "file": str(path),
                "filled": filled,
            },
        )
        try:
            model.validate()
        except ValueError as exc:
            raise ModelError(f"文件模型校验失败: {exc}") from exc
        logger.info(
            "文件模型加载完成: %s, 网格=%dx%d, dx=%.2f, dz=%.2f, 补齐=%s",
            path.name,
            grid.nx,
            grid.nz,
            grid.dx,
            grid.dz,
            filled or "无",
        )
        return model

    @staticmethod
    def _load(
        path: Path,
        config: AppConfig,
    ) -> tuple[dict[str, np.ndarray], float, float, float, float]:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            try:
                with np.load(path, allow_pickle=False) as data:
                    arrays = {
                        key: np.asarray(data[key])
                        for key in ("vp", "vs", "rho", "qp", "qs")
                        if key in data
                    }
                    dx = float(data["dx"]) if "dx" in data else float(config.grid.dx)
                    dz = float(data["dz"]) if "dz" in data else float(config.grid.dz)
                    x0 = float(data["x0"]) if "x0" in data else float(config.grid.x0)
                    z0 = float(data["z0"]) if "z0" in data else float(config.grid.z0)
                    if "meta" in data:
                        try:
                            meta = json.loads(str(np.asarray(data["meta"]).item()))
                            grid_meta = meta.get("grid", {})
                            dx = float(grid_meta.get("dx", dx))
                            dz = float(grid_meta.get("dz", dz))
                            x0 = float(grid_meta.get("x0", x0))
                            z0 = float(grid_meta.get("z0", z0))
                        except (ValueError, TypeError):
                            pass
            except (OSError, KeyError, ValueError) as exc:
                raise ModelError(f"读取 npz 速度模型失败: {path}: {exc}") from exc
            if "vp" not in arrays:
                raise ModelError(f"npz 速度模型缺少 'vp' 键: {path}")
            return arrays, dx, dz, x0, z0

        if suffix in {".sgy", ".segy"}:
            from .segy import read_segy_image

            try:
                vp, dz, x = read_segy_image(path)
            except Exception as exc:
                raise ModelError(f"读取 SEG-Y 速度模型失败: {path}: {exc}") from exc
            dx = float(np.median(np.diff(x))) if x.size > 1 else float(config.grid.dx)
            return {"vp": vp}, dx, dz, float(x[0]) if x.size else 0.0, 0.0

        raise ModelError(
            f"不支持的速度模型格式 {suffix!r}（支持 .npz / .sgy / .segy）: {path}"
        )


def build_model_builder(config: AppConfig):
    """按 ``model.source`` 选择模型构建器。"""
    if config.model.source == "file":
        return FileModelBuilder()
    return LayeredModelBuilder()


def build_model(config: AppConfig) -> ElasticModel:
    """模块级便捷函数：按配置选择层状或文件模型。"""
    return build_model_builder(config).build(config)
