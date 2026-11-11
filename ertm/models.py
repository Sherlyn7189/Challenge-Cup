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

from .config import AnomalyConfig, AppConfig, ModelConfig
from .exceptions import ModelError
from .logging_utils import get_logger
from .types import ElasticModel, GridSpec

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


def build_model(config: AppConfig) -> ElasticModel:
    """模块级便捷函数，等价于 ``LayeredModelBuilder().build(config)``。"""
    return LayeredModelBuilder().build(config)
