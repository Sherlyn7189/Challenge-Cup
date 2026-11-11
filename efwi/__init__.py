"""EFWI —— 二维弹性波全波形反演 / 逆时偏移 / 测井尺度建模。

模块划分（与项目文档 2.2.6 的对应关系）::

    efwi.config        网格、观测系统、多尺度与约束配置
    efwi.model         弹性参数模型 (vp, vs, rho[, epsilon, delta, gamma])
    efwi.fd           交错网格弹性波正演与伴随算子（含 VTI/HTI）
    efwi.adjoint      残差/目标函数/伴随源/逐炮梯度累积/差分检验
    efwi.gradient     伴随态梯度 -> (vp, vs, rho) 链式转换
    efwi.pseudo_hessian  伪 Hessian 对角照明补偿
    efwi.optimize     L-BFGS + 抛物线线搜索
    efwi.constraints  Dykstra / ADMM 凸集投影（地质与岩石物理约束）
    efwi.inversion    多尺度 EFWI 主循环
    efwi.imaging      RTM 保幅成像、反射点与倾角提取
    efwi.welllog      LAS 测井解析、初始模型构建、尺度换算
    efwi.workflow     “反演—成像—地质建模”闭环
"""

from .config import ConstraintConfig, FWIConfig, GridConfig, StageConfig
from .model import ElasticModel

__all__ = [
    "ElasticModel",
    "GridConfig",
    "StageConfig",
    "FWIConfig",
    "ConstraintConfig",
]

__version__ = "1.0.0"
