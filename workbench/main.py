# This Python file uses the following encoding: utf-8
"""ERTM 求解器工作台入口。

界面使用 Qt Quick (QML)，数据来自同机的 SeismicProjectERTM 项目：
配置、模型、炮记录、成像产物和源码都通过 ``ErtmBridge`` 读取。
"""

import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from ertm_bridge import ErtmBridge


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    app.setApplicationName("ERTM Workbench")
    app.setOrganizationName("SeismicProjectERTM")

    bridge = ErtmBridge()
    engine = QQmlApplicationEngine()
    engine.addImportPath(Path(__file__).parent)
    engine.rootContext().setContextProperty("bridge", bridge)
    engine.loadFromModule("untitled", "Main")
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
