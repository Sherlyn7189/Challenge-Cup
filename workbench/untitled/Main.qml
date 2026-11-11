import QtQuick
import QtQuick.Window
import "theme/Theme.js" as Theme
import "components"

Window {
    id: window

    width: 1280
    height: 720
    minimumWidth: 1280
    minimumHeight: 720
    visible: true
    title: "ERTM 弹性波逆时偏移求解器 · SeismicProjectERTM"
    color: Theme.bg0

    property int selectedIndex: 0
    property bool entered: false

    readonly property var sections: [
        {
            "eyebrow": "SECTION 01 · CONFIG",
            "title": "参数配置",
            "desc": "读取 config/default.json，核对网格、层状模型、观测系统与求解参数；预览来自项目真实模型与炮记录。",
            "formula": "nx × nz = 201 × 151 · dt = 1 ms · nt = 600",
            "tag": "已加载",
            "tagColor": "#31d98a"
        },
        {
            "eyebrow": "SECTION 02 · IMAGING",
            "title": "典型模型对比",
            "desc": "对比溶洞、倾斜裂缝、交叉裂缝三类典型模型的成像剖面，检查异常体归位与横向分辨率。",
            "formula": "I(x,z) = Σs ∫ Ws(x,z,t) · Wr(x,z,t) dt",
            "tag": "待接入",
            "tagColor": "#efa63f"
        },
        {
            "eyebrow": "SECTION 03 · REAL DATA",
            "title": "实测资料对比",
            "desc": "RTM 与 Kirchhoff / 高斯束偏移的实测资料对比；当前 data/ 目录尚未导入 SEG-Y。",
            "formula": "RTM  vs  Kirchhoff  vs  Gaussian Beam",
            "tag": "未导入",
            "tagColor": "#efa63f"
        },
        {
            "eyebrow": "SECTION 04 · Q COMP",
            "title": "Q 补偿",
            "desc": "GSLS 黏声引擎：接收侧补偿、增益上限与时变低通，输出补偿前后 QC 对比、Q 模型与增益曲线；可直接运行 viscoacoustic 配置。",
            "formula": "G(t) = exp(π f t / Q) ≤ Gcap",
            "tag": "已接入",
            "tagColor": "#31d98a"
        },
        {
            "eyebrow": "SECTION 05 · SOURCE",
            "title": "求解器源码",
            "desc": "浏览 ertm/*.py 的职责、行数与实现状态；可直接查看配置、求解器、成像与流水线源码。",
            "formula": "build_solver(config) → ForwardSolver",
            "tag": "可查看",
            "tagColor": "#38c9f5"
        }
    ]

    // ── 背景 ────────────────────────────────────────────────────
    Canvas {
        anchors.fill: parent

        onPaint: {
            var ctx = getContext("2d");
            ctx.clearRect(0, 0, width, height);

            var bg = ctx.createLinearGradient(0, 0, width * 0.65, height);
            bg.addColorStop(0.0, "#071b2c");
            bg.addColorStop(0.45, "#04101d");
            bg.addColorStop(1.0, "#020810");
            ctx.fillStyle = bg;
            ctx.fillRect(0, 0, width, height);

            var glow = ctx.createRadialGradient(width * 0.36, height * 0.28, 20,
                                               width * 0.36, height * 0.28, width * 0.64);
            glow.addColorStop(0.0, "rgba(28, 120, 168, 0.15)");
            glow.addColorStop(0.55, "rgba(18, 78, 118, 0.05)");
            glow.addColorStop(1.0, "rgba(0, 0, 0, 0)");
            ctx.fillStyle = glow;
            ctx.fillRect(0, 0, width, height);

            ctx.strokeStyle = "rgba(64, 140, 180, 0.035)";
            ctx.lineWidth = 1;
            for (var x = 24; x < width; x += 48) {
                ctx.beginPath();
                ctx.moveTo(x + 0.5, 0);
                ctx.lineTo(x + 0.5, height);
                ctx.stroke();
            }
            for (var y = 24; y < height; y += 48) {
                ctx.beginPath();
                ctx.moveTo(0, y + 0.5);
                ctx.lineTo(width, y + 0.5);
                ctx.stroke();
            }

            // 底部微弱的波场横纹
            ctx.strokeStyle = "rgba(48, 140, 180, 0.05)";
            for (var k = 0; k < 5; ++k) {
                ctx.beginPath();
                for (var px = 0; px < width; px += 6) {
                    var py = height - 34 - k * 9 + Math.sin(px * 0.012 + k) * 4;
                    if (px === 0)
                        ctx.moveTo(px, py);
                    else
                        ctx.lineTo(px, py);
                }
                ctx.stroke();
            }
        }

        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        Component.onCompleted: requestPaint()
    }

    Item {
        id: root
        anchors.fill: parent
        anchors.margins: 10
        opacity: window.entered ? 1.0 : 0.0
        enabled: window.entered
        Behavior on opacity { NumberAnimation { duration: 350; easing.type: Easing.OutCubic } }

        // ── 左侧导航 ────────────────────────────────────────────
        Sidebar {
            id: sidebar
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 210
            backend: bridge
            selectedIndex: window.selectedIndex
            onSelect: function(index) { window.selectedIndex = index; }
            onOpenProject: if (bridge) bridge.openProject()
            onRefresh: if (bridge) bridge.reload()
        }

        // ── 右侧状态栏 ──────────────────────────────────────────
        RightPanel {
            id: rightPanel
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 290
            backend: bridge
            section: window.sections[window.selectedIndex]
        }

        // ── 中栏 ────────────────────────────────────────────────
        Item {
            id: mainArea
            anchors.left: sidebar.right
            anchors.leftMargin: 10
            anchors.right: rightPanel.left
            anchors.rightMargin: 10
            anchors.top: parent.top
            anchors.bottom: parent.bottom

            Item {
                id: header
                width: parent.width
                height: 84

                Text {
                    x: 2
                    y: 0
                    text: "ELASTIC REVERSE TIME MIGRATION · SOLVER WORKBENCH"
                    color: "#4d7b98"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                    font.letterSpacing: 1.4
                }

                Text {
                    x: 2
                    y: 12
                    width: parent.width - 372
                    text: "ERTM 弹性波逆时偏移求解器"
                    color: "#f2faff"
                    font.family: Theme.fontUi
                    font.pixelSize: 22
                    font.weight: Font.Bold
                    font.letterSpacing: 0.5
                    elide: Text.ElideRight
                }

                Text {
                    x: 3
                    y: 46
                    width: parent.width - 372
                    text: "解耦一阶速度—应力方程 · 正 / 反传波场延拓 · PP/PS 成像 · Q 补偿"
                    color: "#7fa2ba"
                    font.family: Theme.fontUi
                    font.pixelSize: 9
                    elide: Text.ElideRight
                }

                // 运行控制组
                Row {
                    id: runControls
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.topMargin: 8
                    spacing: 6

                    Rectangle {
                        width: 132
                        height: 28
                        radius: 5
                        color: "#061a28"
                        border.width: 1
                        border.color: "#16445c"

                        Text {
                            x: 9
                            anchors.verticalCenter: parent.verticalCenter
                            text: bridge ? bridge.projectName : "ERTM"
                            color: "#cfeaf8"
                            font.family: Theme.fontUi
                            font.pixelSize: 7
                            font.weight: Font.DemiBold
                        }
                        Text {
                            anchors.right: parent.right
                            anchors.rightMargin: 8
                            anchors.verticalCenter: parent.verticalCenter
                            text: bridge ? "v" + bridge.projectVersion : "—"
                            color: "#4f7189"
                            font.family: Theme.fontMono
                            font.pixelSize: 7
                        }
                    }

                    Rectangle {
                        width: 76
                        height: 28
                        radius: 5
                        color: "#061a28"
                        border.width: 1
                        border.color: "#16445c"

                        Rectangle {
                            x: 9
                            anchors.verticalCenter: parent.verticalCenter
                            width: 6
                            height: 6
                            radius: 3
                            color: bridge && bridge.running ? "#38c9f5"
                                 : bridge && bridge.runState === "完成" ? "#31d98a"
                                 : bridge && bridge.runState.indexOf("失败") >= 0 ? "#ef5f6b" : "#3c6b83"
                        }
                        Text {
                            x: 21
                            anchors.verticalCenter: parent.verticalCenter
                            text: bridge ? bridge.runState : "空闲"
                            color: bridge && bridge.running ? "#8fe3ff" : "#8fadc2"
                            font.family: Theme.fontUi
                            font.pixelSize: 8
                        }
                    }

                    LabButton {
                        width: 96
                        height: 28
                        text: bridge && bridge.running ? "运行中…" : "运行流水线"
                        variant: "primary"
                        enabled: bridge ? !bridge.running : false
                        onClicked: if (bridge) bridge.runPipeline()
                    }

                    LabButton {
                        width: 44
                        height: 28
                        text: "输出"
                        variant: "ghost"
                        onClicked: if (bridge) bridge.openOutput()
                    }
                }

                // 运行进度细线
                Rectangle {
                    x: 0
                    y: 75
                    width: parent.width
                    height: 2
                    color: "#0b2b3d"

                    Rectangle {
                        width: parent.width * (bridge ? bridge.progress : 0)
                        height: parent.height
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: "#1d84b5" }
                            GradientStop { position: 1.0; color: "#38c9f5" }
                        }
                        Behavior on width { NumberAnimation { duration: 250 } }
                    }
                }
            }

            Loader {
                id: viewLoader
                objectName: "viewLoader"
                anchors.top: header.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                source: [
                    "views/ConfigView.qml",
                    "views/ModelCompareView.qml",
                    "views/RealDataView.qml",
                    "views/QCompView.qml",
                    "views/SourceView.qml"
                ][window.selectedIndex]
            }
        }
    }

    SplashScreen {
        id: splash
        anchors.fill: parent
        z: 1000
        visible: !window.entered
        backend: bridge
        onFinished: window.entered = true
    }

    Shortcut {
        sequences: [ "F5" ]
        enabled: window.entered
        onActivated: if (bridge) bridge.runPipeline()
    }
    Shortcut {
        sequence: "Ctrl+R"
        enabled: window.entered
        onActivated: if (bridge) bridge.reload()
    }
}
