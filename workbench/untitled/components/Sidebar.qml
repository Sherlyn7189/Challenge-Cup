import QtQuick
import "../theme/Theme.js" as Theme

/* 左侧工作流导航 */
Item {
    id: sidebar

    property var backend: null
    property int selectedIndex: 0

    signal select(int index)
    signal openProject()
    signal refresh()

    CardBackground { anchors.fill: parent }

    readonly property var env: sidebar.backend ? sidebar.backend.environment : ({})
    readonly property int readyModules: {
        var count = 0;
        if (!sidebar.backend)
            return 0;
        for (var i = 0; i < sidebar.backend.modules.length; ++i) {
            if (sidebar.backend.modules[i].status === "ready")
                count++;
        }
        return count;
    }

    Text {
        x: 14
        y: 12
        text: "SEISMIC SOLVER · WORKBENCH"
        color: "#3f6d88"
        font.family: Theme.fontUi
        font.pixelSize: 7
        font.letterSpacing: 1.4
    }

    // ── 品牌 ────────────────────────────────────────────────────
    Item {
        x: 12
        y: 28
        width: parent.width - 24
        height: 38

        Rectangle {
            id: mark
            width: 36
            height: 36
            radius: 8
            border.width: 1
            border.color: "#4fc4e8"
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#12506f" }
                GradientStop { position: 1.0; color: "#082434" }
            }

            Canvas {
                anchors.fill: parent
                onPaint: {
                    var ctx = getContext("2d");
                    ctx.clearRect(0, 0, width, height);
                    ctx.strokeStyle = "#8fe3ff";
                    ctx.lineWidth = 1.6;
                    ctx.beginPath();
                    ctx.moveTo(5, height / 2);
                    ctx.lineTo(10, height / 2);
                    ctx.lineTo(13, height / 2 - 7);
                    ctx.lineTo(17, height / 2 + 9);
                    ctx.lineTo(21, height / 2 - 12);
                    ctx.lineTo(25, height / 2 + 6);
                    ctx.lineTo(29, height / 2 - 3);
                    ctx.lineTo(32, height / 2);
                    ctx.stroke();
                }
            }
        }

        Text {
            anchors.left: mark.right
            anchors.leftMargin: 9
            anchors.top: parent.top
            anchors.topMargin: -2
            text: "ERTM"
            color: "#f0f9ff"
            font.family: Theme.fontUi
            font.pixelSize: 18
            font.weight: Font.Bold
            font.letterSpacing: 1.2
        }
        Text {
            anchors.left: mark.right
            anchors.leftMargin: 10
            anchors.top: parent.top
            anchors.topMargin: 22
            text: "弹性波逆时偏移"
            color: "#6f93aa"
            font.family: Theme.fontUi
            font.pixelSize: 8
            font.letterSpacing: 0.6
        }
    }

    // ── 项目标识 ────────────────────────────────────────────────
    Rectangle {
        x: 12
        y: 74
        width: parent.width - 24
        height: 28
        radius: 5
        color: "#082032"
        border.width: 1
        border.color: "#123c53"

        Text {
            x: 9
            anchors.verticalCenter: parent.verticalCenter
            text: sidebar.backend ? sidebar.backend.projectName : "ERTM"
            color: "#cfeaf8"
            font.family: Theme.fontUi
            font.pixelSize: 8
            font.weight: Font.DemiBold
        }
        Text {
            anchors.right: parent.right
            anchors.rightMargin: 9
            anchors.verticalCenter: parent.verticalCenter
            text: sidebar.backend ? "v" + sidebar.backend.projectVersion : "—"
            color: "#5f8199"
            font.family: Theme.fontMono
            font.pixelSize: 7
        }
    }

    Rectangle {
        x: 12
        y: 110
        width: parent.width - 24
        height: 1
        color: "#123a54"
    }

    // ── 导航 ────────────────────────────────────────────────────
    Column {
        id: nav
        x: 10
        y: 118
        width: parent.width - 20
        spacing: 6

        NavItem {
            width: nav.width
            step: 1
            title: "参数配置"
            subtitle: "config · grid · model"
            state: "ready"
            selected: sidebar.selectedIndex === 0
            onClicked: sidebar.select(0)
        }
        NavItem {
            width: nav.width
            step: 2
            title: "典型模型对比"
            subtitle: "cave · fracture · cross"
            state: "warn"
            selected: sidebar.selectedIndex === 1
            onClicked: sidebar.select(1)
        }
        NavItem {
            width: nav.width
            step: 3
            title: "实测资料对比"
            subtitle: "SEG-Y · RTM vs GBM"
            state: "idle"
            selected: sidebar.selectedIndex === 2
            onClicked: sidebar.select(2)
        }
        NavItem {
            width: nav.width
            step: 4
            title: "Q 补偿"
            subtitle: "GSLS · gain control"
            state: "ready"
            selected: sidebar.selectedIndex === 3
            onClicked: sidebar.select(3)
        }
        NavItem {
            width: nav.width
            step: 5
            title: "求解器源码"
            subtitle: "ertm/*.py · " + (sidebar.backend ? sidebar.backend.modules.length : 0) + " files"
            state: "ready"
            selected: sidebar.selectedIndex === 4
            onClicked: sidebar.select(4)
        }
    }

    // ── 核心模块状态 ────────────────────────────────────────────
    Text {
        x: 14
        y: 416
        text: "CORE MODULES"
        color: "#3f6d88"
        font.family: Theme.fontUi
        font.pixelSize: 7
        font.letterSpacing: 1.3
    }

    Column {
        id: coreList
        x: 12
        y: 432
        width: parent.width - 24
        spacing: 0

        Repeater {
            model: [
                { "name": "差分算子", "state": "待接入", "color": "#efa63f" },
                { "name": "正 / 反传延拓", "state": "占位", "color": "#efa63f" },
                { "name": "PP / PS 成像", "state": "占位", "color": "#efa63f" },
                { "name": "Q 补偿", "state": "已实现", "color": "#31d98a" }
            ]

            Item {
                width: coreList.width
                height: 26

                Rectangle {
                    x: 2
                    anchors.verticalCenter: parent.verticalCenter
                    width: 4
                    height: 4
                    radius: 2
                    color: modelData.color
                }
                Text {
                    x: 12
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData.name
                    color: "#7d9fb5"
                    font.family: Theme.fontUi
                    font.pixelSize: 8
                }
                Text {
                    anchors.right: parent.right
                    anchors.rightMargin: 2
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData.state
                    color: modelData.color
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
            }
        }
    }

    // ── 环境 ────────────────────────────────────────────────────
    Rectangle {
        id: envCard
        x: 12
        y: 544
        width: parent.width - 24
        height: 64
        radius: 6
        color: "#081f30"
        border.width: 1
        border.color: "#12384e"

        Text {
            x: 9
            y: 7
            text: "运行环境"
            color: "#5f8199"
            font.family: Theme.fontUi
            font.pixelSize: 7
            font.letterSpacing: 0.5
        }
        Text {
            x: 9
            y: 22
            text: "Python " + (sidebar.env.python || "—")
            color: "#b9d7e8"
            font.family: Theme.fontMono
            font.pixelSize: 7
        }
        Text {
            x: 9
            y: 35
            text: "NumPy " + (sidebar.env.numpy || "—") + "  ·  " + (sidebar.env.platform || "—")
            color: "#5f8199"
            font.family: Theme.fontMono
            font.pixelSize: 7
            elide: Text.ElideRight
            width: parent.width - 18
        }
        Text {
            x: 9
            y: 48
            text: sidebar.readyModules + " / " + (sidebar.backend ? sidebar.backend.modules.length : 0) + " 模块已实现"
            color: "#31d98a"
            font.family: Theme.fontUi
            font.pixelSize: 7
        }
    }

    // ── 底部按钮 ────────────────────────────────────────────────
    LabButton {
        id: openProjectBtn
        x: 12
        width: parent.width - 24
        height: 34
        anchors.bottom: refreshBtn.top
        anchors.bottomMargin: 8
        text: "打开项目文件夹"
        variant: "ghost"
        onClicked: sidebar.openProject()
    }

    LabButton {
        id: refreshBtn
        x: 12
        width: parent.width - 24
        height: 34
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 6
        text: "刷新配置与产物"
        variant: "ghost"
        onClicked: sidebar.refresh()
    }
}
