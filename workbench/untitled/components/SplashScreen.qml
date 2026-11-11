import QtQuick
import "../theme/Theme.js" as Theme

/* ERTM 启动封面：加载配置、产物与预览，再进入工作台 */
Item {
    id: splash

    property var backend: null
    property real progress: 0.0
    property bool done: false
    signal finished()

    readonly property var previews: splash.backend ? splash.backend.previews : ({})

    Text {
        id: eyebrow
        x: 110
        y: 108
        text: "AI VIRTUAL EXPERIMENT · ERTM WORKBENCH"
        color: "#4d7b98"
        font.family: Theme.fontUi
        font.pixelSize: 8
        font.letterSpacing: 2.2
    }

    Rectangle {
        id: logo
        x: 110
        y: 138
        width: 64
        height: 64
        radius: 14
        border.width: 1
        border.color: "#5fd0f2"
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#156080" }
            GradientStop { position: 1.0; color: "#082434" }
        }

        Text {
            anchors.centerIn: parent
            text: "ERTM"
            color: "#eafaff"
            font.family: Theme.fontMono
            font.pixelSize: 14
            font.weight: Font.Bold
            font.letterSpacing: 1
        }
        Rectangle {
            anchors.fill: parent
            anchors.margins: 5
            radius: 10
            color: "transparent"
            border.width: 1
            border.color: "#8fe3ff"
            opacity: 0.18
        }
    }

    Text {
        x: 194
        y: 142
        text: "ERTM"
        color: "#f2faff"
        font.family: Theme.fontUi
        font.pixelSize: 44
        font.weight: Font.Bold
        font.letterSpacing: 2
    }
    Text {
        x: 198
        y: 192
        text: "Elastic Reverse Time Migration"
        color: "#6f93aa"
        font.family: Theme.fontUi
        font.pixelSize: 9
        font.letterSpacing: 0.8
    }

    Text {
        x: 110
        y: 238
        text: "弹性波逆时偏移求解器"
        color: "#e7f4ff"
        font.family: Theme.fontUi
        font.pixelSize: 30
        font.weight: Font.DemiBold
        font.letterSpacing: 1.2
    }
    Text {
        x: 112
        y: 286
        text: "参数配置  ·  典型模型对比  ·  实测资料对比  ·  Q 补偿  ·  求解器源码"
        color: "#7fa2ba"
        font.family: Theme.fontUi
        font.pixelSize: 9
        font.letterSpacing: 0.4
    }

    // 进度条
    Rectangle {
        x: 110
        y: 382
        width: 520
        height: 6
        radius: 3
        color: "#0b2b3d"

        Rectangle {
            width: parent.width * splash.progress
            height: parent.height
            radius: 3
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#1d84b5" }
                GradientStop { position: 0.7; color: "#38c9f5" }
                GradientStop { position: 1.0; color: "#8fe3ff" }
            }
            Behavior on width { NumberAnimation { duration: 120; easing.type: Easing.OutCubic } }
        }
    }
    Text {
        x: 640
        y: 371
        text: Math.round(splash.progress * 100) + "%"
        color: "#8fe3ff"
        font.family: Theme.fontMono
        font.pixelSize: 11
        font.weight: Font.DemiBold
    }

    Text {
        id: statusText
        x: 110
        y: 400
        text: splash.progress < 0.22 ? "正在读取 ERTM 配置与模型参数…"
              : splash.progress < 0.48 ? "正在加载炮记录与成像产物…"
              : splash.progress < 0.74 ? "正在生成模型与成像预览…"
              : splash.progress < 0.95 ? "正在准备求解器工作台…"
              : "工作台就绪，正在进入…"
        color: "#7fa2ba"
        font.family: Theme.fontUi
        font.pixelSize: 9
        font.letterSpacing: 0.3
    }

    // 检查点
    Row {
        x: 110
        y: 438
        spacing: 20

        Repeater {
            model: [
                { "label": "配置", "threshold": 0.18 },
                { "label": "运行产物", "threshold": 0.46 },
                { "label": "预览图", "threshold": 0.74 },
                { "label": "工作台", "threshold": 0.95 }
            ]

            Row {
                spacing: 5
                Rectangle {
                    anchors.verticalCenter: parent.verticalCenter
                    width: 6
                    height: 6
                    radius: 3
                    color: splash.progress >= modelData.threshold ? "#31d98a" : "#23495f"
                }
                Text {
                    text: modelData.label
                    color: splash.progress >= modelData.threshold ? "#8fadc2" : "#4f7189"
                    font.family: Theme.fontUi
                    font.pixelSize: 8
                }
            }
        }
    }

    // 右侧反演模型预览
    Rectangle {
        x: 742
        y: 156
        width: 430
        height: 330
        radius: 10
        color: "#04121c"
        border.width: 1
        border.color: "#154159"
        clip: true

        Image {
            anchors.fill: parent
            anchors.margins: 1
            source: splash.previews.modelVp || ""
            fillMode: Image.PreserveAspectFit
            smooth: true
            cache: false
            asynchronous: true
            opacity: 0.88
        }

        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: 40
            gradient: Gradient {
                GradientStop { position: 0.0; color: "transparent" }
                GradientStop { position: 1.0; color: "#04121c" }
            }
        }

        Text {
            x: 12
            y: 10
            text: "ERTM · Vp 速度模型"
            color: "#a9edff"
            font.family: Theme.fontUi
            font.pixelSize: 8
            font.letterSpacing: 0.5
        }
        Text {
            anchors.right: parent.right
            anchors.rightMargin: 12
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 10
            text: "output / models / model.npz"
            color: "#5f8199"
            font.family: Theme.fontMono
            font.pixelSize: 6
        }
    }

    // 装饰波形
    Canvas {
        x: 0
        y: 560
        width: parent.width
        height: 130
        opacity: 0.35

        onPaint: {
            var ctx = getContext("2d");
            ctx.clearRect(0, 0, width, height);
            for (var k = 0; k < 3; ++k) {
                ctx.strokeStyle = k === 0 ? "#38c9f5" : (k === 1 ? "#1fd1c4" : "#1d84b5");
                ctx.lineWidth = 1;
                ctx.beginPath();
                for (var x = 0; x < width; x += 4) {
                    var y = height / 2 + Math.sin(x * 0.012 + k * 1.7) * (14 + k * 7)
                            * Math.exp(-Math.pow((x - width * 0.35) / 420, 2));
                    if (x === 0)
                        ctx.moveTo(x, y);
                    else
                        ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
        }
    }

    Text {
        x: 110
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 32
        text: "ERTM Workbench · PySide6 / Qt Quick · SeismicProjectERTM"
        color: "#3f6479"
        font.family: Theme.fontMono
        font.pixelSize: 7
    }

    Text {
        anchors.right: parent.right
        anchors.rightMargin: 40
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 30
        text: "跳过 ›"
        color: "#5f8199"
        font.family: Theme.fontUi
        font.pixelSize: 8
        visible: !splash.done

        MouseArea {
            anchors.fill: parent
            anchors.margins: -6
            cursorShape: Qt.PointingHandCursor
            onClicked: splash.progress = 1.0
        }
    }

    Timer {
        interval: 60
        running: splash.progress < 1.0
        repeat: true
        onTriggered: splash.progress = Math.min(1.0, splash.progress + 0.025)
    }

    onProgressChanged: {
        if (progress >= 1.0 && !done) {
            done = true;
            finishTimer.start();
        }
    }

    Timer {
        id: finishTimer
        interval: 500
        onTriggered: splash.finished()
    }

    // 背景
    Canvas {
        z: -1
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d");
            var bg = ctx.createLinearGradient(0, 0, width, height);
            bg.addColorStop(0.0, "#071b2c");
            bg.addColorStop(0.5, "#04101d");
            bg.addColorStop(1.0, "#020810");
            ctx.fillStyle = bg;
            ctx.fillRect(0, 0, width, height);

            var glow = ctx.createRadialGradient(width * 0.36, height * 0.34, 20,
                                               width * 0.36, height * 0.34, width * 0.6);
            glow.addColorStop(0.0, "rgba(28, 120, 168, 0.18)");
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
        }
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        Component.onCompleted: requestPaint()
    }
}
