import QtQuick
import "../theme/Theme.js" as Theme

/* 单炮记录说明弹层 */
Item {
    id: dialog

    property var info: bridge ? bridge.shotInfo : ({})
    property string source: bridge ? (bridge.previews.shot || "") : ""
    signal closed()

    z: 100

    function fmt(value, digits, fallback) {
        var n = Number(value);
        if (!isFinite(n))
            return fallback === undefined ? "—" : fallback;
        return digits === undefined ? String(n) : n.toFixed(digits);
    }
    function intText(value) {
        var n = Number(value);
        return isFinite(n) ? String(Math.round(n)) : "—";
    }
    function sourceKind(value) {
        return value === "ricker" ? "Ricker" : value === "gaussian" ? "Gaussian" : (value || "—");
    }

    readonly property var metaItems: [
        { "label": "炮点", "value": "(" + intText(info.sx) + ", " + intText(info.sz) + ") m" },
        { "label": "检波点", "value": intText(info.nrec) + " 道 · " + (info.component || "vz") },
        { "label": "接收测线", "value": intText(info.rx0) + "–" + intText(info.rx1) + " m · z=" + intText(info.rz) + " m" },
        { "label": "时间采样", "value": intText(info.nt) + " × " + fmt(Number(info.dt) * 1000, 1) + " ms · " + fmt(info.tmax, 3) + " s" },
        { "label": "子波", "value": sourceKind(info.source_kind) + " · " + intText(info.f0) + " Hz" },
        { "label": "平均速度 / RMS", "value": (Number(info.vmean) > 0 ? intText(info.vmean) + " m/s" : "—") + " · " + (info.available ? fmt(info.rms, 3) : "—") }
    ]

    Rectangle {
        anchors.fill: parent
        color: "#01070e"
        opacity: 0.78

        MouseArea {
            anchors.fill: parent
            onClicked: dialog.closed()
        }
    }

    Rectangle {
        id: card
        width: 640
        height: 520
        anchors.centerIn: parent
        radius: 10
        color: "#071c2c"
        border.width: 1
        border.color: "#1f5d7c"

        Rectangle {
            anchors.fill: parent
            anchors.margins: 1
            radius: 9
            color: "transparent"
            border.width: 1
            border.color: "#2f86aa"
            opacity: 0.25
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {}
        }

        Text {
            x: 22
            y: 16
            text: "单炮记录（Shot Record）"
            color: "#f2faff"
            font.family: Theme.fontUi
            font.pixelSize: 15
            font.weight: Font.DemiBold
            font.letterSpacing: 0.5
        }
        Text {
            x: 22
            y: 39
            text: "时间 × 检波点 · ERTM 检波点波场的输入数据"
            color: "#5f8199"
            font.family: Theme.fontUi
            font.pixelSize: 8
            font.letterSpacing: 0.4
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 18
            anchors.top: parent.top
            anchors.topMargin: 10
            text: "×"
            color: "#8fadc2"
            font.family: Theme.fontUi
            font.pixelSize: 18

            MouseArea {
                anchors.fill: parent
                anchors.margins: -6
                cursorShape: Qt.PointingHandCursor
                onClicked: dialog.closed()
            }
        }

        ShotPreview {
            x: 20
            y: 60
            width: parent.width - 40
            height: 230
            compact: false
            source: dialog.source
            info: dialog.info
        }

        Text {
            x: 22
            y: 300
            width: parent.width - 44
            height: 48
            text: "震源在炮点激发一次，沿测线布设的检波器按时间记录波场振动，形成“时间 × 检波点”的二维记录：每一列是一道（trace），每一行是一个时刻。ERTM 将记录时间反转后从检波点注入，得到检波点波场 W_r，再与正传源波场 W_s 做零延迟互相关，形成成像值。"
            wrapMode: Text.WordWrap
            color: "#9dbfd6"
            font.family: Theme.fontUi
            font.pixelSize: 8
            lineHeight: 1.5
            elide: Text.ElideRight
        }

        Grid {
            x: 22
            y: 356
            width: parent.width - 44
            columns: 2
            columnSpacing: 8
            rowSpacing: 6

            Repeater {
                model: dialog.metaItems

                Item {
                    width: (card.width - 52) / 2
                    height: 22

                    Rectangle {
                        anchors.bottom: parent.bottom
                        width: parent.width
                        height: 1
                        color: "#10384d"
                        opacity: 0.7
                    }
                    Text {
                        x: 0
                        anchors.verticalCenter: parent.verticalCenter
                        text: modelData.label
                        color: "#5f8199"
                        font.family: Theme.fontUi
                        font.pixelSize: 7
                    }
                    Text {
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width - 92
                        horizontalAlignment: Text.AlignRight
                        text: modelData.value
                        color: "#cfeaf8"
                        font.family: Theme.fontMono
                        font.pixelSize: 7
                        elide: Text.ElideRight
                    }
                }
            }
        }

        Rectangle {
            x: 22
            y: 438
            width: parent.width - 44
            height: 46
            radius: 5
            color: "#1e1a10"
            border.width: 1
            border.color: "#5a4520"

            Rectangle {
                x: 0
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                width: 3
                radius: 1.5
                color: "#efa63f"
                opacity: 0.8
            }
            Text {
                x: 12
                y: 7
                width: parent.width - 22
                text: dialog.info.available
                      ? "当前为占位直达波：按 t = delay + r / v_mean 生成，只用于验证 I/O 与流程；真实正演将包含反射 / 转换波、CPML 吸收边界与 Q 衰减。"
                      : "尚未生成单炮记录：运行流水线后按 t = delay + r / v_mean 生成占位直达波；真实正演将包含反射 / 转换波、CPML 吸收边界与 Q 衰减。"
                wrapMode: Text.WordWrap
                color: dialog.info.available ? "#c9a56c" : "#8fadc2"
                font.family: Theme.fontUi
                font.pixelSize: 7
                lineHeight: 1.4
            }
        }

        LabButton {
            anchors.right: parent.right
            anchors.rightMargin: 20
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            width: 80
            height: 26
            text: "关闭"
            variant: "primary"
            onClicked: dialog.closed()
        }
    }
}
