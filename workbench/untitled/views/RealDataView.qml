import QtQuick
import "../components"
import "../theme/Theme.js" as Theme

/* 实测资料对比：RTM / Kirchhoff / 高斯束 / 差异 QC */
Item {
    id: view

    readonly property var dataInfo: bridge ? bridge.realData : ({})

    function openDataDir() {
        if (!bridge || !bridge.projectRoot)
            return;
        var path = bridge.projectRoot.replace(/\\/g, "/") + "/data";
        Qt.openUrlExternally("file:///" + path);
    }

    // ── 资料信息 ────────────────────────────────────────────────
    Rectangle {
        width: parent.width
        height: 84
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "实测资料"
            tag: view.dataInfo.available ? "已导入" : "未导入"
            tagColor: view.dataInfo.available ? "#31d98a" : "#efa63f"
        }

        Row {
            x: 12
            y: 45
            spacing: 8

            Repeater {
                model: [
                    { "label": "数据状态", "value": view.dataInfo.text || "实测资料未导入" },
                    { "label": "数据格式", "value": "SEG-Y / NPZ" },
                    { "label": "观测系统", "value": "2C / 3C · 任意坐标" }
                ]

                Rectangle {
                    width: 190
                    height: 30
                    radius: 4
                    color: "#082032"
                    border.width: 1
                    border.color: "#12384e"

                    Text {
                        x: 8; y: 4
                        text: modelData.label
                        color: "#5f8199"
                        font.family: Theme.fontUi
                        font.pixelSize: 7
                    }
                    Text {
                        x: 8; y: 15
                        width: parent.width - 16
                        text: modelData.value
                        color: modelData.label === "数据状态" ? "#efa63f" : "#b9d7e8"
                        font.family: Theme.fontMono
                        font.pixelSize: 8
                        elide: Text.ElideRight
                    }
                }
            }
        }

        Rectangle {
            x: parent.width - 118
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 10
            width: 106
            height: 24
            radius: 4
            color: "#0a2a40"
            border.width: 1
            border.color: "#1a4b66"

            Text {
                anchors.centerIn: parent
                text: "打开 data 目录"
                color: "#a9cfe2"
                font.family: Theme.fontUi
                font.pixelSize: 8
            }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: view.openDataDir()
            }
        }
    }

    // ── 四象限对比 ──────────────────────────────────────────────
    FieldPane {
        x: 0
        y: 94
        width: (view.width - 10) / 2
        height: 201
        title: "ERTM 实测成像 · PP"
        tag: "待导入"
        tagColor: "#efa63f"
        placeholderText: "ERTM 成像待接入"
        placeholderNote: "ertm/imaging.py · SEG-Y 输入"
        footerNote: "深度域 · z / m"
    }
    FieldPane {
        x: (view.width - 10) / 2 + 10
        y: 94
        width: (view.width - 10) / 2
        height: 201
        title: "Kirchhoff 偏移"
        tag: "参考"
        tagColor: "#3c6b83"
        placeholderText: "参考剖面待接入"
        placeholderNote: "Kirchhoff PSDM"
        footerNote: "同一观测系统 · 深度域"
    }
    FieldPane {
        x: 0
        y: 305
        width: (view.width - 10) / 2
        height: 201
        title: "高斯束偏移"
        tag: "参考"
        tagColor: "#3c6b83"
        placeholderText: "参考剖面待接入"
        placeholderNote: "Gaussian Beam Migration"
        footerNote: "束中心 / 走时场对比"
    }
    FieldPane {
        x: (view.width - 10) / 2 + 10
        y: 305
        width: (view.width - 10) / 2
        height: 201
        title: "差异剖面 · QC"
        tag: "QC"
        tagColor: "#7fc4df"
        placeholderText: "等待实测资料"
        placeholderNote: "RTM − Reference"
        footerNote: "振幅差 / 相关系数 / 频谱"
    }

    // ── 指标 ────────────────────────────────────────────────────
    MetricTile {
        x: 0
        y: 516
        width: 180
        height: 100
        label: "信噪比"
        value: "—"
        note: "目标：相对参考 +3 dB"
        state: "muted"
    }
    MetricTile {
        x: 190
        y: 516
        width: 180
        height: 100
        label: "同相轴连续性"
        value: "—"
        note: "人工 QC / 自动追踪"
        state: "muted"
    }
    MetricTile {
        x: 380
        y: 516
        width: 180
        height: 100
        label: "构造归位"
        value: "—"
        note: "与参考剖面比对"
        state: "muted"
    }

    Rectangle {
        x: 570
        y: 516
        width: parent.width - 570
        height: 100
        radius: 6
        color: "#081f30"
        border.width: 1
        border.color: "#12384e"

        Text {
            x: 10; y: 8
            text: "资料接入说明"
            color: "#6f93aa"
            font.family: Theme.fontUi
            font.pixelSize: 8
        }
        Text {
            x: 10; y: 24
            width: parent.width - 20
            text: "将 SEG-Y 放入项目 data/ 目录，并在配置中指定观测系统后即可生成对比剖面。"
            wrapMode: Text.WordWrap
            color: "#5f8199"
            font.family: Theme.fontUi
            font.pixelSize: 7
            lineHeight: 1.4
        }
    }
}
