import QtQuick
import "../components"
import "../theme/Theme.js" as Theme

/* 典型模型成像对比：溶洞 / 倾斜裂缝 / 交叉裂缝 */
Item {
    id: view

    readonly property var comparisons: bridge ? bridge.comparisons : []
    property int selectedIndex: 0
    readonly property var current: comparisons.length > 0 ? comparisons[selectedIndex] : ({})
    readonly property string modelSource: bridge && current.modelKey
        ? (bridge.previews[current.modelKey] || "") : ""

    function modelInfo() {
        var id = current.id || "";
        if (id === "cave")
            return [
                { "label": "异常体", "value": "椭圆溶洞" },
                { "label": "中心 / 尺寸", "value": "500, 350 m · 40×30 m", "color": "#8fe3ff" },
                { "label": "Vp / Qp", "value": "1800 m/s · 40", "color": "#8fe3ff" },
                { "label": "模型来源", "value": "config/default.json" }
            ];
        if (id === "fracture")
            return [
                { "label": "异常体", "value": "倾斜裂缝 55°" },
                { "label": "中心 / 长度", "value": "500, 390 m · 300 m", "color": "#8fe3ff" },
                { "label": "Vp / 半宽", "value": "2000 m/s · 12 m", "color": "#8fe3ff" },
                { "label": "模型来源", "value": "UI 预设（示意）" }
            ];
        return [
            { "label": "异常体", "value": "交叉裂缝 ±42°" },
            { "label": "中心 / 长度", "value": "500, 400 m · 260 m", "color": "#8fe3ff" },
            { "label": "Vp / 半宽", "value": "2000 m/s · 9 m", "color": "#8fe3ff" },
            { "label": "模型来源", "value": "UI 预设（示意）" }
        ];
    }

    // ── 模型选择卡片 ────────────────────────────────────────────
    Row {
        x: 0
        y: 0
        spacing: 10

        Repeater {
            model: view.comparisons

            Rectangle {
                width: (view.width - 20) / 3
                height: 72
                radius: 7
                color: index === view.selectedIndex ? "#0d344c" : "#081e2f"
                border.width: 1
                border.color: index === view.selectedIndex ? "#38c9f5" : "#123a54"

                Rectangle {
                    x: 8
                    y: 11
                    width: 50
                    height: 50
                    radius: 4
                    color: "#04111c"
                    border.width: 1
                    border.color: "#123a54"
                    clip: true

                    Image {
                        anchors.fill: parent
                        source: bridge ? (bridge.previews[modelData.modelKey] || "") : ""
                        fillMode: Image.PreserveAspectCrop
                        smooth: true
                        cache: false
                        asynchronous: true
                    }
                }

                Text {
                    x: 66
                    y: 10
                    text: modelData.name
                    color: index === view.selectedIndex ? "#f0faff" : "#c0dced"
                    font.family: Theme.fontUi
                    font.pixelSize: 11
                    font.weight: Font.DemiBold
                }
                Text {
                    x: 66
                    y: 26
                    width: parent.width - 74
                    text: modelData.subtitle
                    color: "#6f93aa"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                    elide: Text.ElideRight
                }
                Text {
                    x: 66
                    y: 48
                    text: modelData.status
                    color: modelData.id === "cave" ? "#31d98a" : "#efa63f"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }

                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: view.selectedIndex = index
                }
            }
        }
    }

    // ── 对比主卡 ────────────────────────────────────────────────
    Rectangle {
        id: compareCard
        y: 82
        width: parent.width
        height: 424
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "成像剖面对比"
            tag: "典型模型 · 3 类"
            tagColor: "#7fc4df"
        }

        ComparePane {
            x: 8
            y: 46
            width: (compareCard.width - 16 - 20) / 3
            height: 368
            title: "地质模型 · Vp"
            tag: "模型"
            tagColor: "#31d98a"
            source: view.modelSource
            placeholderText: "模型预览"
            placeholderNote: "output/models/model.npz"
            info: view.modelInfo()
            footer: view.current.note || ""
        }

        ComparePane {
            x: 8 + (compareCard.width - 16 - 20) / 3 + 10
            y: 46
            width: (compareCard.width - 16 - 20) / 3
            height: 368
            title: "ERTM 成像 · PP"
            tag: "占位"
            tagColor: "#efa63f"
            placeholderText: "ERTM 成像待接入"
            placeholderNote: "ertm/imaging.py"
            info: [
                { "label": "成像条件", "value": "PP · 零延迟互相关" },
                { "label": "归一化", "value": "源照明归一化" },
                { "label": "后处理", "value": "Laplacian" },
                { "label": "成像状态", "value": "零场占位", "color": "#efa63f" }
            ]
            footer: "等待算法分工同学接入"
        }

        ComparePane {
            x: 8 + ((compareCard.width - 16 - 20) / 3 + 10) * 2
            y: 46
            width: (compareCard.width - 16 - 20) / 3
            height: 368
            title: "参考偏移 · 深度域"
            tag: "待接入"
            tagColor: "#3c6b83"
            placeholderText: "参考剖面待接入"
            placeholderNote: "Kirchhoff / Gaussian Beam"
            info: [
                { "label": "参考方法", "value": "Kirchhoff" },
                { "label": "备选方法", "value": "高斯束偏移" },
                { "label": "用途", "value": "归位 / 极性对比" },
                { "label": "参考状态", "value": "未生成", "color": "#6f93aa" }
            ]
            footer: "用于与 ERTM 剖面互相关对比"
        }
    }

    // ── 指标 ────────────────────────────────────────────────────
    MetricTile {
        x: 0
        y: 516
        width: 180
        height: 100
        label: "成像归位误差"
        value: "—"
        note: "目标 < 1 个网格"
        state: "muted"
    }
    MetricTile {
        x: 190
        y: 516
        width: 180
        height: 100
        label: "异常体识别"
        value: "—"
        note: view.current.name ? view.current.name : "三类典型体"
        state: "muted"
    }
    MetricTile {
        x: 380
        y: 516
        width: 180
        height: 100
        label: "RTM / 参考一致性"
        value: "—"
        note: "相关系数 / RMSE"
        state: "muted"
    }

    Rectangle {
        x: 570
        y: 516
        width: 170
        height: 100
        radius: 6
        color: "#081f30"
        border.width: 1
        border.color: "#12384e"

        Text {
            x: 10
            y: 8
            text: "当前状态"
            color: "#6f93aa"
            font.family: Theme.fontUi
            font.pixelSize: 8
        }
        Text {
            x: 10
            y: 24
            width: parent.width - 20
            text: "模型预览来自项目真实配置；成像与参考剖面为占位，算法接入后自动更新。"
            wrapMode: Text.WordWrap
            color: "#5f8199"
            font.family: Theme.fontUi
            font.pixelSize: 7
            lineHeight: 1.4
        }
    }
}
