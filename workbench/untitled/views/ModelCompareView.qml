import QtQuick
import "../components"
import "../theme/Theme.js" as Theme

/* 典型模型成像对比：溶洞原始图件 + ERTM 多分量 + Kirchhoff 参考 */
Item {
    id: view

    readonly property var comparisons: bridge ? bridge.comparisons : []
    property int selectedIndex: 0
    readonly property var current: comparisons.length > 0 ? comparisons[selectedIndex] : ({})
    readonly property bool caveActive: current.id === "cave"
    readonly property string modelSource: bridge && current.modelKey
        ? (bridge.previews[current.modelKey] || "") : ""

    readonly property url caveModelAsset: Qt.resolvedUrl("../../assets/compare/cave_model.png")
    readonly property url ertmAsset: Qt.resolvedUrl("../../assets/compare/ertm_vx.png")
    readonly property url kirchhoffAsset: Qt.resolvedUrl("../../assets/compare/kirchhoff.png")

    function modelInfo() {
        var id = current.id || "";
        if (id === "cave")
            return [
                { "label": "异常体", "value": "溶洞" },
                { "label": "模型尺度", "value": "20 × 50 m", "color": "#8fe3ff" },
                { "label": "图件来源", "value": "图19(b)" }
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
                        source: modelData.id === "cave"
                                ? view.caveModelAsset
                                : (bridge ? (bridge.previews[modelData.modelKey] || "") : "")
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
                    text: modelData.id === "cave" ? "图件已接入" : modelData.status
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
            tag: view.caveActive ? "溶洞 · 原始图件" : "典型模型 · 3 类"
            tagColor: view.caveActive ? "#31d98a" : "#7fc4df"
        }

        ComparePane {
            x: 8
            y: 46
            width: (compareCard.width - 16 - 20) / 3
            height: 368
            title: "地质模型 · Vp"
            tag: view.caveActive ? "图19(b)" : "模型"
            tagColor: "#31d98a"
            source: view.caveActive ? view.caveModelAsset : view.modelSource
            placeholderText: "模型预览"
            placeholderNote: view.caveActive ? "assets/compare/cave_model.png" : "output/models/model.npz"
            imageHeight: view.caveActive ? 230 : 190
            info: view.modelInfo()
            footer: view.caveActive ? "图19(b) 溶洞模型 · 仅重配色，数据未改" : (view.current.note || "")
        }

        ComparePane {
            x: 8 + (compareCard.width - 16 - 20) / 3 + 10
            y: 46
            width: (compareCard.width - 16 - 20) / 3
            height: 368
            title: "ERTM 成像 · 多分量"
            tag: view.caveActive ? "vx 分量" : "占位"
            tagColor: view.caveActive ? "#31d98a" : "#efa63f"
            source: view.caveActive ? view.ertmAsset : ""
            placeholderText: "ERTM 成像待接入"
            placeholderNote: "ertm/imaging.py"
            imageHeight: view.caveActive ? 230 : 190
            info: view.caveActive ? [
                { "label": "求解", "value": "解耦一阶速度—应力" },
                { "label": "成像", "value": "PP · 多分量" },
                { "label": "显示", "value": "vx（图21(a)）", "color": "#8fe3ff" }
            ] : [
                { "label": "成像条件", "value": "PP · 零延迟互相关" },
                { "label": "归一化", "value": "源照明归一化" },
                { "label": "后处理", "value": "Laplacian" },
                { "label": "成像状态", "value": "零场占位", "color": "#efa63f" }
            ]
            footer: view.caveActive
                    ? "多分量 vx / vs / vz / vxz · 当前显示 vx"
                    : "等待算法分工同学接入"
        }

        ComparePane {
            x: 8 + ((compareCard.width - 16 - 20) / 3 + 10) * 2
            y: 46
            width: (compareCard.width - 16 - 20) / 3
            height: 368
            title: "参考偏移 · 深度域"
            tag: view.caveActive ? "Kirchhoff" : "待接入"
            tagColor: view.caveActive ? "#31d98a" : "#3c6b83"
            source: view.caveActive ? view.kirchhoffAsset : ""
            placeholderText: "参考剖面待接入"
            placeholderNote: "Kirchhoff / Gaussian Beam"
            imageHeight: view.caveActive ? 230 : 190
            info: view.caveActive ? [
                { "label": "方法", "value": "Kirchhoff" },
                { "label": "域", "value": "深度域" },
                { "label": "图件来源", "value": "图1(c)", "color": "#8fe3ff" }
            ] : [
                { "label": "参考方法", "value": "Kirchhoff" },
                { "label": "备选方法", "value": "高斯束偏移" },
                { "label": "用途", "value": "归位 / 极性对比" },
                { "label": "参考状态", "value": "未生成", "color": "#6f93aa" }
            ]
            footer: view.caveActive
                    ? "与 ERTM 剖面做归位 / 极性对比"
                    : "用于与 ERTM 剖面互相关对比"
        }
    }

    // ── 指标 ────────────────────────────────────────────────────
    MetricTile {
        x: 0
        y: 516
        width: 180
        height: 100
        label: "地质模型"
        value: view.caveActive ? "溶洞" : "示意图"
        note: view.caveActive ? "图19(b) · 20 × 50 m" : "UI 预设"
        state: "ok"
    }
    MetricTile {
        x: 190
        y: 516
        width: 180
        height: 100
        label: "ERTM 成像"
        value: view.caveActive ? "vx" : "—"
        note: view.caveActive ? "多分量 · 图21(a)" : "等待接入"
        state: view.caveActive ? "ok" : "muted"
    }
    MetricTile {
        x: 380
        y: 516
        width: 180
        height: 100
        label: "参考偏移"
        value: view.caveActive ? "Kirchhoff" : "—"
        note: view.caveActive ? "深度域 · 图1(c)" : "参考剖面"
        state: view.caveActive ? "ok" : "muted"
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
            text: "图件说明"
            color: "#6f93aa"
            font.family: Theme.fontUi
            font.pixelSize: 8
        }
        Text {
            x: 10
            y: 24
            width: parent.width - 20
            text: view.caveActive
                  ? "三张原始图件已按深色界面重绘：仅改变配色，未改变模型与成像数据。"
                  : "当前模型为界面预设示意；溶洞模型已接入原始图件。"
            wrapMode: Text.WordWrap
            color: "#5f8199"
            font.family: Theme.fontUi
            font.pixelSize: 7
            lineHeight: 1.4
        }
    }
}
