import QtQuick
import "../components"
import "../theme/Theme.js" as Theme

/* 参数配置：读取 config/default.json 并显示真实产物预览 */
Item {
    id: view

    readonly property var cfg: bridge ? bridge.cfg : ({})
    readonly property var grid: cfg.grid || ({})
    readonly property var model: cfg.model || ({})
    readonly property var layers: model.layers || []
    readonly property var anomaly: model.anomaly || ({})
    readonly property var source: cfg.source || ({})
    readonly property var receivers: cfg.receivers || ({})
    readonly property var solver: cfg.solver || ({})
    readonly property var imaging: cfg.imaging || ({})
    readonly property var checks: bridge ? bridge.checks : []

    property int fieldIndex: 0
    property bool shotDetailOpen: false
    readonly property var fieldKeys: [ "modelVp", "modelVs", "modelRho" ]
    readonly property var fieldNames: [ "Vp", "Vs", "ρ" ]

    function num(value, digits) {
        if (value === undefined || value === null || isNaN(Number(value)))
            return "—";
        return Number(value).toFixed(digits === undefined ? 1 : digits);
    }
    function rangeOf(key) {
        var values = [];
        if (model["background_" + key] !== undefined)
            values.push(Number(model["background_" + key]));
        for (var i = 0; i < layers.length; ++i) {
            if (layers[i][key] !== undefined)
                values.push(Number(layers[i][key]));
        }
        if (anomaly[key] !== undefined && anomaly[key] !== null)
            values.push(Number(anomaly[key]));
        values = values.filter(function (v) { return isFinite(v); });
        if (values.length === 0)
            return "—";
        return num(Math.min.apply(null, values), 0) + "–" + num(Math.max.apply(null, values), 0);
    }
    function anomalyText() {
        var kind = anomaly.kind || "";
        if (kind === "cave") return "溶洞";
        if (kind === "fracture") return "裂缝";
        if (kind === "fault") return "断层";
        return kind || "—";
    }
    function kindName(value) {
        return value === "ricker" ? "Ricker" : value === "gaussian" ? "Gaussian" : value || "—";
    }
    function metricValue(key) {
        if (!bridge)
            return "—";
        for (var i = 0; i < bridge.metrics.length; ++i) {
            if (bridge.metrics[i].key === key)
                return bridge.metrics[i].value;
        }
        return "—";
    }

    readonly property string vpRange: rangeOf("vp")
    readonly property string vsRange: rangeOf("vs")
    readonly property string rhoRange: rangeOf("rho")

    // ── 左栏 · 网格与地质模型 ──────────────────────────────────
    Rectangle {
        id: modelCard
        width: 440
        height: 300
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            id: modelHeader
            width: parent.width
            title: "网格与地质模型"
            tag: "config/default.json"
            tagColor: "#7fc4df"
        }

        Row {
            x: 12
            y: 46
            spacing: 8

            Repeater {
                model: [
                    { "label": "网格 nx × nz", "value": view.num(view.grid.nx, 0) + " × " + view.num(view.grid.nz, 0), "note": (Number(view.grid.nx) * Number(view.grid.nz)).toLocaleString() + " 点" },
                    { "label": "间距 dx / dz", "value": view.num(view.grid.dx, 1) + " / " + view.num(view.grid.dz, 1), "note": "m / m" },
                    { "label": "范围 x / z", "value": view.num((Number(view.grid.x0) + (Number(view.grid.nx) - 1) * Number(view.grid.dx)), 0) + " / " + view.num((Number(view.grid.z0) + (Number(view.grid.nz) - 1) * Number(view.grid.dz)), 0), "note": "m / m" }
                ]

                Rectangle {
                    width: 133
                    height: 42
                    radius: 5
                    color: "#082234"
                    border.width: 1
                    border.color: "#123c53"

                    Text {
                        x: 8; y: 5
                        text: modelData.label
                        color: "#5f8199"
                        font.family: Theme.fontUi
                        font.pixelSize: 7
                    }
                    Text {
                        x: 8; y: 17
                        text: modelData.value
                        color: "#d9ecf8"
                        font.family: Theme.fontMono
                        font.pixelSize: 10
                        font.weight: Font.DemiBold
                    }
                    Text {
                        x: 8; y: 31
                        text: modelData.note
                        color: "#42657b"
                        font.family: Theme.fontMono
                        font.pixelSize: 7
                    }
                }
            }
        }

        Text {
            x: 12
            y: 98
            text: "层状背景"
            color: "#7b9fb5"
            font.family: Theme.fontUi
            font.pixelSize: 8
            font.weight: Font.DemiBold
        }
        Text {
            anchors.right: parent.right
            anchors.rightMargin: 12
            y: 99
            text: view.layers.length + " 层"
            color: "#4f7189"
            font.family: Theme.fontUi
            font.pixelSize: 7
        }

        // 层参数表
        Item {
            id: layerTable
            x: 12
            y: 114
            width: parent.width - 24
            height: 84

            Rectangle {
                anchors.fill: parent
                radius: 4
                color: "#061a28"
                border.width: 1
                border.color: "#10384d"
            }

            Row {
                id: layerHeader
                x: 6
                y: 3
                width: parent.width - 12
                height: 14

                Repeater {
                    model: [ "层顶 z", "Vp", "Vs", "ρ", "Qp", "Qs" ]
                    Text {
                        width: (layerHeader.width - 44) / 5 + (index === 0 ? 44 : 0)
                        text: modelData
                        color: "#4f7189"
                        font.family: Theme.fontUi
                        font.pixelSize: 7
                        horizontalAlignment: index === 0 ? Text.AlignLeft : Text.AlignHCenter
                    }
                }
            }

            Repeater {
                model: view.layers

                Rectangle {
                    x: 0
                    y: 18 + index * 21
                    width: layerTable.width
                    height: 21
                    color: index % 2 === 0 ? "#081f30" : "transparent"

                    Row {
                        x: 6
                        width: layerTable.width - 12
                        height: parent.height

                        Repeater {
                            model: [
                                view.num(modelData.z_top, 0) + " m",
                                view.num(modelData.vp, 0),
                                view.num(modelData.vs, 0),
                                view.num(modelData.rho, 0),
                                view.num(modelData.qp, 0),
                                view.num(modelData.qs, 0)
                            ]
                            Text {
                                width: (layerTable.width - 56) / 5 + (index === 0 ? 44 : 0)
                                text: modelData
                                color: index === 0 ? "#8fadc2" : "#cfeaf8"
                                font.family: Theme.fontMono
                                font.pixelSize: 8
                                horizontalAlignment: index === 0 ? Text.AlignLeft : Text.AlignHCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }
                    }
                }
            }
        }

        // 背景参数
        Text {
            x: 12
            y: 202
            text: "背景  Vp " + view.num(model.background_vp, 0) + " · Vs " + view.num(model.background_vs, 0)
                  + " · ρ " + view.num(model.background_rho, 0)
                  + " · Qp " + view.num(model.background_qp, 0)
                  + " · Qs " + view.num(model.background_qs, 0)
            color: "#5f8199"
            font.family: Theme.fontMono
            font.pixelSize: 7
        }

        // 异常体
        Rectangle {
            x: 12
            y: 220
            width: parent.width - 24
            height: 34
            radius: 5
            color: "#0a2436"
            border.width: 1
            border.color: "#17465f"

            StatusChip {
                x: 8
                anchors.verticalCenter: parent.verticalCenter
                text: view.anomalyText()
                chipColor: "#4fd0e8"
                height: 18
            }
            Text {
                anchors.left: parent.left
                anchors.leftMargin: 62
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                text: "中心 (" + view.num(view.anomaly.x, 0) + ", " + view.num(view.anomaly.z, 0) + ") m"
                      + " · " + view.num(view.anomaly.rx, 0) + " × " + view.num(view.anomaly.rz, 0) + " m"
                      + " · Vp " + view.num(view.anomaly.vp, 0) + " · Qp " + view.num(view.anomaly.qp, 0)
                color: "#8fadc2"
                font.family: Theme.fontMono
                font.pixelSize: 7
                elide: Text.ElideRight
            }
        }

        // 校验摘要
        Row {
            x: 12
            y: 264
            spacing: 6

            Repeater {
                model: [ "vs < vp", "层序递增", "参数为正", "Vp 均值 " + view.metricValue("model_vp_mean") ]
                Row {
                    spacing: 4
                    Rectangle {
                        anchors.verticalCenter: parent.verticalCenter
                        width: 4; height: 4; radius: 2
                        color: "#31d98a"
                    }
                    Text {
                        text: modelData
                        color: "#5f8199"
                        font.family: Theme.fontUi
                        font.pixelSize: 7
                    }
                }
            }
        }
    }

    // ── 左栏 · 观测系统与求解参数 ──────────────────────────────
    Rectangle {
        id: solverCard
        y: 310
        width: 440
        height: 306
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "观测系统与求解参数"
            tag: "AppConfig"
            tagColor: "#31d98a"
        }

        Grid {
            x: 12
            y: 46
            width: parent.width - 24
            columns: 2
            columnSpacing: 12
            rowSpacing: 6

            Repeater {
                model: [
                    { "label": "震源", "value": view.kindName(view.source.kind) + " · " + view.num(view.source.f0, 0) + " Hz", "note": "source.kind / f0" },
                    { "label": "震源位置", "value": "(" + view.num(view.source.x, 0) + ", " + view.num(view.source.z, 0) + ") m", "note": "x / z" },
                    { "label": "检波点", "value": view.num(view.receivers.count, 0) + " 道 · " + (view.receivers.component || "—"), "note": view.num(view.receivers.x_start, 0) + "–" + view.num(view.receivers.x_end, 0) + " m" },
                    { "label": "检波深度", "value": view.num(view.receivers.z, 0) + " m", "note": "receiver z" },
                    { "label": "时间采样", "value": view.num(view.solver.nt, 0) + " × " + view.num(Number(view.solver.dt) * 1000, 1) + " ms", "note": (Number(view.solver.nt) * Number(view.solver.dt)).toFixed(3) + " s" },
                    { "label": "空间格式", "value": view.num(view.solver.order, 0) + " 阶 · " + String(view.solver.boundary || "—").toUpperCase(), "note": "order / boundary" },
                    { "label": "数据类型", "value": view.solver.dtype || "—", "note": "solver.dtype" },
                    { "label": "成像模式", "value": String(view.imaging.mode || "—").toUpperCase(), "note": "imaging.mode" }
                ]

                Item {
                    width: (solverCard.width - 36) / 2
                    height: 34

                    Text {
                        x: 0; y: 0
                        text: modelData.label
                        color: "#5f8199"
                        font.family: Theme.fontUi
                        font.pixelSize: 7
                    }
                    Text {
                        x: 0; y: 12
                        width: parent.width
                        text: modelData.value
                        color: "#d9ecf8"
                        font.family: Theme.fontMono
                        font.pixelSize: 10
                        elide: Text.ElideRight
                    }
                    Text {
                        x: 0; y: 25
                        text: modelData.note
                        color: "#3f6479"
                        font.family: Theme.fontMono
                        font.pixelSize: 6
                    }
                }
            }
        }

        Rectangle {
            x: 12
            y: 210
            width: parent.width - 24
            height: 1
            color: "#10384d"
        }

        Text {
            x: 12
            y: 218
            text: "成像条件"
            color: "#7b9fb5"
            font.family: Theme.fontUi
            font.pixelSize: 8
            font.weight: Font.DemiBold
        }

        Row {
            x: 12
            y: 234
            spacing: 6

            Repeater {
                model: [
                    { "text": "PP 互相关", "on": true },
                    { "text": "归一化 " + (view.imaging.normalize ? "ON" : "OFF"), "on": !!view.imaging.normalize },
                    { "text": "Laplacian " + (view.imaging.laplacian ? "ON" : "OFF"), "on": !!view.imaging.laplacian },
                    { "text": "多炮叠加", "on": true }
                ]

                Rectangle {
                    width: 88
                    height: 22
                    radius: 4
                    color: modelData.on ? "#0b3348" : "#0a2233"
                    border.width: 1
                    border.color: modelData.on ? "#23617f" : "#12384e"

                    Text {
                        anchors.centerIn: parent
                        text: modelData.text
                        color: modelData.on ? "#a9e4f7" : "#4f7189"
                        font.family: Theme.fontUi
                        font.pixelSize: 8
                    }
                }
            }
        }

        Row {
            x: 12
            y: 266
            spacing: 6

            Repeater {
                model: view.checks.length >= 4
                       ? [ view.checks[0].value, "CFL " + view.checks[2].value, "PPW " + view.checks[3].value, view.checks[4].value ]
                       : []

                Rectangle {
                    width: 88
                    height: 20
                    radius: 4
                    color: "#08202f"
                    border.width: 1
                    border.color: "#143c53"

                    Text {
                        anchors.centerIn: parent
                        text: modelData
                        color: "#6f9bb2"
                        font.family: Theme.fontMono
                        font.pixelSize: 7
                    }
                }
            }
        }

        Text {
            x: 12
            y: 292
            text: "config/default.json · AppConfig 校验通过"
            color: "#3f6479"
            font.family: Theme.fontMono
            font.pixelSize: 6
        }
    }

    // ── 右栏 · 速度模型 ────────────────────────────────────────
    Rectangle {
        id: previewCard
        x: 450
        width: 290
        height: 290
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "速度模型"
            tag: view.fieldNames[view.fieldIndex] + " · " + view.num(view.grid.nx, 0) + " × " + view.num(view.grid.nz, 0)
            tagColor: "#7fc4df"
        }

        Rectangle {
            x: 8
            y: 42
            width: parent.width - 16
            height: 202
            radius: 5
            color: "#04111c"
            border.width: 1
            border.color: "#0e2c3f"
            clip: true

            Image {
                id: modelImage
                anchors.fill: parent
                anchors.margins: 1
                source: bridge ? (bridge.previews[view.fieldKeys[view.fieldIndex]] || "") : ""
                fillMode: Image.PreserveAspectFit
                smooth: true
                cache: false
                asynchronous: true
            }

            // 异常体标注
            Item {
                visible: view.anomaly.x !== undefined && view.grid.nx !== undefined
                width: 16
                height: 16
                x: parent.width * (Number(view.anomaly.x) - Number(view.grid.x0))
                   / Math.max(1, (Number(view.grid.nx) - 1) * Number(view.grid.dx)) - 8
                y: parent.height * (Number(view.anomaly.z) - Number(view.grid.z0))
                   / Math.max(1, (Number(view.grid.nz) - 1) * Number(view.grid.dz)) - 8

                Rectangle {
                    anchors.fill: parent
                    radius: 8
                    color: "transparent"
                    border.width: 1
                    border.color: "#8fe3ff"
                    opacity: 0.75
                }
                Rectangle {
                    anchors.centerIn: parent
                    width: 24
                    height: 1
                    color: "#8fe3ff"
                    opacity: 0.35
                }
                Rectangle {
                    anchors.centerIn: parent
                    width: 1
                    height: 24
                    color: "#8fe3ff"
                    opacity: 0.35
                }
                Text {
                    x: 18
                    y: -17
                    text: view.anomalyText()
                    color: "#bfe9f5"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
            }

            Text {
                x: 7
                y: 5
                text: "z / m"
                color: "#4f7189"
                font.family: Theme.fontMono
                font.pixelSize: 6
            }
            Text {
                anchors.right: parent.right
                anchors.rightMargin: 7
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 5
                text: "x / m"
                color: "#4f7189"
                font.family: Theme.fontMono
                font.pixelSize: 6
            }
            Text {
                anchors.right: parent.right
                anchors.rightMargin: 7
                y: 5
                text: "0–1000 m"
                color: "#4f7189"
                font.family: Theme.fontMono
                font.pixelSize: 6
            }
        }

        Row {
            x: 8
            y: 250
            spacing: 5

            Repeater {
                model: view.fieldNames
                Rectangle {
                    width: 42
                    height: 22
                    radius: 4
                    color: view.fieldIndex === index ? "#0e3d58" : "#08202f"
                    border.width: 1
                    border.color: view.fieldIndex === index ? "#38c9f5" : "#12384e"

                    Text {
                        anchors.centerIn: parent
                        text: modelData
                        color: view.fieldIndex === index ? "#a9edff" : "#5f8199"
                        font.family: Theme.fontMono
                        font.pixelSize: 8
                    }
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: view.fieldIndex = index
                    }
                }
            }
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 10
            y: 255
            text: view.fieldIndex === 0 ? view.vpRange + " m/s"
                                         : view.fieldIndex === 1 ? view.vsRange + " m/s" : view.rhoRange + " kg/m³"
            color: "#5f8199"
            font.family: Theme.fontMono
            font.pixelSize: 7
        }
    }

    // ── 右栏 · 单炮记录 ────────────────────────────────────────
    ShotRecordCard {
        id: shotCard
        x: 450
        y: 300
        width: 290
        height: 210
        onShowDetails: view.shotDetailOpen = true
    }

    // ── 右栏 · 成像 ────────────────────────────────────────────
    Rectangle {
        id: imageCard
        x: 450
        y: 520
        width: 290
        height: 96
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "PP 成像"
            tag: "零场占位"
            tagColor: "#efa63f"
            divider: false
        }

        Rectangle {
            x: 8
            y: 36
            width: parent.width - 16
            height: 52
            radius: 5
            color: "#04111c"
            border.width: 1
            border.color: "#0e2c3f"
            clip: true

            Repeater {
                model: 7
                Rectangle {
                    x: index * parent.width / 6
                    width: 1
                    height: parent.height
                    color: "#10384d"
                    opacity: 0.45
                }
            }
            Repeater {
                model: 4
                Rectangle {
                    y: index * parent.height / 3
                    width: parent.width
                    height: 1
                    color: "#10384d"
                    opacity: 0.45
                }
            }

            Text {
                anchors.centerIn: parent
                text: "零场 · 等待 ertm/imaging.py 接入"
                color: "#4f7189"
                font.family: Theme.fontMono
                font.pixelSize: 7
            }
        }
    }

    ShotRecordDialog {
        anchors.fill: parent
        visible: view.shotDetailOpen
        onClosed: view.shotDetailOpen = false
    }
}
