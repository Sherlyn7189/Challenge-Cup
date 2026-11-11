import QtQuick
import "../components"
import "../theme/Theme.js" as Theme

/* Q 补偿：GSLS 黏声反演、补偿前后 QC、增益与稳定性 */
Item {
    id: view

    readonly property var q: bridge ? bridge.q : ({})
    readonly property var previews: bridge ? bridge.qPreviews : ({})
    readonly property var alphas: (q.alpha_mechanisms && q.alpha_mechanisms.length > 0)
                                  ? q.alpha_mechanisms : [0.9, 0.88, 0.0]
    property int modelMode: 0       // 0 Qp, 1 Qs
    property bool reportOpen: false

    function num(v, d) {
        var n = Number(v);
        return isFinite(n) ? n.toFixed(d === undefined ? 2 : d) : "—";
    }
    function pct(v, d) {
        var n = Number(v);
        return isFinite(n) ? (n * 100).toFixed(d === undefined ? 1 : d) + "%" : "—";
    }

    // ── 参数条 ──────────────────────────────────────────────────
    Rectangle {
        width: parent.width
        height: 84
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "Q 补偿参数"
            tag: view.q.available ? "GSLS · 真实运行" : "未运行"
            tagColor: view.q.available ? "#31d98a" : "#efa63f"
        }

        Row {
            x: 10
            y: 44
            spacing: 6

            Repeater {
                model: [
                    { "label": "αc", "value": view.num(view.q.alpha_c, 2), "unit": "补偿强度" },
                    { "label": "f_ref", "value": view.num(view.q.f_ref, 0), "unit": "Hz" },
                    { "label": "Gain cap", "value": view.num(view.q.gain_cap, 0), "unit": "倍" },
                    { "label": "GSLS", "value": view.num(view.q.n_mechanisms, 0), "unit": "机制" },
                    { "label": "f_max", "value": view.num(view.q.f_max, 0), "unit": "Hz" },
                    { "label": "增益预算", "value": view.num(view.q.gain_budget_db, 0), "unit": "dB" }
                ]

                Rectangle {
                    width: 112
                    height: 32
                    radius: 4
                    color: "#082032"
                    border.width: 1
                    border.color: "#12384e"

                    Text {
                        x: 8; y: 4
                        text: modelData.label
                        color: "#5f8199"
                        font.family: Theme.fontMono
                        font.pixelSize: 6
                    }
                    Text {
                        x: 8; y: 15
                        text: modelData.value
                        color: "#d9ecf8"
                        font.family: Theme.fontMono
                        font.pixelSize: 10
                        font.weight: Font.DemiBold
                    }
                    Text {
                        anchors.right: parent.right
                        anchors.rightMargin: 7
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 3
                        text: modelData.unit
                        color: "#3f6479"
                        font.family: Theme.fontUi
                        font.pixelSize: 6
                    }
                }
            }
        }
    }

    // ── Q 模型 ──────────────────────────────────────────────────
    Rectangle {
        id: qModelCard
        y: 94
        width: 440
        height: 300
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "衰减模型"
            tag: view.modelMode === 0 ? "Qp" : "Qs"
            tagColor: "#efa63f"
        }

        Row {
            x: 8
            y: 44
            spacing: 5

            Repeater {
                model: [ "Qp", "Qs" ]
                Rectangle {
                    width: 56
                    height: 22
                    radius: 4
                    color: view.modelMode === index ? "#3a2a12" : "#08202f"
                    border.width: 1
                    border.color: view.modelMode === index ? "#c98a32" : "#12384e"

                    Text {
                        anchors.centerIn: parent
                        text: modelData
                        color: view.modelMode === index ? "#f2c078" : "#5f8199"
                        font.family: Theme.fontMono
                        font.pixelSize: 8
                    }
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: view.modelMode = index
                    }
                }
            }
        }

        Rectangle {
            x: 8
            y: 72
            width: parent.width - 16
            height: 184
            radius: 5
            color: "#04111c"
            border.width: 1
            border.color: "#0e2c3f"
            clip: true

            Image {
                anchors.fill: parent
                anchors.margins: 1
                source: bridge ? (bridge.qPreviews[view.modelMode === 0 ? "qp" : "qs"] || "") : ""
                fillMode: Image.Stretch
                smooth: true
                cache: false
                asynchronous: true
            }

            Text {
                x: 6; y: 4
                text: "z / m"
                color: "#4f7189"
                font.family: Theme.fontMono
                font.pixelSize: 6
            }
            Text {
                anchors.right: parent.right
                anchors.rightMargin: 6
                y: 4
                text: "x / m"
                color: "#4f7189"
                font.family: Theme.fontMono
                font.pixelSize: 6
            }

            Rectangle {
                x: 6
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 6
                width: view.modelMode === 0 ? 92 : 92
                height: 18
                radius: 3
                color: "#03101a"
                opacity: 0.85

                Text {
                    anchors.centerIn: parent
                    text: "Qp " + view.num(view.q.qp_min, 0) + "–" + view.num(view.q.qp_max, 0)
                    visible: view.modelMode === 0
                    color: "#c9a56c"
                    font.family: Theme.fontMono
                    font.pixelSize: 6
                }
                Text {
                    anchors.centerIn: parent
                    text: "Qs " + view.num(view.q.qs_min, 0) + "–" + view.num(view.q.qs_max, 0)
                    visible: view.modelMode === 1
                    color: "#c9a56c"
                    font.family: Theme.fontMono
                    font.pixelSize: 6
                }
            }
        }

        Text {
            x: 8
            y: 263
            text: "GSLS 常 Q 模型 · " + view.num(view.q.n_mechanisms, 0) + " 机制 · Qp/Q 拟合误差 " + Number(view.q.q_fit_error || 0).toExponential(2)
            color: "#5f8199"
            font.family: Theme.fontMono
            font.pixelSize: 6
        }
        Text {
            x: 8
            y: 278
            text: "补偿作用于反传边界（接收侧），源侧保持物理衰减"
            color: "#3f6479"
            font.family: Theme.fontUi
            font.pixelSize: 6
        }
    }

    // ── 补偿后成像 ──────────────────────────────────────────────
    Rectangle {
        id: imageCard
        x: 450
        y: 94
        width: 290
        height: 300
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "补偿后成像"
            tag: "PP · viscoacoustic"
            tagColor: "#31d98a"
        }

        Rectangle {
            x: 8
            y: 44
            width: parent.width - 16
            height: 184
            radius: 5
            color: "#04111c"
            border.width: 1
            border.color: "#0e2c3f"
            clip: true

            Image {
                anchors.fill: parent
                anchors.margins: 1
                source: bridge ? (bridge.qPreviews.image || "") : ""
                fillMode: Image.Stretch
                smooth: true
                cache: false
                asynchronous: true
            }
            Text {
                anchors.centerIn: parent
                visible: !view.q.available
                text: "运行 Q 补偿后生成"
                color: "#5f8199"
                font.family: Theme.fontUi
                font.pixelSize: 8
            }
        }

        Text {
            x: 8
            y: 236
            text: "深部窗口 " + view.num(view.q.band_top, 0) + "–" + view.num(view.q.band_bottom, 0) + " m"
            color: "#5f8199"
            font.family: Theme.fontMono
            font.pixelSize: 6
        }
        Text {
            x: 8
            y: 250
            text: "频带 RMS " + view.num(view.q.baseline_rms, 4) + " → " + view.num(view.q.compensated_rms, 4)
            color: "#8fe3ff"
            font.family: Theme.fontMono
            font.pixelSize: 6
        }
        Text {
            x: 8
            y: 264
            text: "深/浅比 " + view.num(view.q.baseline_ratio, 3) + " → " + view.num(view.q.compensated_ratio, 3)
            color: "#5ff0ad"
            font.family: Theme.fontMono
            font.pixelSize: 6
        }
        Text {
            x: 8
            y: 278
            text: "与未补偿图像相关系数 " + view.num(view.q.correlation, 3)
            color: "#3f6479"
            font.family: Theme.fontMono
            font.pixelSize: 6
        }
    }

    // ── 增益曲线 ────────────────────────────────────────────────
    Rectangle {
        id: gainCard
        y: 404
        width: 440
        height: 212
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "补偿增益曲线"
            tag: "dB / (t/Q)"
            tagColor: "#7fc4df"
        }

        Canvas {
            id: chart
            x: 12
            y: 46
            width: parent.width - 24
            height: 128

            onPaint: {
                var ctx = getContext("2d");
                ctx.clearRect(0, 0, width, height);
                var left = 38, right = 8, top = 8, bottom = 20;
                var w = width - left - right;
                var h = height - top - bottom;
                var tmax = 0.6;
                var ymax = 450;
                var i;
                ctx.strokeStyle = "rgba(60, 120, 150, 0.25)";
                ctx.lineWidth = 1;
                for (i = 0; i <= 4; ++i) {
                    var gy = top + h - (i / 4) * h;
                    ctx.beginPath();
                    ctx.moveTo(left, gy + 0.5);
                    ctx.lineTo(left + w, gy + 0.5);
                    ctx.stroke();
                    ctx.fillStyle = "#4f7189";
                    ctx.font = "7px Consolas";
                    ctx.textAlign = "right";
                    ctx.fillText(String(i * 112), left - 5, gy + 3);
                }
                for (i = 0; i <= 6; ++i) {
                    var gx = left + (i / 6) * w;
                    ctx.beginPath();
                    ctx.moveTo(gx + 0.5, top);
                    ctx.lineTo(gx + 0.5, top + h);
                    ctx.stroke();
                    ctx.fillStyle = "#4f7189";
                    ctx.textAlign = "center";
                    ctx.fillText((i * 0.1).toFixed(1), gx, top + h + 13);
                }
                ctx.strokeStyle = "#2d6079";
                ctx.beginPath();
                ctx.moveTo(left, top);
                ctx.lineTo(left, top + h);
                ctx.lineTo(left + w, top + h);
                ctx.stroke();

                var freqs = [8, 30, 115];
                var colors = ["#38c9f5", "#efa63f", "#31d98a"];
                for (var c = 0; c < 3; ++c) {
                    var alpha = Number(view.alphas[c]) || 0;
                    ctx.strokeStyle = colors[c];
                    ctx.lineWidth = 1.6;
                    ctx.beginPath();
                    for (i = 0; i <= 60; ++i) {
                        var t = (i / 60) * tmax;
                        var db = alpha * 20 * Math.PI * freqs[c] * t / Math.LN10;
                        var px = left + (t / tmax) * w;
                        var py = top + h - Math.min(1, db / ymax) * h;
                        if (i === 0)
                            ctx.moveTo(px, py);
                        else
                            ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                }
                ctx.fillStyle = "#5f8199";
                ctx.font = "7px Consolas";
                ctx.textAlign = "left";
                ctx.fillText("dB", 4, top + 4);
                ctx.textAlign = "right";
                ctx.fillText("t / s", left + w, height - 3);
            }

            Connections {
                target: bridge
                function onQChanged() { chart.requestPaint(); }
            }
        }

        Row {
            x: 42
            y: 178
            spacing: 14

            Repeater {
                model: [
                    { "label": "f=8 Hz α=" + view.num(view.alphas[0], 2), "color": "#38c9f5" },
                    { "label": "f=30 Hz α=" + view.num(view.alphas[1], 2), "color": "#efa63f" },
                    { "label": "f=115 Hz α=" + view.num(view.alphas[2], 2), "color": "#31d98a" }
                ]

                Row {
                    spacing: 4
                    Rectangle {
                        anchors.verticalCenter: parent.verticalCenter
                        width: 10
                        height: 2
                        color: modelData.color
                    }
                    Text {
                        text: modelData.label
                        color: "#5f8199"
                        font.family: Theme.fontMono
                        font.pixelSize: 6
                    }
                }
            }
        }
    }

    // ── QC 与运行 ───────────────────────────────────────────────
    Rectangle {
        id: qcCard
        x: 450
        y: 404
        width: 290
        height: 212
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "QC 与稳定性"
            tag: "qc.py"
            tagColor: "#7fc4df"
        }

        Repeater {
            model: [
                { "label": "频带振幅增益", "value": view.num(view.q.band_gain, 2) + "×" },
                { "label": "深/浅能量比", "value": view.num(view.q.baseline_ratio, 2) + " → " + view.num(view.q.compensated_ratio, 2) },
                { "label": "相关系数", "value": view.num(view.q.correlation, 3) },
                { "label": "差值 RMS", "value": view.num(view.q.difference_rms, 3) },
                { "label": "限幅 / 低通步", "value": view.num(view.q.clipped_steps, 0) + " / " + view.num(view.q.lowpass_steps, 0) }
            ]

            Item {
                x: 12
                y: 44 + index * 22
                width: qcCard.width - 24
                height: 21

                Rectangle {
                    anchors.bottom: parent.bottom
                    width: parent.width
                    height: 1
                    color: "#0d2d41"
                    opacity: 0.7
                }
                Text {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData.label
                    color: "#8fadc2"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
                Text {
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData.value
                    color: "#8fe3ff"
                    font.family: Theme.fontMono
                    font.pixelSize: 7
                }
            }
        }

        Row {
            x: 12
            y: 166
            spacing: 8

            LabButton {
                width: 128
                height: 28
                text: bridge && bridge.running ? "运行中…" : "运行 Q 补偿"
                variant: "primary"
                enabled: bridge ? !bridge.running : false
                onClicked: if (bridge) bridge.runQComp()
            }
            LabButton {
                width: 128
                height: 28
                text: "补偿对比图"
                variant: "ghost"
                onClicked: view.reportOpen = true
            }
        }
    }

    // ── QC 报告图弹层 ───────────────────────────────────────────
    Item {
        anchors.fill: parent
        z: 100
        visible: view.reportOpen

        Rectangle {
            anchors.fill: parent
            color: "#01070e"
            opacity: 0.72
            MouseArea {
                anchors.fill: parent
                onClicked: view.reportOpen = false
            }
        }

        Rectangle {
            width: 700
            height: 420
            anchors.centerIn: parent
            radius: 8
            color: "#f7fafc"
            border.width: 1
            border.color: "#c8d6de"

            MouseArea { anchors.fill: parent }

            Text {
                x: 18
                y: 12
                text: "Q 补偿：补偿前 / 补偿后 / 差值（ertm/qc.py 输出）"
                color: "#123a54"
                font.family: Theme.fontUi
                font.pixelSize: 11
                font.weight: Font.DemiBold
            }
            Text {
                anchors.right: parent.right
                anchors.rightMargin: 16
                y: 10
                text: "×"
                color: "#42657b"
                font.family: Theme.fontUi
                font.pixelSize: 18
                MouseArea {
                    anchors.fill: parent
                    anchors.margins: -6
                    cursorShape: Qt.PointingHandCursor
                    onClicked: view.reportOpen = false
                }
            }
            Image {
                x: 14
                y: 40
                width: parent.width - 28
                height: 340
                source: bridge ? (bridge.qPreviews.compare || "") : ""
                fillMode: Image.PreserveAspectFit
                smooth: true
                cache: false
                asynchronous: true
            }
            Text {
                x: 14
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 10
                text: "深部窗口 " + view.num(view.q.band_top, 0) + "–" + view.num(view.q.band_bottom, 0)
                      + " m · 频带增益 " + view.num(view.q.band_gain, 2) + "× · 深/浅比 "
                      + view.num(view.q.baseline_ratio, 2) + " → " + view.num(view.q.compensated_ratio, 2)
                color: "#5f8199"
                font.family: Theme.fontMono
                font.pixelSize: 7
            }
        }
    }
}
