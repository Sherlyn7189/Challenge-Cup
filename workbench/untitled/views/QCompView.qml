import QtQuick
import "../components"
import "../theme/Theme.js" as Theme

/* Q 补偿：参数、增益曲线与稳定性检查 */
Item {
    id: view

    readonly property var cfg: bridge ? bridge.cfg : ({})
    readonly property var model: cfg.model || ({})
    readonly property var source: cfg.source || ({})
    readonly property var solver: cfg.solver || ({})
    readonly property real f0: Number(source.f0) || 20

    function num(value, digits) {
        if (value === undefined || value === null || isNaN(Number(value)))
            return "—";
        return Number(value).toFixed(digits === undefined ? 0 : digits);
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
            tag: "GSLS · 待算法接入"
            tagColor: "#efa63f"
        }

        Row {
            x: 10
            y: 44
            spacing: 6

            Repeater {
                model: [
                    { "label": "Qp", "value": view.num(view.model.background_qp, 0), "unit": "背景" },
                    { "label": "Qs", "value": view.num(view.model.background_qs, 0), "unit": "背景" },
                    { "label": "GSLS", "value": "3", "unit": "机制" },
                    { "label": "Gmax", "value": "30", "unit": "增益上限" },
                    { "label": "αc", "value": "0.85", "unit": "补偿强度" },
                    { "label": "LPF", "value": "0.8", "unit": "Nyquist" }
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
                        font.pixelSize: 7
                    }
                    Text {
                        x: 8; y: 14
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

    // ── 增益曲线 ────────────────────────────────────────────────
    Rectangle {
        id: gainCard
        y: 94
        width: 440
        height: 300
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "增益曲线"
            tag: "G(t) = exp(π f t / Q)"
            tagColor: "#7fc4df"
        }

        Canvas {
            id: chart
            x: 12
            y: 46
            width: parent.width - 24
            height: 218

            onPaint: {
                var ctx = getContext("2d");
                ctx.clearRect(0, 0, width, height);

                var left = 36, right = 10, top = 10, bottom = 26;
                var plotW = width - left - right;
                var plotH = height - top - bottom;
                var tMax = 0.6;
                var dbMax = 10.0;
                var i;

                // 网格
                ctx.strokeStyle = "rgba(60, 120, 150, 0.25)";
                ctx.lineWidth = 1;
                for (i = 0; i <= 5; ++i) {
                    var gy = top + plotH - (i / 5) * plotH;
                    ctx.beginPath();
                    ctx.moveTo(left, gy + 0.5);
                    ctx.lineTo(left + plotW, gy + 0.5);
                    ctx.stroke();

                    ctx.fillStyle = "#4f7189";
                    ctx.font = "8px Consolas";
                    ctx.textAlign = "right";
                    ctx.fillText(String(i * 2), left - 6, gy + 3);
                }
                for (i = 0; i <= 6; ++i) {
                    var gx = left + (i / 6) * plotW;
                    ctx.beginPath();
                    ctx.moveTo(gx + 0.5, top);
                    ctx.lineTo(gx + 0.5, top + plotH);
                    ctx.stroke();

                    ctx.fillStyle = "#4f7189";
                    ctx.font = "8px Consolas";
                    ctx.textAlign = "center";
                    ctx.fillText((i * 0.1).toFixed(1), gx, top + plotH + 14);
                }

                // 坐标轴
                ctx.strokeStyle = "#2d6079";
                ctx.beginPath();
                ctx.moveTo(left, top);
                ctx.lineTo(left, top + plotH);
                ctx.lineTo(left + plotW, top + plotH);
                ctx.stroke();

                ctx.fillStyle = "#5f8199";
                ctx.font = "8px Consolas";
                ctx.textAlign = "left";
                ctx.fillText("dB", 4, top + 4);
                ctx.textAlign = "right";
                ctx.fillText("t / s", left + plotW, height - 4);

                // 曲线：G_dB = 20*pi*f*t/(Q*ln10)
                var curves = [
                    { "q": 40, "color": "#efa63f" },
                    { "q": 80, "color": "#38c9f5" },
                    { "q": 120, "color": "#31d98a" }
                ];
                for (var c = 0; c < curves.length; ++c) {
                    var curve = curves[c];
                    ctx.strokeStyle = curve.color;
                    ctx.lineWidth = 1.6;
                    ctx.beginPath();
                    for (i = 0; i <= 60; ++i) {
                        var t = (i / 60) * tMax;
                        var db = 20 * Math.PI * view.f0 * t / (curve.q * Math.LN10);
                        var px = left + (t / tMax) * plotW;
                        var py = top + plotH - (Math.min(db, dbMax) / dbMax) * plotH;
                        if (i === 0)
                            ctx.moveTo(px, py);
                        else
                            ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                }

                // Gmax 参考线（30 倍 ≈ 29.5 dB，超出图示范围，仅标注）
                ctx.setLineDash([4, 4]);
                ctx.strokeStyle = "#6b3f3f";
                ctx.beginPath();
                ctx.moveTo(left, top + 2);
                ctx.lineTo(left + plotW, top + 2);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.fillStyle = "#8a5a5a";
                ctx.font = "7px Consolas";
                ctx.textAlign = "right";
                ctx.fillText("Gmax 30 ≈ 29.5 dB（超出图示）", left + plotW, top + 10);
            }
        }

        Row {
            x: 36
            y: 270
            spacing: 16

            Repeater {
                model: [
                    { "label": "Q = 40", "color": "#efa63f" },
                    { "label": "Q = 80", "color": "#38c9f5" },
                    { "label": "Q = 120", "color": "#31d98a" }
                ]

                Row {
                    spacing: 5
                    Rectangle {
                        anchors.verticalCenter: parent.verticalCenter
                        width: 12
                        height: 2
                        color: modelData.color
                    }
                    Text {
                        text: modelData.label
                        color: "#5f8199"
                        font.family: Theme.fontMono
                        font.pixelSize: 7
                    }
                }
            }
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 12
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            text: "f0 = " + view.num(view.f0, 0) + " Hz"
            color: "#3f6479"
            font.family: Theme.fontMono
            font.pixelSize: 7
        }
    }

    // ── 稳定性检查 ──────────────────────────────────────────────
    Rectangle {
        id: stability
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
            title: "稳定性检查"
            tag: "待运行"
            tagColor: "#efa63f"
        }

        Repeater {
            model: [
                { "label": "离散伴随点积", "value": "未运行", "state": "warn" },
                { "label": "最大增益", "value": "—", "state": "muted" },
                { "label": "时变限幅", "value": "Gmax 30", "state": "ok" },
                { "label": "NaN / 溢出", "value": "—", "state": "muted" },
                { "label": "补偿强度 αc", "value": "0.85", "state": "ok" },
                { "label": "GSLS 机制", "value": "3 组", "state": "ok" }
            ]

            Item {
                x: 14
                y: 46 + index * 31
                width: stability.width - 28
                height: 30

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
                    font.pixelSize: 8
                }
                Rectangle {
                    anchors.right: valueText.left
                    anchors.rightMargin: 8
                    anchors.verticalCenter: parent.verticalCenter
                    width: 5
                    height: 5
                    radius: 2.5
                    color: modelData.state === "warn" ? "#efa63f"
                         : modelData.state === "ok" ? "#31d98a" : "#3c6b83"
                }
                Text {
                    id: valueText
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData.value
                    color: modelData.state === "warn" ? "#efa63f"
                         : modelData.state === "ok" ? "#b9d7e8" : "#5f8199"
                    font.family: Theme.fontMono
                    font.pixelSize: 8
                }
            }
        }

        Text {
            x: 14
            y: 250
            width: parent.width - 28
            text: "Q=∞ 时须与弹性 RTM 逐点一致；补偿后振幅/频谱改善，且不得出现 NaN。"
            wrapMode: Text.WordWrap
            color: "#3f6479"
            font.family: Theme.fontUi
            font.pixelSize: 7
            lineHeight: 1.4
        }
    }

    // ── 结果对比 ────────────────────────────────────────────────
    FieldPane {
        x: 0
        y: 406
        width: (view.width - 10) / 2
        height: 210
        title: "Q = ∞ 弹性 RTM 参考"
        tag: "参考"
        tagColor: "#3c6b83"
        placeholderText: "弹性 RTM 参考待接入"
        placeholderNote: "Q = infinity"
        footerNote: "用于与补偿结果逐点对比"
    }
    FieldPane {
        x: (view.width - 10) / 2 + 10
        y: 406
        width: (view.width - 10) / 2
        height: 210
        title: "Q 补偿后成像"
        tag: "待接入"
        tagColor: "#efa63f"
        placeholderText: "Q 补偿结果待接入"
        placeholderNote: "ertm/migration/qcomp.py"
        footerNote: "增益限幅 + 时变低通"
    }
}
