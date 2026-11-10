import QtQuick
import "../theme/Theme.js" as Theme

/* 右侧信息栏：板块说明 + 运行状态 + 验收指标 + 日志 */
Item {
    id: panel

    property var backend: null
    property var section: ({})

    CardBackground { anchors.fill: parent }

    readonly property string stateColor: !panel.backend ? "#6f93aa"
        : panel.backend.running ? "#38c9f5"
        : panel.backend.runState === "完成" ? "#31d98a"
        : panel.backend.runState.indexOf("失败") >= 0 ? "#ef5f6b"
        : "#6f93aa"

    // ── 当前板块 ────────────────────────────────────────────────
    Item {
        id: sectionBlock
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 150

        Text {
            x: 14
            y: 11
            text: panel.section.eyebrow || "SECTION"
            color: "#4f7d97"
            font.family: Theme.fontMono
            font.pixelSize: 7
            font.letterSpacing: 1.2
        }

        StatusChip {
            visible: !!panel.section.tag
            anchors.right: parent.right
            anchors.rightMargin: 12
            anchors.top: parent.top
            anchors.topMargin: 8
            text: panel.section.tag || ""
            chipColor: panel.section.tagColor || Theme.accentSoft
            height: 17
        }

        Text {
            x: 14
            y: 24
            text: panel.section.title || "—"
            color: "#f0faff"
            font.family: Theme.fontUi
            font.pixelSize: 15
            font.weight: Font.DemiBold
            font.letterSpacing: 0.4
        }

        Text {
            x: 14
            y: 49
            width: parent.width - 28
            height: 46
            text: panel.section.desc || ""
            wrapMode: Text.WordWrap
            color: "#8fadc2"
            font.family: Theme.fontUi
            font.pixelSize: 9
            lineHeight: 1.45
            elide: Text.ElideRight
        }

        Rectangle {
            x: 14
            y: 104
            width: parent.width - 28
            height: 32
            radius: 5
            color: "#071e2d"
            border.width: 1
            border.color: "#16445c"

            Rectangle {
                x: 7
                anchors.verticalCenter: parent.verticalCenter
                width: 2
                height: 16
                radius: 1
                color: Theme.accent
            }

            Text {
                anchors.left: parent.left
                anchors.leftMargin: 15
                anchors.right: parent.right
                anchors.rightMargin: 7
                anchors.verticalCenter: parent.verticalCenter
                text: panel.section.formula || ""
                color: "#8fe3ff"
                font.family: Theme.fontMono
                font.pixelSize: 8
                elide: Text.ElideRight
            }
        }
    }

    Rectangle {
        anchors.top: sectionBlock.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.leftMargin: 12
        anchors.rightMargin: 12
        height: 1
        color: "#123a54"
    }

    // ── 运行状态 ────────────────────────────────────────────────
    Item {
        id: runBlock
        anchors.top: sectionBlock.bottom
        anchors.topMargin: 10
        anchors.left: parent.left
        anchors.right: parent.right
        height: 134

        Text {
            x: 14
            y: 0
            text: "运行状态"
            color: "#7b9fb5"
            font.family: Theme.fontUi
            font.pixelSize: 9
            font.weight: Font.DemiBold
            font.letterSpacing: 0.5
        }

        Rectangle {
            x: 14
            y: 22
            width: 6
            height: 6
            radius: 3
            color: panel.stateColor

            SequentialAnimation on opacity {
                running: panel.backend ? panel.backend.running : false
                loops: Animation.Infinite
                NumberAnimation { to: 0.25; duration: 600 }
                NumberAnimation { to: 1.0; duration: 600 }
            }
        }

        Text {
            x: 28
            y: 16
            text: panel.backend ? panel.backend.runState : "空闲"
            color: panel.stateColor
            font.family: Theme.fontUi
            font.pixelSize: 16
            font.weight: Font.DemiBold
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 14
            y: 20
            text: panel.backend ? panel.backend.projectVersion : "—"
            color: "#4f7189"
            font.family: Theme.fontMono
            font.pixelSize: 7
        }

        Text {
            x: 14
            y: 45
            width: parent.width - 28
            text: panel.backend ? panel.backend.projectName + "  ·  " + panel.backend.projectVersion : "—"
            color: "#6f93aa"
            font.family: Theme.fontUi
            font.pixelSize: 8
            elide: Text.ElideRight
        }

        Rectangle {
            x: 14
            y: 62
            width: parent.width - 28
            height: 4
            radius: 2
            color: "#0b2b3d"

            Rectangle {
                width: parent.width * (panel.backend ? panel.backend.progress : 0)
                height: parent.height
                radius: 2
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "#1d84b5" }
                    GradientStop { position: 1.0; color: "#38c9f5" }
                }
                Behavior on width { NumberAnimation { duration: 300 } }
            }
        }

        Text {
            x: 14
            y: 71
            text: "最近运行"
            color: "#4f7189"
            font.family: Theme.fontUi
            font.pixelSize: 8
        }
        Text {
            x: 14
            y: 84
            text: panel.backend ? panel.backend.lastRun : "—"
            color: "#a9c6d9"
            font.family: Theme.fontMono
            font.pixelSize: 8
        }
        Text {
            x: parent.width - 14 - width
            y: 84
            text: panel.backend ? panel.backend.elapsed.toFixed(2) + " s" : "—"
            color: "#8fe3ff"
            font.family: Theme.fontMono
            font.pixelSize: 8
        }

        Rectangle {
            x: 14
            y: 104
            width: 76
            height: 19
            radius: 4
            color: "#2a2113"
            border.width: 1
            border.color: "#6e4d1c"

            Text {
                anchors.centerIn: parent
                text: panel.backend && !panel.backend.placeholder ? "物理结果" : "占位实现"
                color: "#efa63f"
                font.family: Theme.fontUi
                font.pixelSize: 8
            }
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 14
            y: 108
            text: panel.backend && panel.backend.running ? "执行中…" : "main.py"
            color: "#4f7189"
            font.family: Theme.fontMono
            font.pixelSize: 7
        }
    }

    Rectangle {
        anchors.top: runBlock.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.leftMargin: 12
        anchors.rightMargin: 12
        height: 1
        color: "#123a54"
    }

    // ── 验收指标 ────────────────────────────────────────────────
    Item {
        id: acceptance
        anchors.top: runBlock.bottom
        anchors.topMargin: 10
        anchors.left: parent.left
        anchors.right: parent.right
        height: 154

        Text {
            x: 14
            y: 0
            text: "验收指标"
            color: "#7b9fb5"
            font.family: Theme.fontUi
            font.pixelSize: 9
            font.weight: Font.DemiBold
            font.letterSpacing: 0.5
        }
        Text {
            anchors.right: parent.right
            anchors.rightMargin: 14
            y: 2
            text: "算法接入后自动刷新"
            color: "#3f6479"
            font.family: Theme.fontUi
            font.pixelSize: 7
        }

        Repeater {
            model: [
                { "label": "走时误差", "target": "< 1%", "state": "待验证", "color": "#efa63f" },
                { "label": "界面归位", "target": "< 1 网格", "state": "待验证", "color": "#efa63f" },
                { "label": "CPML 反射", "target": "< -60 dB", "state": "待验证", "color": "#efa63f" },
                { "label": "伴随点积", "target": "< 1e-10", "state": "待接入", "color": "#3c6b83" }
            ]

            Item {
                x: 14
                y: 24 + index * 31
                width: acceptance.width - 28
                height: 29

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
                Text {
                    anchors.right: statusText.left
                    anchors.rightMargin: 10
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData.target
                    color: "#5f8199"
                    font.family: Theme.fontMono
                    font.pixelSize: 8
                }
                Text {
                    id: statusText
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData.state
                    color: modelData.color
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
            }
        }
    }

    Rectangle {
        anchors.top: acceptance.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.leftMargin: 12
        anchors.rightMargin: 12
        height: 1
        color: "#123a54"
    }

    // ── 日志 ────────────────────────────────────────────────────
    Item {
        id: logBlock
        anchors.top: acceptance.bottom
        anchors.topMargin: 10
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 12

        Text {
            x: 14
            y: 0
            text: "运行日志"
            color: "#7b9fb5"
            font.family: Theme.fontUi
            font.pixelSize: 9
            font.weight: Font.DemiBold
            font.letterSpacing: 0.5
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 14
            y: 1
            text: "清空"
            color: "#5f8199"
            font.family: Theme.fontUi
            font.pixelSize: 8

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: if (panel.backend) panel.backend.clearLog()
            }
        }

        Rectangle {
            id: logBox
            anchors.top: parent.top
            anchors.topMargin: 20
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            anchors.bottom: openBtn.top
            anchors.bottomMargin: 8
            radius: 5
            color: "#04121c"
            border.width: 1
            border.color: "#0f3448"
            clip: true

            function tail(text, count) {
                var lines = text.split("\n");
                var output = [];
                for (var i = 0; i < lines.length; ++i) {
                    if (lines[i].length > 0)
                        output.push(lines[i]);
                }
                return output.slice(Math.max(0, output.length - count)).join("\n");
            }

            Text {
                anchors.fill: parent
                anchors.margins: 7
                text: panel.backend ? logBox.tail(panel.backend.log, 8) : ""
                color: "#6f9bb2"
                font.family: Theme.fontMono
                font.pixelSize: 7
                lineHeight: 1.35
                wrapMode: Text.WrapAnywhere
                clip: true
            }
        }

        LabButton {
            id: openBtn
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            anchors.bottom: parent.bottom
            height: 28
            text: "打开输出目录"
            variant: "ghost"
            onClicked: if (panel.backend) panel.backend.openOutput()
        }
    }
}
