import QtQuick
import "../components"
import "../theme/Theme.js" as Theme

/* 求解器源码浏览：模块列表 + 只读代码视图 */
Item {
    id: view

    readonly property var modules: bridge ? bridge.modules : []
    readonly property int readyCount: countState("ready")
    readonly property int placeholderCount: countState("placeholder")
    readonly property int plannedCount: countState("planned")

    function countState(state) {
        var count = 0;
        for (var i = 0; i < modules.length; ++i) {
            if (modules[i].status === state)
                count++;
        }
        return count;
    }

    function openCurrentFile() {
        if (!bridge || !bridge.projectRoot || !bridge.codeFile)
            return;
        var path = bridge.projectRoot.replace(/\\/g, "/") + "/" + bridge.codeFile;
        Qt.openUrlExternally("file:///" + path);
    }

    // ── 模块列表 ────────────────────────────────────────────────
    Rectangle {
        width: 260
        height: parent.height
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        CardHeader {
            width: parent.width
            title: "求解器源码"
            tag: view.modules.length + " 个文件"
            tagColor: "#7fc4df"
        }

        Row {
            x: 10
            y: 44
            spacing: 5
            Rectangle {
                width: 72; height: 19; radius: 3
                color: "#0a2b3e"
                border.width: 1; border.color: "#17465f"
                Text {
                    anchors.centerIn: parent
                    text: view.readyCount + " 已实现"
                    color: "#31d98a"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
            }
            Rectangle {
                width: 58; height: 19; radius: 3
                color: "#2a2113"
                border.width: 1; border.color: "#6e4d1c"
                Text {
                    anchors.centerIn: parent
                    text: view.placeholderCount + " 占位"
                    color: "#efa63f"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
            }
            Rectangle {
                width: 58; height: 19; radius: 3
                color: "#0a2233"
                border.width: 1; border.color: "#12384e"
                Text {
                    anchors.centerIn: parent
                    text: view.plannedCount + " 待建"
                    color: "#6f93aa"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
            }
        }

        ListView {
            id: moduleList
            anchors.top: parent.top
            anchors.topMargin: 72
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.leftMargin: 8
            anchors.rightMargin: 8
            anchors.bottomMargin: 8
            clip: true
            spacing: 1
            model: view.modules
            boundsBehavior: Flickable.StopAtBounds

            delegate: ModuleRow {
                width: moduleList.width
                fileName: modelData.name
                role: modelData.role
                lines: modelData.lines
                state: modelData.status
                selected: bridge ? bridge.codeFile === modelData.path : false
                onClicked: bridge.loadSource(modelData.path)
            }
        }
    }

    // ── 代码视图 ────────────────────────────────────────────────
    Rectangle {
        x: 270
        width: parent.width - 270
        height: parent.height
        radius: 8
        color: "#071a2a"
        border.width: 1
        border.color: "#123a54"

        Item {
            id: codeHeader
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: 56

            Text {
                x: 12
                y: 8
                width: parent.width - 150
                text: bridge ? bridge.codeTitle : "—"
                color: "#e7f4ff"
                font.family: Theme.fontMono
                font.pixelSize: 10
                elide: Text.ElideRight
            }
            Text {
                x: 12
                y: 25
                width: parent.width - 150
                text: bridge ? (bridge.codeRole + "  ·  " + bridge.codeLines + " 行") : ""
                color: "#5f8199"
                font.family: Theme.fontUi
                font.pixelSize: 7
                elide: Text.ElideRight
            }

            StatusChip {
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.top: parent.top
                anchors.topMargin: 8
                text: bridge ? (bridge.codeStatus === "ready" ? "已实现"
                             : bridge.codeStatus === "placeholder" ? "占位实现" : "待接入") : "—"
                chipColor: bridge && bridge.codeStatus === "ready" ? "#31d98a"
                         : bridge && bridge.codeStatus === "placeholder" ? "#efa63f" : "#3c6b83"
                height: 18
            }

            Rectangle {
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 8
                width: 72
                height: 20
                radius: 3
                color: "#0a2a40"
                border.width: 1
                border.color: "#1a4b66"

                Text {
                    anchors.centerIn: parent
                    text: "打开文件"
                    color: "#a9cfe2"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: view.openCurrentFile()
                }
            }

            Rectangle {
                anchors.bottom: parent.bottom
                width: parent.width
                height: 1
                color: "#123a54"
            }
        }

        Flickable {
            id: codeFlick
            anchors.top: codeHeader.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.margins: 8
            clip: true
            contentWidth: Math.max(width, codeText.implicitWidth + 58)
            contentHeight: Math.max(height, codeText.implicitHeight + 16)
            boundsBehavior: Flickable.StopAtBounds
            flickableDirection: Flickable.AutoFlickDirection

            Text {
                id: lineNumbers
                x: 4
                y: 6
                text: {
                    var lines = [];
                    var total = bridge ? bridge.codeLines : 0;
                    for (var i = 1; i <= total; ++i)
                        lines.push(i);
                    return lines.join("\n");
                }
                color: "#31596f"
                font.family: Theme.fontMono
                font.pixelSize: 8
                lineHeight: 1.4
                horizontalAlignment: Text.AlignRight
                width: 38
            }

            Text {
                id: codeText
                x: 48
                y: 6
                text: bridge ? bridge.code : ""
                color: "#a9cbdd"
                font.family: Theme.fontMono
                font.pixelSize: 8
                lineHeight: 1.4
                wrapMode: Text.NoWrap
            }

            Connections {
                target: bridge
                function onCodeChanged() {
                    codeFlick.contentY = 0;
                    codeFlick.contentX = 0;
                }
            }
        }

        Rectangle {
            id: scrollTrack
            visible: codeFlick.contentHeight > codeFlick.height
            anchors.right: parent.right
            anchors.rightMargin: 4
            anchors.top: codeHeader.bottom
            anchors.topMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            width: 3
            radius: 1.5
            color: "#0a2436"

            Rectangle {
                width: parent.width
                radius: 1.5
                color: "#2c6d8c"
                height: Math.max(24, scrollTrack.height * codeFlick.height / codeFlick.contentHeight)
                y: (scrollTrack.height - height)
                   * codeFlick.contentY / Math.max(1, codeFlick.contentHeight - codeFlick.height)
            }
        }
    }
}
