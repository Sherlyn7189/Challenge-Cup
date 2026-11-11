import QtQuick
import "../theme/Theme.js" as Theme

/* 深色数据图窗：真实预览图 + 占位态 */
Rectangle {
    id: pane

    property string title: ""
    property string tag: ""
    property color tagColor: Theme.accentSoft
    property string source: ""
    property string placeholderText: "等待算法接入"
    property string placeholderNote: ""
    property string footerNote: ""
    property string cornerText: ""
    property bool showGrid: true

    readonly property bool hasImage: source.length > 0 && image.status === Image.Ready

    radius: 7
    color: "#071a2a"
    border.width: 1
    border.color: "#123a54"
    clip: true

    // ── 标题 ────────────────────────────────────────────────────
    Item {
        id: top
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 30

        Text {
            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.verticalCenter: parent.verticalCenter
            text: pane.title
            color: "#c8e2f1"
            font.family: Theme.fontUi
            font.pixelSize: 9
            font.weight: Font.DemiBold
            font.letterSpacing: 0.3
        }

        StatusChip {
            visible: pane.tag.length > 0
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.verticalCenter: parent.verticalCenter
            text: pane.tag
            chipColor: pane.tagColor
            height: 17
        }

        Rectangle {
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            height: 1
            color: "#10384f"
        }
    }

    // ── 图像 / 占位 ─────────────────────────────────────────────
    Rectangle {
        id: imageBox
        anchors.top: top.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: footer.visible ? footer.top : parent.bottom
        anchors.margins: 6
        color: "#04111c"
        border.width: 1
        border.color: "#0e2c3f"
        clip: true

        Image {
            id: image
            anchors.fill: parent
            anchors.margins: 1
            source: pane.source
            fillMode: Image.PreserveAspectFit
            smooth: true
            cache: false
            asynchronous: true
            visible: pane.hasImage
        }

        // 占位网格
        Item {
            anchors.fill: parent
            visible: !pane.hasImage && pane.showGrid
            opacity: 0.55

            Repeater {
                model: 9
                Rectangle {
                    x: index * imageBox.width / 8
                    width: 1
                    height: parent.height
                    color: "#123a50"
                    opacity: 0.5
                }
            }
            Repeater {
                model: 7
                Rectangle {
                    y: index * imageBox.height / 6
                    width: parent.width
                    height: 1
                    color: "#123a50"
                    opacity: 0.5
                }
            }
        }

        // 占位文案
        Column {
            anchors.centerIn: parent
            spacing: 5
            visible: !pane.hasImage

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "∿"
                color: "#2f7d9e"
                font.family: Theme.fontMono
                font.pixelSize: 28
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: pane.placeholderText
                color: "#6f93aa"
                font.family: Theme.fontUi
                font.pixelSize: 10
                font.letterSpacing: 0.5
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                width: imageBox.width - 24
                horizontalAlignment: Text.AlignHCenter
                text: pane.placeholderNote
                color: "#41647b"
                font.family: Theme.fontMono
                font.pixelSize: 7
                wrapMode: Text.WordWrap
                visible: pane.placeholderNote.length > 0
            }
        }

        Text {
            anchors.left: parent.left
            anchors.leftMargin: 6
            anchors.top: parent.top
            anchors.topMargin: 5
            visible: !pane.hasImage
            text: "PLACEHOLDER"
            color: "#2c5a72"
            font.family: Theme.fontMono
            font.pixelSize: 6
            font.letterSpacing: 1.2
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 6
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 5
            visible: pane.cornerText.length > 0 && pane.hasImage
            text: pane.cornerText
            color: "#7fb2c9"
            font.family: Theme.fontMono
            font.pixelSize: 7
        }
    }

    // ── 底部说明 ────────────────────────────────────────────────
    Item {
        id: footer
        visible: pane.footerNote.length > 0
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 22

        Text {
            anchors.left: parent.left
            anchors.leftMargin: 9
            anchors.right: parent.right
            anchors.rightMargin: 9
            anchors.verticalCenter: parent.verticalCenter
            text: pane.footerNote
            color: "#4f7189"
            font.family: Theme.fontUi
            font.pixelSize: 8
            elide: Text.ElideRight
        }
    }
}
