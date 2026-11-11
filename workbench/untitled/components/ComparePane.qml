import QtQuick
import "../theme/Theme.js" as Theme

/* 对比图窗：标题 + 图像/占位 + 参数明细 */
Rectangle {
    id: pane

    property string title: ""
    property string tag: ""
    property color tagColor: Theme.accentSoft
    property string source: ""
    property string placeholderText: "待接入"
    property string placeholderNote: ""
    property var info: []
    property string footer: ""

    radius: 7
    color: "#071a2a"
    border.width: 1
    border.color: "#123a54"
    clip: true

    readonly property bool hasImage: source.length > 0 && image.status === Image.Ready

    Item {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 28

        Text {
            x: 10
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
            width: parent.width
            height: 1
            color: "#10384f"
        }
    }

    Rectangle {
        id: imageBox
        x: 6
        y: 34
        width: parent.width - 12
        height: 190
        radius: 4
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

        Item {
            anchors.fill: parent
            visible: !pane.hasImage
            opacity: 0.55

            Repeater {
                model: 8
                Rectangle {
                    x: index * imageBox.width / 7
                    width: 1
                    height: parent.height
                    color: "#123a50"
                    opacity: 0.5
                }
            }
            Repeater {
                model: 6
                Rectangle {
                    y: index * imageBox.height / 5
                    width: parent.width
                    height: 1
                    color: "#123a50"
                    opacity: 0.5
                }
            }
        }

        Column {
            anchors.centerIn: parent
            spacing: 4
            visible: !pane.hasImage

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "∿"
                color: "#2f7d9e"
                font.family: Theme.fontMono
                font.pixelSize: 26
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: pane.placeholderText
                color: "#6f93aa"
                font.family: Theme.fontUi
                font.pixelSize: 9
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                width: imageBox.width - 20
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
            anchors.topMargin: 4
            visible: !pane.hasImage
            text: "PLACEHOLDER"
            color: "#2c5a72"
            font.family: Theme.fontMono
            font.pixelSize: 6
            font.letterSpacing: 1
        }
    }

    Column {
        id: infoList
        x: 10
        y: 232
        width: parent.width - 20
        spacing: 0

        Repeater {
            model: pane.info

            Item {
                width: infoList.width
                height: 27

                Text {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    text: modelData.label
                    color: "#5f8199"
                    font.family: Theme.fontUi
                    font.pixelSize: 7
                }
                Text {
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    width: parent.width - 56
                    horizontalAlignment: Text.AlignRight
                    text: modelData.value
                    color: modelData.color || "#b9d7e8"
                    font.family: Theme.fontMono
                    font.pixelSize: 8
                    elide: Text.ElideRight
                }
                Rectangle {
                    anchors.bottom: parent.bottom
                    width: parent.width
                    height: 1
                    color: "#0d2d41"
                    opacity: 0.7
                }
            }
        }
    }

    Text {
        anchors.left: parent.left
        anchors.leftMargin: 10
        anchors.right: parent.right
        anchors.rightMargin: 10
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 7
        text: pane.footer
        color: "#3f6479"
        font.family: Theme.fontMono
        font.pixelSize: 7
        elide: Text.ElideRight
        visible: pane.footer.length > 0
    }
}
