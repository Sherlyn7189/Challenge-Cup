import QtQuick
import "../theme/Theme.js" as Theme

/* 统一的卡片标题栏（紧凑版） */
Item {
    id: header

    property string title: ""
    property string tag: ""
    property color tagColor: Theme.accentSoft
    property bool divider: true

    implicitHeight: 38

    Rectangle {
        x: 12
        anchors.verticalCenter: parent.verticalCenter
        width: 3
        height: 14
        radius: 1.5
        color: Theme.accent
    }

    Text {
        anchors.left: parent.left
        anchors.leftMargin: 22
        anchors.verticalCenter: parent.verticalCenter
        text: header.title
        color: "#e7f4ff"
        font.family: Theme.fontUi
        font.pixelSize: 11
        font.weight: Font.DemiBold
        font.letterSpacing: 0.4
    }

    StatusChip {
        visible: header.tag.length > 0
        anchors.right: parent.right
        anchors.rightMargin: 10
        anchors.verticalCenter: parent.verticalCenter
        text: header.tag
        chipColor: header.tagColor
        height: 18
    }

    Rectangle {
        visible: header.divider
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.leftMargin: 12
        anchors.right: parent.right
        anchors.rightMargin: 12
        height: 1
        color: "#123a54"
        opacity: 0.8
    }
}
