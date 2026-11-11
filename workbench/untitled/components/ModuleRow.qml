import QtQuick
import "../theme/Theme.js" as Theme

/* 源码模块列表项 */
Item {
    id: row

    property string fileName: ""
    property string role: ""
    property int lines: 0
    property string state: "ready"      // ready | placeholder | planned
    property bool selected: false
    signal clicked()

    implicitHeight: 33

    Rectangle {
        anchors.fill: parent
        radius: 4
        color: row.selected ? "#0e3550" : hover.containsMouse ? "#0a283b" : "transparent"
        border.width: 1
        border.color: row.selected ? "#23617f" : "transparent"
    }

    Rectangle {
        x: 7
        anchors.verticalCenter: parent.verticalCenter
        width: 4
        height: 4
        radius: 2
        color: row.state === "ready" ? "#31d98a"
             : row.state === "placeholder" ? "#efa63f" : "#3c6b83"
    }

    Text {
        id: nameText
        x: 17
        anchors.verticalCenter: parent.verticalCenter
        width: 108
        text: row.fileName
        color: row.selected ? "#eafaff" : "#b9d7e8"
        font.family: Theme.fontMono
        font.pixelSize: 9
        elide: Text.ElideRight
    }

    Text {
        anchors.left: nameText.right
        anchors.leftMargin: 2
        anchors.right: countText.left
        anchors.rightMargin: 4
        anchors.verticalCenter: parent.verticalCenter
        text: row.role
        color: row.selected ? "#8fc6dd" : "#52768d"
        font.family: Theme.fontUi
        font.pixelSize: 7
        elide: Text.ElideRight
    }

    Text {
        id: countText
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.verticalCenter: parent.verticalCenter
        text: row.state === "planned" ? "待建" : row.lines + " 行"
        color: row.state === "planned" ? "#efa63f" : "#3f6479"
        font.family: Theme.fontMono
        font.pixelSize: 7
    }

    MouseArea {
        id: hover
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: row.clicked()
    }
}
