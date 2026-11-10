import QtQuick
import "../theme/Theme.js" as Theme

/* 小型状态标签 */
Rectangle {
    id: chip

    property string text: ""
    property color chipColor: Theme.accent
    property bool outline: true

    implicitWidth: label.implicitWidth + 14
    implicitHeight: 18
    radius: 4
    color: Qt.rgba(chipColor.r, chipColor.g, chipColor.b, 0.10)
    border.width: outline ? 1 : 0
    border.color: Qt.rgba(chipColor.r, chipColor.g, chipColor.b, 0.45)

    Text {
        id: label
        anchors.centerIn: parent
        text: chip.text
        color: chip.chipColor
        font.family: Theme.fontUi
        font.pixelSize: 8
        font.letterSpacing: 0.2
    }
}
