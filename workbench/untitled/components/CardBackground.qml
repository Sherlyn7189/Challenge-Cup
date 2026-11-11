import QtQuick
import "../theme/Theme.js" as Theme

/* 深色科技风卡片底板：渐变 + 细描边 + 顶部高光 */
Rectangle {
    id: card

    property color topColor: Theme.panelTop
    property color bottomColor: Theme.panelBottom
    property color strokeColor: Theme.panelBorder
    property real cardRadius: 10

    radius: cardRadius
    color: "transparent"
    border.width: 1
    border.color: strokeColor
    antialiasing: true
    gradient: Gradient {
        GradientStop { position: 0.0; color: card.topColor }
        GradientStop { position: 1.0; color: card.bottomColor }
    }

    // 顶部内高光，模拟玻璃质感
    Rectangle {
        anchors.top: parent.top
        anchors.topMargin: 1
        anchors.left: parent.left
        anchors.leftMargin: 18
        anchors.right: parent.right
        anchors.rightMargin: 18
        height: 1
        color: "#3a86ab"
        opacity: 0.35
    }

    // 左下角淡蓝光斑，增加层次
    Rectangle {
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: -40
        anchors.bottomMargin: -40
        width: 180
        height: 180
        radius: width / 2
        color: "#1e7fa8"
        opacity: 0.05
    }
}
