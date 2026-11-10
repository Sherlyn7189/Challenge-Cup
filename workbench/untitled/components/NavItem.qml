import QtQuick
import "../theme/Theme.js" as Theme

/* 左侧工作流导航项 */
Item {
    id: item

    property int step: 1
    property string title: ""
    property string subtitle: ""
    property string state: "idle"      // ready | active | idle | warn
    property bool selected: false
    signal clicked()

    implicitHeight: 52

    Rectangle {
        anchors.fill: parent
        radius: 7
        border.width: 1
        border.color: item.selected ? "#2f8db3"
                     : hover.containsMouse ? "#1c4d68" : "#10344a"
        gradient: Gradient {
            GradientStop {
                position: 0.0
                color: item.selected ? "#0e3852" : hover.containsMouse ? "#0b2a3e" : "#081d2d"
            }
            GradientStop {
                position: 1.0
                color: item.selected ? "#0a2a40" : hover.containsMouse ? "#091f31" : "#071827"
            }
        }
    }

    Rectangle {
        visible: item.selected
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter
        width: 3
        height: 26
        radius: 1.5
        color: Theme.accent
    }

    Rectangle {
        id: badge
        x: 10
        anchors.verticalCenter: parent.verticalCenter
        width: 24
        height: 24
        radius: 12
        color: item.selected ? "#0e3d58" : "#0a2233"
        border.width: 1
        border.color: item.selected ? "#38c9f5" : "#23495f"

        Text {
            anchors.centerIn: parent
            text: item.step
            color: item.selected ? "#9fe9ff" : "#6f93aa"
            font.family: Theme.fontMono
            font.pixelSize: 9
            font.weight: Font.DemiBold
        }
    }

    Column {
        anchors.left: badge.right
        anchors.leftMargin: 9
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.verticalCenter: parent.verticalCenter
        spacing: 2

        Text {
            width: parent.width
            text: item.title
            elide: Text.ElideRight
            color: item.selected ? "#f0faff" : "#aac8da"
            font.family: Theme.fontUi
            font.pixelSize: 11
            font.weight: item.selected ? Font.DemiBold : Font.Medium
            font.letterSpacing: 0.3
        }
        Text {
            width: parent.width
            text: item.subtitle
            elide: Text.ElideRight
            color: item.selected ? "#7fbdd8" : "#4f7189"
            font.family: Theme.fontUi
            font.pixelSize: 8
            font.letterSpacing: 0.1
        }
    }

    Rectangle {
        anchors.right: parent.right
        anchors.rightMargin: 9
        anchors.top: parent.top
        anchors.topMargin: 9
        width: 5
        height: 5
        radius: 2.5
        color: item.state === "ready" ? "#31d98a"
             : item.state === "warn" ? "#efa63f" : "#3c6b83"
        opacity: 0.9
    }

    MouseArea {
        id: hover
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: item.clicked()
    }
}
