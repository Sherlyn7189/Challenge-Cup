import QtQuick
import "../theme/Theme.js" as Theme

/* 实验操作按钮，支持 primary / secondary / warn / ghost 四种外观 */
Rectangle {
    id: button

    property string text: ""
    property string variant: "primary"
    signal clicked()

    implicitWidth: 120
    implicitHeight: 34
    radius: 6
    antialiasing: true
    border.width: 1
    opacity: enabled ? 1.0 : 0.42

    readonly property bool isWarn: variant === "warn"
    readonly property bool isGhost: variant === "ghost"
    readonly property bool isSecondary: variant === "secondary"

    border.color: isWarn ? "#d99a3a"
                 : isGhost ? "#1a4b66"
                 : isSecondary ? "#1d5d80"
                 : "#2f9fd0"

    gradient: Gradient {
        GradientStop {
            position: 0.0
            color: button.isWarn ? "#9a6317"
                 : button.isGhost ? "#0a2a40"
                 : button.isSecondary ? "#0e3d5a"
                 : "#126b96"
        }
        GradientStop {
            position: 1.0
            color: button.isWarn ? "#5b380b"
                 : button.isGhost ? "#071d2e"
                 : button.isSecondary ? "#092a41"
                 : "#0a3c5c"
        }
    }

    Rectangle {
        anchors.fill: parent
        anchors.margins: 1
        radius: parent.radius - 1
        color: "transparent"
        border.width: 1
        border.color: button.isWarn ? "#ffd58c" : "#9fe6ff"
        opacity: hoverArea.containsMouse ? 0.22 : 0.06
    }

    Text {
        anchors.centerIn: parent
        width: parent.width - 16
        text: button.text
        horizontalAlignment: Text.AlignHCenter
        elide: Text.ElideRight
        color: button.isWarn ? "#ffe7bf"
             : button.isGhost ? "#a9cfe2"
             : "#eafaff"
        font.family: Theme.fontUi
        font.pixelSize: 12
        font.weight: Font.Medium
        font.letterSpacing: 0.5
    }

    MouseArea {
        id: hoverArea
        anchors.fill: parent
        hoverEnabled: true
        enabled: button.enabled
        cursorShape: Qt.PointingHandCursor
        onClicked: button.clicked()

        onPressedChanged: button.scale = pressed ? 0.985 : 1.0
        onExited: button.scale = 1.0
    }

    Behavior on scale { NumberAnimation { duration: 90 } }
}
