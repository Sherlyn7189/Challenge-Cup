import QtQuick
import "../theme/Theme.js" as Theme

/* 指标小卡 */
Rectangle {
    id: tile

    property string label: ""
    property string value: "—"
    property string unit: ""
    property string note: ""
    property string state: "ok"      // ok | warn | muted
    property color accentColor: Theme.accent

    implicitHeight: 76
    radius: 6
    color: "#082032"
    border.width: 1
    border.color: "#123c53"

    Text {
        x: 10
        y: 7
        text: tile.label
        color: "#6f93aa"
        font.family: Theme.fontUi
        font.pixelSize: 8
        font.letterSpacing: 0.3
    }

    Text {
        x: 10
        y: 22
        text: tile.value
        color: tile.state === "warn" ? "#f2b75f"
             : tile.state === "muted" ? "#7f9fb2" : "#e7f4ff"
        font.family: Theme.fontMono
        font.pixelSize: 17
        font.weight: Font.DemiBold
    }

    Text {
        anchors.left: parent.left
        anchors.leftMargin: 10
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 7
        text: tile.note
        color: "#4f7189"
        font.family: Theme.fontUi
        font.pixelSize: 7
        elide: Text.ElideRight
        width: parent.width - 20
    }

    Text {
        visible: tile.unit.length > 0
        anchors.right: parent.right
        anchors.rightMargin: 9
        y: 22
        text: tile.unit
        color: "#5f8199"
        font.family: Theme.fontMono
        font.pixelSize: 8
    }
}
