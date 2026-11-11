import QtQuick
import "../theme/Theme.js" as Theme

/* 单炮记录卡片：解释“时间 × 检波点”的记录形式，并可打开详细说明 */
Rectangle {
    id: card

    property var info: bridge ? bridge.shotInfo : ({})
    signal showDetails()

    readonly property bool ready: info.available === true
    readonly property int nrec: Number(info.nrec) || 0
    readonly property int nt: Number(info.nt) || 0
    readonly property real dt: Number(info.dt) || 0
    readonly property real vmean: Number(info.vmean) || 0
    readonly property real rms: Number(info.rms) || 0

    radius: 8
    color: "#071a2a"
    border.width: 1
    border.color: hover.containsMouse ? "#2b7fa3" : "#123a54"

    Item {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 36

        Text {
            x: 12
            y: 6
            text: "单炮记录 · " + (card.info.component || "vz")
            color: "#e7f4ff"
            font.family: Theme.fontUi
            font.pixelSize: 10
            font.weight: Font.DemiBold
            font.letterSpacing: 0.3
        }
        Text {
            x: 12
            y: 21
            text: "SHOT RECORD"
            color: "#3f6d88"
            font.family: Theme.fontMono
            font.pixelSize: 5
            font.letterSpacing: 1
        }

        StatusChip {
            anchors.right: parent.right
            anchors.rightMargin: 10
            anchors.verticalCenter: parent.verticalCenter
            text: card.ready
                  ? (card.info.placeholder === false ? "物理记录" : "直达波占位")
                  : "待生成"
            chipColor: !card.ready ? "#3c6b83"
                     : card.info.placeholder === false ? "#31d98a" : "#efa63f"
            height: 17
        }
    }

    ShotPreview {
        id: preview
        x: 8
        y: 36
        width: parent.width - 16
        height: 118
        compact: true
        source: bridge ? (bridge.previews.shot || "") : ""
        info: card.info
    }

    Text {
        x: 9
        y: 158
        text: card.ready
              ? card.nrec + " 道 · " + card.nt + " 采样 · dt " + (card.dt * 1000).toFixed(1) + " ms"
              : "运行流水线后生成 · output/shots/shot.npz"
        color: "#5f8199"
        font.family: Theme.fontMono
        font.pixelSize: 7
        elide: Text.ElideRight
        width: parent.width - 76
    }

    Text {
        x: 9
        y: 171
        text: card.ready
              ? "RMS " + card.rms.toFixed(3) + "  ·  v̄ " + (card.vmean > 0 ? Math.round(card.vmean) : "—") + " m/s"
              : "单炮记录 = 时间 × 检波点"
        color: "#6f9bb2"
        font.family: Theme.fontMono
        font.pixelSize: 7
        elide: Text.ElideRight
        width: parent.width - 18
    }

    Text {
        x: 9
        y: 187
        text: "一次激发，多道接收；RTM 的检波点波场输入"
        color: "#42657b"
        font.family: Theme.fontUi
        font.pixelSize: 6
        elide: Text.ElideRight
        width: parent.width - 18
    }

    Text {
        anchors.right: parent.right
        anchors.rightMargin: 10
        y: 156
        text: "查看说明 ›"
        color: "#8fe3ff"
        font.family: Theme.fontUi
        font.pixelSize: 7
    }

    MouseArea {
        id: hover
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: card.showDetails()
    }
}
