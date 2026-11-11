import QtQuick
import "../theme/Theme.js" as Theme

/* 单炮记录预览：时间 × 检波点 的二维记录，带坐标轴与炮点标注 */
Item {
    id: preview

    property string source: ""
    property var info: ({})
    property bool compact: false

    readonly property bool hasImage: source.length > 0 && image.status === Image.Ready
    readonly property real x0: isFinite(Number(info.x0)) ? Number(info.x0) : 0
    readonly property real xMax: Number(info.xmax) > x0 ? Number(info.xmax) : 1000
    readonly property real tMax: Number(info.tmax) > 0 ? Number(info.tmax) : 0.6
    readonly property real sx: isFinite(Number(info.sx)) ? Number(info.sx) : NaN
    readonly property real sourceRatio: isFinite(sx) ? (sx - x0) / Math.max(1, xMax - x0) : -1
    readonly property bool placeholder: info.placeholder !== false

    readonly property int padLeft: compact ? 22 : 32
    readonly property int padBottom: compact ? 14 : 20
    readonly property int padTop: 4
    readonly property int padRight: 4
    readonly property int axisFont: compact ? 6 : 8

    Rectangle {
        anchors.fill: parent
        radius: 5
        color: "#04111c"
        border.width: 1
        border.color: "#0e2c3f"
    }

    Item {
        id: imageArea
        x: preview.padLeft
        y: preview.padTop
        width: preview.width - preview.padLeft - preview.padRight
        height: preview.height - preview.padTop - preview.padBottom
        clip: true

        Image {
            id: image
            anchors.fill: parent
            source: preview.source
            fillMode: Image.Stretch
            smooth: true
            cache: false
            asynchronous: true
            visible: preview.hasImage
        }

        Column {
            anchors.centerIn: parent
            spacing: 3
            visible: !preview.hasImage

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "运行流水线后生成单炮记录"
                color: "#5f8199"
                font.family: Theme.fontUi
                font.pixelSize: preview.compact ? 7 : 9
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "output/shots/shot.npz"
                color: "#3f6479"
                font.family: Theme.fontMono
                font.pixelSize: preview.compact ? 5 : 7
            }
        }

        // 炮点位置
        Rectangle {
            visible: preview.hasImage && preview.sourceRatio >= 0 && preview.sourceRatio <= 1
            x: preview.sourceRatio * imageArea.width
            y: 0
            width: 1
            height: imageArea.height
            color: "#efa63f"
            opacity: 0.28
        }
        Text {
            visible: preview.hasImage && preview.sourceRatio >= 0 && preview.sourceRatio <= 1
            x: Math.max(0, Math.min(imageArea.width - width, preview.sourceRatio * imageArea.width - width / 2))
            y: 1
            text: "▼"
            color: "#f2b75f"
            font.family: Theme.fontMono
            font.pixelSize: preview.compact ? 6 : 8
        }
        Text {
            visible: preview.hasImage && preview.sourceRatio >= 0 && preview.sourceRatio <= 1 && !preview.compact
            x: Math.min(imageArea.width - width, preview.sourceRatio * imageArea.width + 7)
            y: 2
            text: "炮点"
            color: "#f2b75f"
            font.family: Theme.fontUi
            font.pixelSize: 7
        }

        // 图例
        Rectangle {
            visible: preview.hasImage
            x: 5
            y: 5
            width: preview.compact ? 46 : 66
            height: preview.compact ? 13 : 16
            radius: 3
            color: "#03101a"
            opacity: 0.82
            border.width: 1
            border.color: "#174257"

            Rectangle {
                x: 5
                anchors.verticalCenter: parent.verticalCenter
                width: 8
                height: 2
                color: "#efa63f"
            }
            Text {
                x: 16
                anchors.verticalCenter: parent.verticalCenter
                text: preview.placeholder
                      ? (preview.compact ? "直达波" : "直达波（占位）")
                      : "单炮记录"
                color: "#a9c6d9"
                font.family: Theme.fontUi
                font.pixelSize: preview.compact ? 5 : 7
            }
        }
    }

    // 坐标轴
    Rectangle {
        x: imageArea.x - 1
        y: imageArea.y
        width: 1
        height: imageArea.height
        color: "#2d6079"
    }
    Rectangle {
        x: imageArea.x
        y: imageArea.y + imageArea.height
        width: imageArea.width
        height: 1
        color: "#2d6079"
    }

    Repeater {
        model: 5

        Text {
            property real value: preview.x0 + (preview.xMax - preview.x0) * index / 4
            x: Math.max(0, Math.min(preview.width - width,
                       imageArea.x + (index / 4) * imageArea.width - width / 2))
            y: imageArea.y + imageArea.height + 2
            text: preview.xMax >= 100 ? Math.round(value).toString() : value.toFixed(1)
            color: "#6f93aa"
            font.family: Theme.fontMono
            font.pixelSize: preview.axisFont
        }
    }

    Repeater {
        model: 4

        Text {
            property real value: preview.tMax * index / 3
            x: preview.padLeft - 4 - width
            y: Math.max(0, Math.min(preview.height - height,
                       imageArea.y + (index / 3) * imageArea.height - height / 2))
            text: index === 0 ? "0" : value.toFixed(1)
            color: "#6f93aa"
            font.family: Theme.fontMono
            font.pixelSize: preview.axisFont
            horizontalAlignment: Text.AlignRight
        }
    }

    Text {
        x: 2
        y: 2
        visible: !preview.compact
        text: "t / s"
        color: "#4f7189"
        font.family: Theme.fontMono
        font.pixelSize: 7
    }
    Text {
        anchors.right: parent.right
        anchors.rightMargin: 3
        y: 2
        text: "x / m"
        color: "#4f7189"
        font.family: Theme.fontMono
        font.pixelSize: preview.axisFont
    }
}
