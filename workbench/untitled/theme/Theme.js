.pragma library

// ── 基础色板 ────────────────────────────────────────────────
var bg0          = "#030b15"
var bg1          = "#071a2b"

var panelTop     = "#0a2237"
var panelBottom  = "#071827"
var panelBorder  = "#123a54"

var cardTop      = "#0b2840"
var cardBottom   = "#081e31"
var cardBorder   = "#17455f"

var accent       = "#38c9f5"
var accentDeep   = "#1d84b5"
var accentSoft   = "#8fe3ff"
var teal         = "#1fd1c4"

var text         = "#e7f4ff"
var textDim      = "#9dbfd6"
var muted        = "#6e92aa"
var faint        = "#4c7088"

var green        = "#31d98a"
var amber        = "#efa63f"
var red          = "#ef5f6b"

// ── 字体 ────────────────────────────────────────────────────
var fontUi       = "Microsoft YaHei UI"
var fontMono     = "Consolas"

// ── 常用渐变 ────────────────────────────────────────────────
function panelGradient() {
    return [
        { "position": 0.0, "color": panelTop },
        { "position": 1.0, "color": panelBottom }
    ]
}
