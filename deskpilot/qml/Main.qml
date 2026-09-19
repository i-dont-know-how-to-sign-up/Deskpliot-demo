import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    visible: true
    width: 1440
    height: 900
    minimumWidth: 1120
    minimumHeight: 720
    title: "DeskPilot"
    color: theme.bg
    font.family: "Microsoft YaHei UI"
    font.pixelSize: 13

    QtObject {
        id: theme
        readonly property color bg: "#0d1016"
        readonly property color surface: "#151a23"
        readonly property color surfaceRaised: "#1c2330"
        readonly property color surfaceSoft: "#11161f"
        readonly property color border: "#2a3444"
        readonly property color text: "#edf2f7"
        readonly property color muted: "#94a3b8"
        readonly property color purple: "#8b5cf6"
        readonly property color purpleSoft: "#2e2250"
        readonly property color cyan: "#22d3ee"
        readonly property color cyanSoft: "#123746"
        readonly property color userBubble: "#2563eb"
        readonly property color aiBubble: "#1c2532"
        readonly property color danger: "#f87171"
    }

    function ask() {
        const value = composer.text.trim()
        if (!value) return
        deskPilot.ask(value)
        composer.clear()
    }

    Connections {
        target: deskPilot
        function onErrorRaised(message) {
            errorText.text = message
            errorDialog.open()
        }
    }

    component SectionTitle: Label {
        color: theme.muted
        font.pixelSize: 11
        font.weight: Font.DemiBold
        font.letterSpacing: 0.8
    }

    component SmallButton: Button {
        id: control
        implicitHeight: 34
        padding: 10
        font.pixelSize: 12
        contentItem: Label {
            text: control.text
            color: control.enabled ? theme.text : theme.muted
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            radius: 6
            color: control.down ? theme.surfaceRaised : (control.hovered ? "#263143" : theme.surface)
            border.color: control.activeFocus ? theme.cyan : theme.border
            border.width: 1
        }
    }

    component PanelFrame: Rectangle {
        color: theme.surface
        border.color: theme.border
        border.width: 1
        radius: 8
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 14

        PanelFrame {
            Layout.preferredWidth: 250
            Layout.minimumWidth: 220
            Layout.fillHeight: true
            color: theme.surfaceSoft

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                RowLayout {
                    Layout.fillWidth: true
                    Label { text: "DeskPilot"; color: theme.text; font.pixelSize: 19; font.weight: Font.DemiBold }
                    Item { Layout.fillWidth: true }
                    Rectangle { implicitWidth: 8; implicitHeight: 8; radius: 4; color: theme.cyan; Layout.alignment: Qt.AlignVCenter }
                }
                Label { text: "Personal office agent"; color: theme.muted; font.pixelSize: 11 }

                SectionTitle { text: "会话"; Layout.topMargin: 10 }
                ListView {
                    id: sessionList
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 4
                    model: deskPilot.sessions
                    delegate: ItemDelegate {
                        id: sessionDelegate
                        width: sessionList.width
                        height: 48
                        highlighted: modelData.active
                        onClicked: deskPilot.selectSession(modelData.id)
                        contentItem: RowLayout {
                            spacing: 8
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 3
                                Label { text: (modelData.pinned ? "置顶 · " : "") + modelData.title; color: highlighted ? theme.text : theme.muted; elide: Text.ElideRight; Layout.fillWidth: true; font.pixelSize: 12 }
                                Label { text: modelData.count + " 条消息"; color: theme.muted; font.pixelSize: 10 }
                            }
                            ToolButton {
                                text: "⋮"
                                implicitWidth: 30
                                implicitHeight: 30
                                Layout.alignment: Qt.AlignVCenter
                                onClicked: sessionMenu.open()
                                background: Rectangle {
                                    radius: 6
                                    color: parent.hovered ? theme.surfaceRaised : "transparent"
                                    border.color: parent.hovered ? theme.border : "transparent"
                                }
                                contentItem: Label {
                                    text: parent.text
                                    color: theme.muted
                                    font.pixelSize: 18
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                ToolTip.visible: hovered
                                ToolTip.text: "会话操作"
                            }
                        }
                        background: Rectangle { radius: 6; color: highlighted ? theme.purpleSoft : (sessionDelegate.hovered ? theme.surfaceRaised : "transparent"); border.color: highlighted ? theme.purple : "transparent"; border.width: 1 }

                        Popup {
                            id: sessionMenu
                            x: sessionDelegate.width - width - 4
                            y: sessionDelegate.height
                            padding: 6
                            background: Rectangle { color: theme.surfaceRaised; border.color: theme.border; radius: 6 }
                            Column {
                                spacing: 2
                                MenuItem { text: "重命名"; onTriggered: renameDialog.open() }
                                MenuItem { text: modelData.pinned ? "取消置顶" : "置顶"; onTriggered: deskPilot.togglePinSession(modelData.id) }
                                MenuItem { text: "导出会话"; onTriggered: deskPilot.exportSession(modelData.id) }
                                MenuItem { text: "删除"; enabled: !modelData.active; onTriggered: deskPilot.deleteSession(modelData.id) }
                            }
                        }

                        Dialog {
                            id: renameDialog
                            title: "重命名会话"
                            modal: true
                            standardButtons: Dialog.Ok | Dialog.Cancel
                            contentItem: TextField { id: renameField; text: modelData.title; focus: true; placeholderText: "会话名称"; Component.onCompleted: selectAll() }
                            onAccepted: deskPilot.renameSession(modelData.id, renameField.text)
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    SmallButton { text: "新会话"; Layout.fillWidth: true; onClicked: deskPilot.newSession() }
                    SmallButton { text: "压缩"; Layout.fillWidth: true; onClicked: deskPilot.compactSession() }
                }

                SectionTitle { text: "文档索引"; Layout.topMargin: 10 }
                SmallButton { text: "选择文件"; Layout.fillWidth: true; onClicked: deskPilot.chooseIndexFile() }
                SmallButton { text: "选择文件夹"; Layout.fillWidth: true; onClicked: deskPilot.chooseIndexFolder() }
                SmallButton { text: "清空索引"; Layout.fillWidth: true; onClicked: deskPilot.clearIndex() }
                Label {
                    Layout.fillWidth: true
                    text: deskPilot.stats.documents + " 个文档  ·  " + deskPilot.stats.chunks + " 个片段"
                    color: theme.muted
                    font.pixelSize: 11
                    wrapMode: Text.Wrap
                }
                Label { text: "状态：" + deskPilot.status; color: theme.cyan; font.pixelSize: 11; Layout.fillWidth: true; elide: Text.ElideRight }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            RowLayout {
                Layout.fillWidth: true
                Label { text: "对话工作区"; color: theme.text; font.pixelSize: 20; font.weight: Font.DemiBold }
                Item { Layout.fillWidth: true }
                Rectangle { implicitWidth: 9; implicitHeight: 9; radius: 4.5; color: deskPilot.status.indexOf("正在") === 0 ? theme.cyan : theme.purple }
                Label { text: deskPilot.status; color: theme.muted; font.pixelSize: 12 }
            }

            PanelFrame {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: theme.bg

                ListView {
                    id: chatList
                    property bool followOutput: true
                    anchors.fill: parent
                    anchors.margins: 18
                    clip: true
                    spacing: 16
                    model: deskPilot.messages
                    boundsBehavior: Flickable.StopAtBounds
                    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                    delegate: Item {
                        width: chatList.width
                        height: bubble.height + 4
                        readonly property bool isUser: messageRole === "user"
                        readonly property bool isTool: messageRole === "tool" || messageRole === "system"
                        readonly property bool isConfirmation: messageRole === "confirmation"
                        Rectangle {
                            id: bubble
                            width: Math.max(240, Math.min(chatList.width * 0.78, 714))
                            height: contentColumn.implicitHeight + 26
                            anchors.left: isUser ? undefined : parent.left
                            anchors.right: isUser ? parent.right : undefined
                            radius: 8
                            color: isUser ? theme.userBubble : (isConfirmation ? "#3b2d16" : (isTool ? theme.surfaceSoft : theme.aiBubble))
                            border.color: isUser ? "#3b82f6" : (isConfirmation ? "#d59b35" : theme.border)
                            border.width: 1
                            Column {
                                id: contentColumn
                                anchors.fill: parent
                                anchors.margins: 13
                                spacing: 6
                                Label {
                                    id: roleLabel
                                    text: isUser ? "你" : (isTool ? "工具" : "DeskPilot")
                                    color: isUser ? "#dbeafe" : theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.DemiBold
                                }
                                TextEdit {
                                    id: bubbleText
                                    width: bubble.width - 26
                                    height: contentHeight
                                    text: messageContent
                                    color: isUser ? "white" : theme.text
                                    font.pixelSize: 14
                                    wrapMode: TextEdit.Wrap
                                    textFormat: TextEdit.PlainText
                                    readOnly: true
                                    selectByMouse: true
                                    selectByKeyboard: true
                                    cursorVisible: false
                                    selectionColor: isUser ? "#60a5fa" : theme.cyanSoft
                                    selectedTextColor: isUser ? "white" : theme.text
                                }
                                Row {
                                    id: confirmActions
                                    visible: isConfirmation
                                    spacing: 8
                                    Button {
                                        text: "确认执行"
                                        implicitHeight: 30
                                        onClicked: deskPilot.approveAction("confirmed")
                                    }
                                    Button {
                                        text: "取消"
                                        implicitHeight: 30
                                        onClicked: deskPilot.cancelAction()
                                    }
                                }
                            }
                        }
                    }
                    onMovementStarted: followOutput = false
                    onMovementEnded: followOutput = atYEnd
                    onContentHeightChanged: {
                        if (followOutput)
                            Qt.callLater(function() { chatList.positionViewAtEnd() })
                    }
                    onCountChanged: {
                        followOutput = true
                        Qt.callLater(function() { chatList.positionViewAtEnd() })
                    }
                }
            }

            PanelFrame {
                Layout.fillWidth: true
                Layout.preferredHeight: 142
                color: theme.surface
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        TextArea {
                            id: composer
                            placeholderText: "输入问题、文档任务或网页调研主题..."
                            wrapMode: TextArea.Wrap
                            color: theme.text
                            placeholderTextColor: theme.muted
                            font.pixelSize: 14
                            background: Rectangle { color: theme.surfaceSoft; radius: 6; border.color: composer.activeFocus ? theme.cyan : theme.border; border.width: 1 }
                            Keys.onReturnPressed: function(event) { if (event.modifiers & Qt.ControlModifier) { ask(); event.accepted = true } }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "Ctrl + Enter 发送"; color: theme.muted; font.pixelSize: 10 }
                        Item { Layout.fillWidth: true }
                        SmallButton { text: "网页调研"; onClicked: { deskPilot.research(composer.text, 5); composer.clear() } }
                        Button {
                            id: sendButton
                            text: "发送"
                            implicitHeight: 34
                            padding: 16
                            onClicked: ask()
                            contentItem: Label { text: sendButton.text; color: "#0b1220"; font.weight: Font.DemiBold; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                            background: Rectangle { radius: 6; color: sendButton.down ? "#0e7490" : (sendButton.hovered ? "#67e8f9" : theme.cyan); border.color: sendButton.activeFocus ? "white" : theme.cyan }
                        }
                    }
                }
            }
        }

        PanelFrame {
            Layout.preferredWidth: 330
            Layout.minimumWidth: 280
            Layout.fillHeight: true
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8
                Label { text: "Agent 观察"; color: theme.text; font.pixelSize: 16; font.weight: Font.DemiBold }
                TabBar {
                    id: tabs
                    Layout.fillWidth: true
                    TabButton { text: "Steps" }
                    TabButton { text: "Evidence" }
                    TabButton { text: "Memory" }
                    TabButton { text: "Context" }
                }
                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: tabs.currentIndex
                    ScrollView {
                        clip: true
                        Column { width: parent.width; spacing: 9; Repeater { model: deskPilot.steps; delegate: Rectangle { width: parent.width; height: stepText.contentHeight + 22; radius: 6; color: theme.surfaceSoft; border.color: modelData.status === "failed" ? theme.danger : theme.border; TextEdit { id: stepText; anchors.fill: parent; anchors.margins: 10; text: "[" + modelData.status + "] " + modelData.name + "\n" + modelData.detail; color: theme.text; font.pixelSize: 11; wrapMode: TextEdit.Wrap; textFormat: TextEdit.PlainText; readOnly: true; selectByMouse: true; selectByKeyboard: true; cursorVisible: false; selectionColor: theme.cyanSoft; selectedTextColor: theme.text } } } }
                    }
                    ScrollView {
                        clip: true
                        Column { width: parent.width; spacing: 9; Repeater { model: deskPilot.evidences; delegate: Rectangle { width: parent.width; height: evidenceText.contentHeight + 22; radius: 6; color: theme.surfaceSoft; border.color: theme.border; TextEdit { id: evidenceText; anchors.fill: parent; anchors.margins: 10; text: modelData.source + "  ·  " + modelData.score + "\n" + modelData.text; color: theme.text; font.pixelSize: 11; wrapMode: TextEdit.Wrap; textFormat: TextEdit.PlainText; readOnly: true; selectByMouse: true; selectByKeyboard: true; cursorVisible: false; selectionColor: theme.cyanSoft; selectedTextColor: theme.text } } } }
                    }
                    ScrollView {
                        clip: true
                        Column { width: parent.width; spacing: 9; Repeater { model: deskPilot.memories; delegate: Rectangle { width: parent.width; height: memoryText.contentHeight + 22; radius: 6; color: theme.surfaceSoft; border.color: theme.border; TextEdit { id: memoryText; anchors.fill: parent; anchors.margins: 10; text: modelData.title + "  ·  " + modelData.confidence + "\n" + modelData.content; color: theme.text; font.pixelSize: 11; wrapMode: TextEdit.Wrap; textFormat: TextEdit.PlainText; readOnly: true; selectByMouse: true; selectByKeyboard: true; cursorVisible: false; selectionColor: theme.cyanSoft; selectedTextColor: theme.text } } } }
                    }
                    ScrollView { clip: true; TextEdit { width: parent.width; text: deskPilot.context || "本轮上下文将在这里显示。"; color: theme.text; font.pixelSize: 11; wrapMode: TextEdit.Wrap; textFormat: TextEdit.PlainText; readOnly: true; selectByMouse: true; selectByKeyboard: true; cursorVisible: false; selectionColor: theme.cyanSoft; selectedTextColor: theme.text } }
                }
            }
        }
    }

    Dialog {
        id: errorDialog
        modal: true
        title: "操作失败"
        standardButtons: Dialog.Ok
        anchors.centerIn: parent
        width: 480
        contentItem: Label { id: errorText; color: theme.text; wrapMode: Text.Wrap; padding: 14 }
    }
}
