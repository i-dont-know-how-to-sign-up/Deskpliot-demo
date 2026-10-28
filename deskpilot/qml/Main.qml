import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    property string emailSettingsError: ""
    property string visionSettingsError: ""
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
        if ((!value && deskPilot.pendingAttachments.length === 0) || deskPilot.busy) return
        deskPilot.ask(value)
        composer.clear()
    }

    Connections {
        target: deskPilot
        function onErrorRaised(message) {
            errorText.text = message || "操作失败，请查看诊断信息。"
            if (settingsDialog.visible) {
                if (errorText.text.indexOf("视觉模型") >= 0)
                    visionSettingsError = errorText.text
                else
                    emailSettingsError = errorText.text
            }
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
                    SmallButton { text: "新会话"; Layout.fillWidth: true; enabled: !deskPilot.busy; onClicked: deskPilot.newSession() }
                    SmallButton { text: "压缩"; Layout.fillWidth: true; enabled: !deskPilot.busy; onClicked: deskPilot.compactSession() }
                }

                SectionTitle { text: "文档索引"; Layout.topMargin: 10 }
                SmallButton { text: "选择文件"; Layout.fillWidth: true; enabled: !deskPilot.busy; onClicked: deskPilot.chooseIndexFile() }
                SmallButton { text: "选择文件夹"; Layout.fillWidth: true; enabled: !deskPilot.busy; onClicked: deskPilot.chooseIndexFolder() }
                SmallButton { text: "清空索引"; Layout.fillWidth: true; enabled: !deskPilot.busy; onClicked: deskPilot.clearIndex() }
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
                SmallButton { text: "设置"; visible: deskPilot.releaseMode; onClicked: settingsDialog.open() }
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
                                Flow {
                                    width: bubble.width - 26
                                    spacing: 8
                                    visible: messageAttachments && messageAttachments.length > 0
                                    Repeater {
                                        model: messageAttachments || []
                                        delegate: Rectangle {
                                            required property var modelData
                                            width: 112
                                            height: 82
                                            radius: 6
                                            color: theme.surfaceSoft
                                            border.color: theme.border
                                            clip: true
                                            Image {
                                                anchors.fill: parent
                                                anchors.margins: 3
                                                source: modelData.thumbnail
                                                fillMode: Image.PreserveAspectCrop
                                                asynchronous: true
                                            }
                                            ToolTip.visible: attachmentHover.hovered
                                            ToolTip.text: modelData.name
                                            HoverHandler { id: attachmentHover }
                                        }
                                    }
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
                Layout.preferredHeight: deskPilot.pendingAttachments.length > 0 ? 230 : 150
                color: theme.surface
                DropArea {
                    anchors.fill: parent
                    onDropped: function(drop) {
                        if (drop.hasUrls) {
                            deskPilot.addImageUrls(JSON.stringify(drop.urls))
                            drop.acceptProposedAction()
                        }
                    }
                }
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8
                    ListView {
                        id: attachmentTray
                        Layout.fillWidth: true
                        Layout.preferredHeight: deskPilot.pendingAttachments.length > 0 ? 72 : 0
                        visible: deskPilot.pendingAttachments.length > 0
                        orientation: ListView.Horizontal
                        spacing: 8
                        clip: true
                        model: deskPilot.pendingAttachments
                        delegate: Rectangle {
                            required property var modelData
                            width: 132
                            height: 68
                            radius: 6
                            color: theme.surfaceRaised
                            border.color: theme.border
                            Image {
                                width: 62
                                height: 62
                                anchors.left: parent.left
                                anchors.leftMargin: 3
                                anchors.verticalCenter: parent.verticalCenter
                                source: modelData.thumbnail
                                fillMode: Image.PreserveAspectCrop
                                asynchronous: true
                            }
                            Label {
                                anchors.left: parent.left
                                anchors.leftMargin: 70
                                anchors.right: removeAttachment.left
                                anchors.rightMargin: 4
                                anchors.verticalCenter: parent.verticalCenter
                                text: modelData.name
                                elide: Text.ElideMiddle
                                color: theme.text
                                font.pixelSize: 11
                            }
                            Button {
                                id: removeAttachment
                                width: 28
                                height: 28
                                anchors.right: parent.right
                                anchors.rightMargin: 4
                                anchors.top: parent.top
                                anchors.topMargin: 4
                                text: "×"
                                Accessible.name: "移除图片 " + modelData.name
                                onClicked: deskPilot.removePendingAttachment(modelData.assetId)
                                ToolTip.visible: hovered
                                ToolTip.text: "移除图片"
                                background: Rectangle {
                                    radius: 4
                                    color: removeAttachment.hovered ? "#334155" : "#1c2532"
                                }
                                contentItem: Label {
                                    text: removeAttachment.text
                                    color: theme.muted
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    font.pixelSize: 16
                                }
                            }
                        }
                    }
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
                        SmallButton {
                            text: "添加图片"
                            enabled: !deskPilot.busy
                            Accessible.name: "添加图片附件"
                            onClicked: deskPilot.chooseImageAttachments()
                        }
                        SmallButton {
                            text: "粘贴图片"
                            enabled: !deskPilot.busy
                            Accessible.name: "粘贴剪贴板图片"
                            onClicked: deskPilot.pasteImageAttachment()
                        }
                        SmallButton { text: "网页调研"; enabled: !deskPilot.busy; onClicked: { deskPilot.research(composer.text, 5); composer.clear() } }
                        Button {
                            id: sendButton
                            text: "发送"
                            enabled: !deskPilot.busy
                            implicitHeight: 34
                            padding: 16
                            onClicked: ask()
                            contentItem: Label { text: sendButton.text; color: "#0b1220"; font.weight: Font.DemiBold; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                            background: Rectangle { radius: 6; color: sendButton.down ? "#0e7490" : (sendButton.hovered ? "#67e8f9" : theme.cyan); border.color: sendButton.activeFocus ? "white" : theme.cyan }
                        }
                    }
                }
            }
            Shortcut {
                sequence: "Ctrl+Shift+V"
                enabled: !deskPilot.busy
                onActivated: deskPilot.pasteImageAttachment()
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
                    objectName: "observerTabs"
                    Layout.fillWidth: true
                    TabButton { text: "Steps" }
                    TabButton { text: "Evidence" }
                    TabButton { text: "Memory" }
                    TabButton { text: "Context" }
                    TabButton { text: "Tasks" }
                }
                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: tabs.currentIndex
                    ListView {
                        id: stepsList
                        objectName: "stepsList"
                        clip: true
                        spacing: 9
                        model: deskPilot.steps
                        boundsBehavior: Flickable.StopAtBounds
                        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                        Label {
                            anchors.centerIn: parent
                            visible: stepsList.count === 0
                            text: deskPilot.status === "正在处理" ? "正在生成执行步骤..." : "本轮暂无执行步骤"
                            color: theme.muted
                            font.pixelSize: 11
                        }
                        delegate: Rectangle {
                            required property var modelData
                            width: stepsList.width
                            height: stepText.implicitHeight + 20
                            radius: 6
                            color: theme.surfaceSoft
                            border.color: modelData.status === "failed" ? theme.danger : theme.border
                            TextEdit {
                                id: stepText
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.margins: 10
                                text: "[" + modelData.status + "] " + modelData.name + "\n" + modelData.detail
                                color: theme.text
                                font.pixelSize: 11
                                wrapMode: TextEdit.Wrap
                                textFormat: TextEdit.PlainText
                                readOnly: true
                                selectByMouse: true
                                selectByKeyboard: true
                                cursorVisible: false
                                selectionColor: theme.cyanSoft
                                selectedTextColor: theme.text
                            }
                        }
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
                    ListView {
                        id: tasksList
                        objectName: "tasksList"
                        clip: true
                        spacing: 8
                        model: deskPilot.tasks
                        boundsBehavior: Flickable.StopAtBounds
                        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                        Label {
                            anchors.centerIn: parent
                            visible: tasksList.count === 0
                            text: "暂无任务"
                            color: theme.muted
                            font.pixelSize: 12
                        }
                        delegate: Rectangle {
                            required property var modelData
                            width: tasksList.width
                            height: taskContents.implicitHeight + 20
                            radius: 6
                            color: theme.surfaceSoft
                            border.color: modelData.state === "failed" || modelData.state === "recovery_required" ? theme.danger : theme.border
                            ColumnLayout {
                                id: taskContents
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.margins: 10
                                spacing: 6
                                Label {
                                    Layout.fillWidth: true
                                    text: modelData.title
                                    color: theme.text
                                    font.pixelSize: 12
                                    font.weight: Font.DemiBold
                                    wrapMode: Text.WrapAnywhere
                                }
                                Label { text: modelData.status; color: theme.muted; font.pixelSize: 11 }
                                TextEdit {
                                    Layout.fillWidth: true
                                    text: modelData.id
                                    color: theme.muted
                                    font.pixelSize: 11
                                    textFormat: TextEdit.PlainText
                                    wrapMode: TextEdit.WrapAnywhere
                                    readOnly: true
                                    selectByMouse: true
                                    cursorVisible: false
                                    selectionColor: theme.cyanSoft
                                }
                                ProgressBar {
                                    id: taskProgress
                                    Layout.fillWidth: true
                                    value: modelData.progress
                                    Layout.preferredHeight: 6
                                    background: Rectangle { implicitHeight: 6; color: theme.surfaceRaised; radius: 3 }
                                    contentItem: Item {
                                        implicitHeight: 6
                                        Rectangle { width: taskProgress.visualPosition * parent.width; height: parent.height; radius: 3; color: theme.cyan }
                                    }
                                }
                                TextEdit {
                                    Layout.fillWidth: true
                                    text: modelData.detail
                                    readOnly: true
                                    selectByMouse: true
                                    selectByKeyboard: true
                                    wrapMode: TextEdit.Wrap
                                    textFormat: TextEdit.PlainText
                                    color: theme.muted
                                    font.pixelSize: 11
                                    cursorVisible: false
                                    selectionColor: theme.cyanSoft
                                    selectedTextColor: theme.text
                                }
                                Flow {
                                    Layout.fillWidth: true
                                    spacing: 6
                                    SmallButton { text: "暂停"; visible: modelData.canPause; onClicked: deskPilot.controlTask(modelData.id, "pause") }
                                    SmallButton { text: "取消"; visible: modelData.canCancel; onClicked: deskPilot.controlTask(modelData.id, "cancel") }
                                    SmallButton { text: "继续"; visible: modelData.canResume; enabled: !deskPilot.busy; onClicked: deskPilot.controlTask(modelData.id, "resume") }
                                    SmallButton { text: "补偿文件"; visible: modelData.compensationOperationId !== ""; enabled: !deskPilot.busy; onClicked: deskPilot.requestCompensation(modelData.id, modelData.compensationOperationId) }
                                    SmallButton { text: "核对邮件"; visible: modelData.mailOperationId !== ""; enabled: !deskPilot.busy; onClicked: deskPilot.reconcileMail(modelData.id, modelData.mailOperationId) }
                                    SmallButton { text: modelData.canApprove ? "重新预览" : "查看会话"; enabled: !deskPilot.busy; onClicked: deskPilot.previewTask(modelData.id) }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Dialog {
        id: settingsDialog
        modal: true
        onOpened: { window.emailSettingsError = ""; window.visionSettingsError = "" }
        title: "DeskPilot 设置"
        anchors.centerIn: parent
        width: 560
        height: 540
        standardButtons: Dialog.Close
        contentItem: ScrollView {
            clip: true
            ColumnLayout {
                width: settingsDialog.availableWidth
                spacing: 12
                Label { text: "数据目录"; color: theme.muted; font.pixelSize: 12 }
                TextField { text: deskPilot.releaseInfo.home || ""; readOnly: true; selectByMouse: true; Layout.fillWidth: true }
                Label { text: "模型"; color: theme.muted; font.pixelSize: 12 }
                TextField { id: releaseModel; text: deskPilot.releaseInfo.model || ""; Layout.fillWidth: true; Accessible.name: "模型名称" }
                Label { text: "更换 API Key（留空则保持原值）"; color: theme.muted; font.pixelSize: 12 }
                TextField { id: releaseKey; echoMode: TextInput.Password; Layout.fillWidth: true; Accessible.name: "模型 API Key" }
                CheckBox { id: releaseCloud; text: "允许选中的图片上传到云端模型"; checked: deskPilot.releaseInfo.cloudImages || false }
                RowLayout {
                    SmallButton { text: "保存设置"; onClicked: { deskPilot.saveReleaseSettings(releaseModel.text, releaseKey.text, releaseCloud.checked); releaseKey.clear() } }
                    Label { text: "设置变更重启后生效"; color: theme.muted; font.pixelSize: 11 }
                }
                Label { text: "视觉模型"; color: theme.muted; font.pixelSize: 12 }
                TextField {
                    id: releaseVisionModel
                    text: deskPilot.releaseInfo.visionModel || "qwen-vl-max-latest"
                    Layout.fillWidth: true
                    Accessible.name: "视觉模型名称"
                }
                TextField {
                    id: releaseVisionBaseUrl
                    text: deskPilot.releaseInfo.visionBaseUrl || "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    Layout.fillWidth: true
                    Accessible.name: "视觉模型 API 基础地址"
                }
                TextField {
                    id: releaseVisionKey
                    echoMode: TextInput.Password
                    placeholderText: "视觉模型 API Key（留空保持原值）"
                    Layout.fillWidth: true
                    Accessible.name: "视觉模型 API Key"
                }
                SmallButton {
                    text: "保存视觉模型设置"
                    onClicked: {
                        window.visionSettingsError = ""
                        if (deskPilot.saveVisionSettings(releaseVisionModel.text, releaseVisionBaseUrl.text, releaseVisionKey.text))
                            releaseVisionKey.clear()
                    }
                }
                Label {
                    visible: window.visionSettingsError.length > 0
                    text: window.visionSettingsError
                    color: theme.danger
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    Accessible.name: "视觉模型设置错误"
                }
                Label { text: "邮箱"; color: theme.muted; font.pixelSize: 12 }
                Label {
                    visible: deskPilot.releaseInfo.vaultError || false
                    text: "当前 Windows 用户无法解密已保存的凭据。请确认重置后重新输入模型 Key 和邮箱授权码。"
                    color: theme.danger
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                }
                ComboBox {
                    id: releaseEmailProvider
                    Layout.fillWidth: true
                    model: ["mock", "qq", "163", "outlook"]
                    Component.onCompleted: currentIndex = Math.max(0, model.indexOf(deskPilot.releaseInfo.emailProvider || "mock"))
                    Accessible.name: "邮箱服务商"
                }
                TextField {
                    id: releaseEmailUsername
                    text: deskPilot.releaseInfo.emailUsername || ""
                    placeholderText: "邮箱地址"
                    Layout.fillWidth: true
                    Accessible.name: "邮箱地址"
                }
                TextField {
                    id: releaseEmailPassword
                    echoMode: TextInput.Password
                    placeholderText: "客户端授权码（留空保持原值）"
                    Layout.fillWidth: true
                    Accessible.name: "邮箱客户端授权码"
                }
                SmallButton {
                    text: "保存邮箱设置"
                    onClicked: {
                        window.emailSettingsError = ""
                        if (deskPilot.saveEmailSettings(releaseEmailProvider.currentText, releaseEmailUsername.text, releaseEmailPassword.text))
                            releaseEmailPassword.clear()
                    }
                }
                Label {
                    visible: window.emailSettingsError.length > 0
                    text: window.emailSettingsError
                    color: theme.danger
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    Accessible.name: "邮箱设置错误"
                }
                SmallButton {
                    text: "重置本机凭据"
                    ToolTip.visible: resetHover.hovered
                    ToolTip.text: "仅用于当前 Windows 用户无法解密旧凭据时"
                    HoverHandler { id: resetHover }
                    onClicked: resetCredentialsDialog.open()
                }
                CheckBox {
                    text: "登录后启动后台任务 worker"
                    checked: deskPilot.releaseInfo.autostart || false
                    onClicked: deskPilot.setWorkerAutostart(checked)
                }
                SmallButton { text: "导出诊断信息"; onClicked: deskPilot.exportDiagnostics() }
                Label { text: "反馈（仅保存在本机，不自动上传）"; color: theme.muted; font.pixelSize: 12 }
                TextArea {
                    id: feedbackText
                    Layout.fillWidth: true
                    Layout.preferredHeight: 80
                    wrapMode: TextEdit.Wrap
                    placeholderText: "描述遇到的问题；请勿填写密码或私人正文"
                }
                SmallButton { text: "保存反馈"; enabled: feedbackText.text.trim().length > 0; onClicked: { deskPilot.saveFeedback(feedbackText.text); feedbackText.clear() } }
            }
        }
    }

    Dialog {
        id: resetCredentialsDialog
        modal: true
        title: "重置本机凭据"
        standardButtons: Dialog.Ok | Dialog.Cancel
        anchors.centerIn: parent
        width: Math.min(480, window.width - 32)
        background: Rectangle { color: theme.surface; border.color: theme.border; radius: 8 }
        contentItem: Label {
            text: "旧加密文件会保留；重置后需重新输入模型 API Key 和邮箱授权码。确认继续？"
            color: theme.text
            wrapMode: Text.Wrap
            padding: 14
        }
        onAccepted: deskPilot.resetReleaseCredentials()
    }

    Dialog {
        id: errorDialog
        objectName: "errorDialog"
        modal: true
        title: "操作失败"
        standardButtons: Dialog.Ok
        anchors.centerIn: parent
        width: Math.min(520, window.width - 32)
        height: 220
        background: Rectangle { color: theme.surface; border.color: theme.border; radius: 8 }
        contentItem: TextArea {
            id: errorText
            objectName: "errorText"
            readOnly: true
            selectByMouse: true
            wrapMode: TextEdit.Wrap
            color: theme.text
            background: Rectangle { color: "transparent" }
            padding: 14
            implicitHeight: Math.min(220, Math.max(72, contentHeight + 28))
        }
    }
}
