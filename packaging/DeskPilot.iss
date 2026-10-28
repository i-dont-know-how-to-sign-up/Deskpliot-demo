#define AppVersion "0.9.0-beta.3"
[Setup]
AppId={{C919407A-6520-4AB4-8BF3-1C62F7410035}
AppName=DeskPilot
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\DeskPilot
DefaultGroupName=DeskPilot
OutputDir=..\dist-installer
OutputBaseFilename=DeskPilot-{#AppVersion}-windows-x64
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
UninstallDisplayIcon={app}\DeskPilot.exe

[Files]
Source: "..\dist\DeskPilot\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\DeskPilot"; Filename: "{app}\DeskPilot.exe"
Name: "{group}\DeskPilot 备份与迁移"; Filename: "{app}\DeskPilot.exe"; Parameters: "--maintenance"
Name: "{userdesktop}\DeskPilot"; Filename: "{app}\DeskPilot.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："

[Run]
Filename: "{app}\DeskPilot.exe"; Description: "启动 DeskPilot"; Flags: nowait postinstall skipifsilent
