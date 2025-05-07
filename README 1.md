# Flutter macOS 录音项目搭建与问题解决笔记

## 📋 项目简介

本项目是一个基于 Flutter 的 macOS 桌面应用，  
实现了录音功能，后续计划接入 Whisper 进行语音识别。

项目名：`voice_assistant_fresh`

---

## 📦 开发环境

- macOS Sonoma (或以上)
- Flutter 3.x（稳定版）
- Android Studio / VSCode / Xcode 15+
- Git
- CocoaPods

---

## 🔥 步骤详细记录

### 1. 创建 Flutter 项目

```bash
flutter create voice_assistant_fresh
cd voice_assistant_fresh
```

启用 macOS 桌面支持：
flutter config --enable-macos-desktop

确认 macOS 支持已开启：
flutter devices

### 2. 配置 Flutter 插件

#### pubspec.yaml 添加依赖

dependencies:
  flutter:
    sdk: flutter
  flutter_sound:
    git:
      url: https://github.com/Canardoux/flutter_sound.git
      ref: master
  path_provider: ^2.1.2
  flutter_tts: ^3.8.5

注意：  
⚡ 使用 Git 仓库版本，必须用 `ref: master`，因为 flutter_sound 没有 main 分支。  
⚡ 不要写 `path:`，否则 pub 找不到 pubspec.yaml。

然后执行：
flutter clean
flutter pub get
flutter build macos

### 3. 解决 MissingPluginException 错误

问题：
MissingPluginException(No implementation found for method resetPlugin on channel xyz.canardoux.flutter_sound_recorder)


原因：  
flutter_sound 默认 pub.dev 版是 lite 模式，不支持 macOS，需要拉取 Git 完整版。

解决办法：

- 用上面 pubspec.yaml 的 git 依赖方式
    
- 确保 `GeneratedPluginRegistrant.swift` 文件中包含：FlutterSoundPlugin.register(with: registry.registrar(forPlugin: "FlutterSoundPlugin"))
- - 使用 flutter build macos 后再 flutter run

### 4. macOS 沙盒权限配置（麦克风访问）

**在 Xcode 中设置：**

打开 `macos/Runner.xcworkspace`：

open macos/Runner.xcworkspace


- 选中 Runner Target → Signing & Capabilities → 添加 `App Sandbox`
    
- 在 Sandbox 权限里勾选 `Audio Input`
    
- Info.plist 添加麦克风权限说明：


<key>NSMicrophoneUsageDescription</key>
<string>需要使用麦克风录音</string>
<key>NSMicrophoneUsageDescription</key>
<string>需要使用麦克风录音</string>



### 5. 手动签名 App（本地运行需要）

如果 flutter run 后 build 出的 .app 无法申请权限，需要手动签名：

codesign --entitlements macos/Runner/Release.entitlements --force --sign - build/macos/Build/Products/Debug/voice_assistant_fresh.app

签名后重新打开 .app：

open build/macos/Build/Products/Debug/voice_assistant_fresh.app
⚡ 这个命令可以清除之前拒绝麦克风授权的记录，重新请求权限。

### 7. 录音器初始化防护（重要）

在 Flutter 中，Recorder 是异步初始化的，所以必须添加保护逻辑：

dart

bool _isRecorderInitialized = false;

Future<void> _initRecorder() async {
  await _recorder.openRecorder();
  final dir = await getApplicationDocumentsDirectory();
  _audioPath = '${dir.path}/record.wav';
  _isRecorderInitialized = true;
  setState(() {});
}

@override
void initState() {
  super.initState();
  _initRecorder();
}



** 当前未解决的问题 **

### 1. macOS 系统麦克风权限申请失败

虽然已按照标准流程进行了：

- 配置 App Sandbox，勾选了 Audio Input 权限
    
- Info.plist 添加了 `NSMicrophoneUsageDescription`
    
- 手动对 .app 文件进行了签名
    
- 通过 `flutter run -d macos` 或 `Xcode Run` 正确运行应用
    
- 确认了 `GeneratedPluginRegistrant.swift` 中正确注册了 flutter_sound 插件
    

**但是：**

- 应用运行后**没有触发系统麦克风权限弹窗**
    
- 点击麦克风按钮时无法开始录音
    
- 无明显 Flutter 错误提示（没有崩溃，没有异常输出）
    

---

### 2. 已尝试过的修复手段（均未奏效）

- 使用 `tccutil reset Microphone` 重置麦克风权限
    
- 重新清理项目 `flutter clean`
    
- 重建 macos 目录 `flutter create .`
    
- 确认 Xcode 中 Target -> Signing & Capabilities 配置正确
    
- 尝试 Debug、Release 两种模式分别签名并运行
    

**均未成功触发麦克风授权弹窗或录音成功。**

### 3. 初步推测可能原因

- Flutter macOS 平台与系统沙盒权限集成存在细节差异
    
- `flutter_sound` 在 macOS 上录音部分存在兼容性或初始化异常（未明显暴露）
    
- 需要额外的 macOS 原生开发（Swift/Objective-C）层处理权限申请
    
- 需要在 Xcode 添加 Entitlements 文件中麦克风相关权利声明
    
- 或者需要使用更底层的音频接口（如 AVFoundation）替代 Flutter 插件方案





## 当前项目运行方式

- 开发调试：

flutter run -d macos

手动打开:

open build/macos/Build/Products/Debug/voice_assistant_fresh.app

✨ 后续计划


- 接入 Whisper 语音识别（本地或 OpenAI API）
    
- 实现录音识别后文本展示
    
- UI 美化，增加进度条与提示动画
    
- 适配 Windows/Linux 平台


# 🏁 总结

本项目完整记录了 Flutter macOS 桌面开发中遇到的所有问题、解决方案与最佳实践，  
是一个学习 Flutter 桌面开发 + 音频处理 + 权限管理的重要参考。

✅ 环境搭建  
✅ 插件正确引入  
✅ 权限管理  
✅ 防护崩溃  
✅ 终端命令总结  
✅ 真实踩坑全过程记录