# Voice Assistant Studio New3 - Flutter 安卓语音助手项目开发记录

本项目是一个基于 Flutter 的跨平台语音助手原型，目标是在 Android 模拟器中运行，实现语音识别功能。项目名称为 `voice_assistant_studio_new3`，以下为开发过程中的关键步骤与问题排查记录，尤其关注语音识别相关插件集成中遇到的技术难点与 Flutter 编译环境的坑点。

---

## 🏗️ 项目初始化

```bash
flutter create --platforms=android,ios -a java -i swift voice_assistant_studio_new3
```

- 使用 `.kts` 格式的 `build.gradle.kts`（Kotlin DSL），而不是 `.gradle` 格式
    
- Android NDK 默认版本为 `26.3.11579264`
    

---

## 🎤 插件使用：speech_to_text

### 插件声明

```yaml
dependencies:
  speech_to_text: ^6.5.0
```

### 插件源码路径：

本地调试中曾使用 `dependency_overrides` 方式加载 `speech_to_text` 源码：

```yaml
dependency_overrides:
  speech_to_text:
    path: local_packages/speech_to_text
```

---

## ❌ 遇到的主要问题及解决方案

### ❗ 报错 1：NDK 版本不匹配

```
Your project is configured with Android NDK 26.3.11579264, but plugin requires 27.0.12077973
```

#### ✅ 解决方法：

在 `android/app/build.gradle.kts` 添加：

```kotlin
android {
    ndkVersion = "27.0.12077973"
}
```

---

### ❗ 报错 2：`Unresolved reference: Registrar`

```
SpeechToTextPlugin.kt:37:48 Unresolved reference: Registrar
```

#### ⚠️ 原因分析：

`speech_to_text 6.x` 版本仍使用旧的 `PluginRegistry.Registrar` 注册方式，不兼容新 Flutter 嵌入式插件 API

#### ✅ 解决方法：

- 切换 Flutter 插件方式为新版 `onAttachedToEngine`
    
- 或直接升级插件至 `speech_to_text: ^7.0.0`
    

---

### ❗ 报错 3：未请求麦克风权限

Flutter 内部日志：

```
permissions_handler: No permissions found in manifest for: []
I/flutter: ❌ 麦克风权限未授予
```

#### ✅ 解决方法：

手动在 `AndroidManifest.xml` 中添加权限：

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

并确保动态请求权限代码已触发（通过 `speech_to_text.initialize()`）

---

### ❗ 报错 4：language not supported

Flutter 日志：

```
SpeechRecognitionError msg: error_language_not_supported
```

#### ✅ 解决方法：

- 调用插件的 `locales()` 方法查看支持语言列表
    
- 手动设置 locale 为 `en_US`
    

```dart
await _speech.listen(localeId: 'en_US');
```

---

### ❗ 报错 5：OpenGL ES 错误（模拟器）

```
E/libEGL: called unimplemented OpenGL ES API
```

#### ✅ 解决方法：

- 启动时添加：`flutter run --enable-software-rendering`
    
- 或使用真机调试避免模拟器图形兼容问题
    

---

### ❗ 致命限制：模拟器架构兼容性问题

最终发现 `speech_to_text` 插件在 Android 模拟器上运行语音识别时 **仅支持 x86 架构**，而 Mac (尤其是 M1/M2 芯片) 上的 Android 模拟器默认采用 **arm64 架构**，导致插件底层调用的 Android SpeechRecognizer 无法运行。

#### 🚫 最终结果：

该项目在 Android 模拟器中语音识别始终失败，根本原因是设备架构不兼容。

#### ✅ 替代方案：

- 可在 **真实 Android 设备** 上运行调试
    
- 或尝试更换兼容 x86 架构的模拟器（若可行）
    
- 或转向非 Android 内建识别方案，如 Whisper on-device 推理
    

---

## ✅ 成功运行条件汇总

- Flutter SDK 3.29.3
    
- Android NDK 27.0.12077973
    
- speech_to_text 插件 >= 7.0.0
    
- Android 模拟器支持麦克风输入（需授权）
    
- 确保 AndroidManifest + 权限申请代码齐备
    
- 指定语言为系统支持语言（如 en_US）
    

---

## 🧠 总结与经验

|教训|建议|
|---|---|
|Gradle 脚本 `.kts` 编写不熟|尽量用标准 `.gradle` 格式初始化|
|插件使用旧 API|优先使用兼容 Flutter 3 的插件版本|
|权限申请容易遗漏|强调：Manifest + 动态申请 缺一不可|
|模拟器图形兼容差|遇到 EGL 报错首选 `--enable-software-rendering` 或真机|
|架构不兼容导致语音失败|**建议开发语音功能务必使用真机调试**|

---

## 📌 后续可拓展方向

- 加入语音播报功能（结合 `flutter_tts`）
    
- 实现语音控制功能（比如控制灯光/设备模拟）
    
- 增加聊天对话 UI 和本地 LLM 模型