# 🦙 Flutter + Ollama 本地大模型问答 Demo

本项目演示如何使用 Flutter 通过 HTTP 接口调用本地部署的 Ollama（以 LLaMA3 为例）模型，实现轻量问答 UI 应用。

---

## ✨ 项目特点

- 使用 Flutter 构建跨平台桌面应用（macOS tested）
- 与 Ollama 启动的 LLaMA3 模型通信
- 支持用户自定义输入、模型响应显示
- UI 现代化美化（输入框、按钮、滚动回答框）
- 本地离线大语言模型，无需联网调用 API

---

## 🧱 技术栈

- Flutter（3.x+）
- Dart
- Ollama + LLaMA3 模型（在 macOS 本地运行）
- HTTP 协议交互

---

## 🚀 快速开始

### 1️⃣ 启动本地 Ollama 模型

```bash
# 安装 ollama（若尚未安装）
brew install ollama

# 拉取并运行 llama3 模型（首次会下载约 4GB）
ollama run llama3

```

等终端出现：
>>> llama3: ready to respond

说明模型服务已准备就绪并监听在 `http://localhost:11434`。
### 2️⃣ 运行 Flutter 项目（macOS 桌面）
flutter pub get
flutter run -d macos

#### 🖼 UI 效果截图
- 输入框 + 发送按钮
    
- 滚动查看模型回答
    
- 整体居中 + 响应式设计
    

> 所有文本支持复制 + 滚动查看超长内容。

### 🧩 项目结构说明

|文件|说明|
|---|---|
|`main.dart`|主程序入口，包含 UI 布局与 HTTP 调用逻辑|
|`pubspec.yaml`|项目依赖项，需添加 `http`|
|`macos/Runner/DebugProfile.entitlements`|允许 macOS App 网络访问 localhost|


## ⚠️ 权限设置（macOS）

为允许 Flutter macOS 桌面程序访问 Ollama 本地服务，请确保在以下两个文件中添加：

- `macos/Runner/DebugProfile.entitlements`
    
- `macos/Runner/Release.entitlements`
    

添加内容
<key>com.apple.security.network.client</key>
<true/>

或在 Xcode 添加 App Sandbox -> Network: Outbound Connections (Client)。
## 🛠 常见问题排查

### ❌ 报错：`Connection failed`

- 可能是 Ollama 模型没运行，需运行
ollama run llama3

或 Ollama 被代理工具拦截，尝试关闭 Clash/V2Ray


### ❌ curl 能用但 Flutter 无响应？

- 你可能运行了 Ollama 的交互模式（进入了 >>>）
    
- 需运行模型为 HTTP 服务模式（不要手动对话）：
killall ollama
ollama run llama3

### ❌ yellow overflow 警告

- 原因：返回文本过长，超出页面
    
- 解决：使用 `SingleChildScrollView + Expanded` 让文本可滚动查看
### 🧠 模型接口格式（调用示例）

POST http://localhost:11434/api/generate
Content-Type: application/json

{
  "model": "llama3",
  "prompt": "Flutter 是什么？",
  "stream": false
}

响应结构：
{
  "response": "Flutter 是 Google 推出的 UI 框架...",
  "done": true
}


### 💄 UI 美化细节

|元素|样式|
|---|---|
|输入框|圆角、白底、带 label|
|按钮|圆角、图标、状态提示|
|回答区域|灰底、边框、滚动、自适应|
|整体布局|居中、固定宽度、统一配色|

## ✅ 已验证环境

- macOS 13+ / 14+
    
- Flutter 3.19+
    
- Ollama 0.1.30+
    
- llama3 模型本地运行成功

