# Flutter + AI 安卓平台开发实战学习笔记（基于 MNIST 推理项目）

本项目展示了如何使用 Flutter 在 Android 平台集成 AI 模型（MNIST 手写数字识别），并在安卓模拟器中实现完整的模型加载、图片选择、推理、显示结果的功能。

---

## ✅ 项目目标

- 使用 Flutter 构建跨平台 UI
    
- 集成 TFLite 模型进行本地推理（来自 Hugging Face）
    
- 实现安卓平台中选择图片识别数字的功能
    
- 适配 Android 模拟器运行并解决兼容性问题
    

---

## 🧱 技术栈

- Flutter 3.29.3
    
- Dart 3.3.x
    
- Android SDK（推荐 API 31~34）
    
- Android NDK 27.0.12077973
    
- TensorFlow Lite 模型：fxmarty/resnet-tiny-mnist（Hugging Face）
    
- 插件依赖：
    
    - `tflite_flutter: ^0.11.0`
        
    - `image_picker: ^1.0.7`
        
    - `image: ^3.2.2`
        

---

## 📦 项目初始化步骤

### 1. 创建项目

```bash
flutter create mnist_ai_flutter
cd mnist_ai_flutter
```

### 2. 配置依赖 `pubspec.yaml`

```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.11.0
  image_picker: ^1.0.7
  image: ^3.2.2
```

```bash
flutter pub get
```

### 3. 下载模型

```bash
huggingface-cli download fxmarty/resnet-tiny-mnist --local-dir ./assets
```

将模型 `model.tflite` 放入 `assets/` 文件夹，并在 `pubspec.yaml` 注册：

```yaml
flutter:
  assets:
    - assets/model.tflite
```

---

## 📱 Android 特殊配置

### 1. 配置 NDK

在 `android/app/build.gradle.kts` 添加：

```kotlin
android {
    ndkVersion = "27.0.12077973"
    ...
}
```

### 2. 添加权限

在 `android/app/src/main/AndroidManifest.xml` 中添加：

```xml
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

---

## 🖼️ 模型推理核心逻辑

### 1. 用户点击按钮后选择图片：

```dart
final XFile? image = await ImagePicker().pickImage(source: ImageSource.gallery);
```

### 2. 图像处理

```dart
final img.Image originalImage = img.decodeImage(File(image.path).readAsBytesSync())!;
final img.Image resizedImage = img.copyResize(originalImage, width: 28, height: 28);
final grayscale = img.grayscale(resizedImage);
```

### 3. 输入预处理

```dart
List<List<List<double>>> input = List.generate(28, (y) =>
    List.generate(28, (x) => [grayscale.getPixel(x, y) & 0xFF / 255.0])
);
```

### 4. 模型加载与推理

```dart
final interpreter = await Interpreter.fromAsset('model.tflite');
interpreter.run(input, output);
```

---

## 🧪 常见问题与解决方案

### ❌ 模拟器中“Device not authorized”

```bash
adb kill-server
adb start-server
flutter devices
```

### ❌ OpenGL 报错：`called unimplemented OpenGL ES API`

- 解决方案 1：改用真机调试
    
- 解决方案 2：模拟器设置 Graphics 为 Hardware GLES 2.0
    
- 解决方案 3：加参数运行：
    

```bash
flutter run --enable-software-rendering
```

### ❌ 模型版本不兼容 Dart 3.3

使用 `tflite_flutter: ^0.11.0` 及以上，旧版本（如 0.10.3）会使用已移除方法 `UnmodifiableUint8ListView` 报错。

---

## 🎯 拓展方向建议

|方向|内容|
|---|---|
|拍照识别|加入 `camera` 插件实现相机实时拍摄推理|
|手写输入|自绘画布 Canvas 实现 MNIST 手写识别|
|本地语音播报|使用 `flutter_tts` 播报识别数字|
|多模型切换|用下拉菜单切换多个 AI 模型进行推理|

---

## 📚 参考资源

- Hugging Face 模型：[https://huggingface.co/fxmarty/resnet-tiny-mnist](https://huggingface.co/fxmarty/resnet-tiny-mnist)
    
- Flutter 文档：[https://flutter.dev/docs](https://flutter.dev/docs)
    
- tflite_flutter 插件：[https://pub.dev/packages/tflite_flutter](https://pub.dev/packages/tflite_flutter)