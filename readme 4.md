# Flutter + TensorFlow Lite C API (macOS 桌面项目)

本项目基于 Flutter macOS 桌面应用，结合 TensorFlow Lite 的 C API (`libtensorflowlite_c.dylib`) 实现模型加载与推理。

---

## ✅ 项目目标

- 在 Flutter macOS 桌面环境中运行本地 TFLite 模型
    
- 通过 Dart FFI 调用 `libtensorflowlite_c.dylib` 完成模型初始化与推理
    

---

## 🧱 环境配置

- Flutter 版本：3.x
    
- macOS：Apple Silicon (ARM64)
    
- 已安装工具：
    
    - Xcode
        
    - VSCode / Terminal
        
    - Bazel（若选择自行编译 TFLite）
        

---

## ⚠️ 遇到的问题与解决方案

### ❌ 问题 1：TFLite 动态库加载失败（找不到 `.dylib` 文件）

**报错：**

```
Failed to load dynamic library '.../ai_meme.app/Contents/Resources/libtensorflowlite_c-mac.dylib'
```

**原因：** Flutter 构建系统不会自动将 `.dylib` 放入 macOS App Bundle 的 `Resources` 目录

**解决：**

1. 将 `libtensorflowlite_c-mac.dylib` 放入 `macos/Runner/`
    
2. 打开 `macos/Runner.xcodeproj`
    
3. 添加到 `Build Phases` → `Copy Bundle Resources`
    
4. Flutter 重新构建：
    

```bash
flutter clean
flutter pub get
flutter run -d macos
```

---

### ❌ 问题 2：编译 TensorFlow Lite 源码失败

#### 错误示例：

```bash
#error "TF_MAJOR_VERSION is not defined!"
```

**原因：** 直接构建 `tensorflow/lite/c` 子目录会缺少主版本定义

**解决方法：** 从 `tensorflow/lite` 顶层构建，而不是 `lite/c`

#### 错误示例：farmhash 构建失败

```bash
Build step for farmhash failed: 2
```

**解决方法：**

- 替代：改用官方预编译的 `.dylib` 动态库
    
- 下载链接：使用 [WasmEdge 提供的 TFLite 依赖](https://github.com/second-state/WasmEdge-tensorflow-deps)
    

---

### ❌ 问题 3：Flutter 中模型或资源未加载

**报错：**

```
Unable to load asset: model.tflite
```

**解决：**

- 确保 `pubspec.yaml` 中声明了模型资源：
    

```yaml
flutter:
  assets:
    - assets/model.tflite
    - assets/tokenizer.json
```

- 执行：
    

```bash
flutter pub get
```

---

## ✅ 正确的加载流程（核心代码）

### Dart 中通过 FFI 加载 dylib

```dart
final dylib = DynamicLibrary.open('libtensorflowlite_c-mac.dylib');
final versionPtr = dylib
    .lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>('TfLiteVersion');
print("TFLite version: \${versionPtr().cast<Utf8>().toDartString()}");
```

---

## 📦 模型文件与依赖位置说明

|文件|说明|
|---|---|
|`libtensorflowlite_c-mac.dylib`|放入 `macos/Runner/` 并添加到 Xcode 资源拷贝中|
|`model.tflite`, `tokenizer.json`|放入 `assets/` 并在 `pubspec.yaml` 中声明|

---

## 🔚 总结建议

- 尽量使用官方或社区提供的 `.dylib` 而不是自行构建（降低风险）
    
- 动态库必须加入 Xcode 的 `Copy Bundle Resources`
    
- 使用 `flutter run -d macos` 运行时注意检查 `.dylib` 路径是否在 `.app/Contents/Resources/` 中
    

如需更复杂的模型推理调用绑定或模型输入输出映射示例，请参考 TensorFlow 官方的 C API 文档。