import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const MaterialApp(
      debugShowCheckedModeBanner: false,//这个设置用于控制是否在应用右上角显示 "DEBUG" 标签。
      home: LlamaUI(),//这个设置用于指定应用的入口页面。
    ));

class LlamaUI extends StatefulWidget {
  const LlamaUI({super.key});
//构造函数声明，
  @override
  State<LlamaUI> createState() => _LlamaUIState();//创建状态实例
}

class _LlamaUIState extends State<LlamaUI> {
  final TextEditingController _controller = TextEditingController();
  //创建一个文本编辑控制器，用于处理用户输入
  String _response = '';
  //创建一个字符串变量，用于存储模型回答
  bool _loading = false;
  //创建一个布尔变量，用于控制加载状态

  Future<void> askLlama() async {
    final prompt = _controller.text.trim();//获取用户输入的文本
    if (prompt.isEmpty) {
      setState(() {
        _response = '⚠️ 请输入问题';
      });
      return;
    }

    setState(() {
      _loading = true;//设置加载状态
      _response = '';//设置回答
    });//设置加载状态和回答

    try {
      final res = await http.post(
        Uri.parse('http://localhost:11434/api/generate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'model': 'llama3',
          'prompt': prompt,
          'stream': false,
        }),
      );

      if (res.statusCode == 200) {//如果请求成功，
        final data = jsonDecode(res.body);
        setState(() {
          _response = data['response'];//设置回答
        });
      } else {
        setState(() {
          _response = '❌ 请求失败: ${res.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _response = '❌ 网络错误: $e';
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);//获取当前主题
    return Scaffold(
      backgroundColor: const Color(0xFFF7F7F7),//设置应用栏的背景颜色
      appBar: AppBar(
        backgroundColor: Colors.indigo,//设置应用栏的背景颜色
        title: const Text('🦙 LLaMA3 大模型问答'),//设置应用栏的标题
        centerTitle: true,//设置标题是否居中
        elevation: 2,//设置应用栏的阴影
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 700),//设置约束框的最大宽度
          child: Padding(
            padding: const EdgeInsets.all(24),//设置内边距  
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,//设置列的交叉轴对齐方式
              children: [
                TextField(
                  controller: _controller,//设置文本编辑控制器。
                  decoration: InputDecoration(
                    labelText: '请输入你的问题',//设置输入框的标签文本
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),//设置输入框的圆角
                    ),
                    filled: true,
                    fillColor: Colors.white,
                  ),
                ),
                const SizedBox(height: 16),
                ElevatedButton.icon(
                  onPressed: _loading ? null : askLlama,//设置按钮的点击事件
                  icon: const Icon(Icons.send),//设置按钮的图标
                  label: Text(_loading ? '提问中...' : '向 LLaMA3 提问'),//设置按钮的文本
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.indigo,//设置按钮的背景颜色
                    foregroundColor: Colors.white,//设置按钮的前景色
                    padding: const EdgeInsets.symmetric(vertical: 14),//设置按钮的垂直内边距
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),//设置按钮的圆角
                    ),
                  ),
                ),
                const SizedBox(height: 24),//设置间距
                const Text(
                  '🧠 回答：',//设置回答的文本
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),//设置回答的样式
                ),
                const SizedBox(height: 8),//设置间距
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(16),//设置内边距
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,//设置容器的背景颜色
                      border: Border.all(color: Colors.grey.shade300),//设置容器的边框颜色
                      borderRadius: BorderRadius.circular(12),//设置容器的圆角
                    ),
                    child: SingleChildScrollView(
                      child: SelectableText(
                        _response.isEmpty
                            ? '（等待模型回答...）'
                            : _response,//设置文本的内容
                        style: const TextStyle(fontSize: 16),//设置文本的样式
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
