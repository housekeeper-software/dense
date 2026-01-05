# Dense — Windows Calculator (C# WinForms)

这是一个最小可运行的 Windows 计算器示例（C#，WinForms），包含：
- 基本四则运算（+ - * /）和幂运算（^）
- 小数、括号优先级
- 常见初等函数：sin, cos, tan, log, exp, sqrt
- 常数：pi, e
- 简单的界面与历史记录

构建与运行
1. 需要 .NET 6 SDK（或更高）并确保支持 Windows 桌面。
2. 在项目目录下运行：
   - dotnet build
   - dotnet run

发布为单文件可执行（示例 x64）：
- dotnet publish -c Release -r win-x64 --self-contained true /p:PublishSingleFile=true -o publish

测试
- 我提供了一个简单 xUnit 测试示例（可选），可用 `dotnet test` 运行。

后续改进建议（对应 issues）
- 更完善的 UI（主题、键盘映射改进）
- 表达式历史持久化
- 更严格的解析器与错误消息本地化
- 生成安装程序（Inno Setup / WiX / MSIX)
