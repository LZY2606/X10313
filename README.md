# ScopeLens — 宏作用域镜：语法对象、卫生重命名与来源展开图

一个**本地优先**的小型宏系统工作台：编译器工具组导入一门小语言（ScopeLens）的
token、词法作用域与宏定义，基于 **scope set（作用域集合）** 展开 syntax object，
解释每一个标识符绑定，并把来源映射、宏调用栈、生成符号、转换步骤、候选与拒绝原因
全部作为一等数据持久化到本地 SQLite。

## 安装（固定命令）

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 演示（固定命令）

```bash
.venv/bin/python -m pytest -q; .venv/bin/uvicorn app:app --host 127.0.0.1 --port 5396
```

成功后打开 <http://127.0.0.1:5396>：

1. 打开「案例包」页，导入离线案例包 **宏作用域镜**（自动逐步重放）；
2. 在任一程序版本上展开，查看候选、作出接受/拒绝/参数调整裁定；
3. 关闭服务、删除/重命名本地数据库后重新导入，稳定身份、顺序与来源关系保持；
4. 使用「版本对比」比较两个宏版本；刷新或换浏览器（数据全在服务端 SQLite）仍保持。

## 领域主张（机器可校验）

- **捕获由 scope 关系判定，而非字符串改名**：宏引入的标识符与使用处标识符携带
  不同 scope 标签（cancellation-mark 对称差），按 scope 交集大小（specificity）剪枝；
- **datum → syntax 的上下文选择必须可见**：`datum->syntax` 是唯一引入外部 scope
  的原语（显式捕获），每一步上下文选择都记录 `ctx_sid`；
- **递归展开有预算且失败保留部分展开图**：预算耗尽时保留已展开部分与失败节点，
  状态为 `budget_exceeded`；无法唯一决定时保留**全部合法候选**与拒绝原因。

## 仓库布局

- `scopelens/` — tokenizer / syntax 对象 / scope-set 展开器 / 绑定解释 / 求值 / 存储 / 事件溯源
- `cases/macro-scope-mirror.json` — 专门的离线案例包（含同名局部变量、嵌套宏、
  definition-site 引用、显式捕获、递归宏、生成临时名、错误来源回映）
- `app.py` / `static/index.html` — FastAPI + 原生 SVG 复核界面
- `tests/` — 固定 seed、临时目录、虚拟时间的自动化测试
