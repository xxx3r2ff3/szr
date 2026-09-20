# 数字人系统 (SZR2026) 客户端与前端工作台

全本地化运行的数字人播报与创作系统前端及核心服务。

## 目录说明

* `candidate/local_ui/`：前端静态工作台（HTML、CSS、JS、图标组件）。
* `candidate/modules/`：后端 API 路由与业务逻辑服务。
* `candidate/start.py`：系统启动入口。
* `PROJECT_PLAN.md`：项目功能演进规划与台账。

## 快速调试与界面预览

### 方式 1：纯前端所见即所得（无需 Python 与模型权重）
若仅需调整前端按钮、弹窗、样式与页面布局：
1. 直接使用浏览器打开 `candidate/local_ui/index.html`。
2. 或在 `candidate/local_ui/` 目录下启动本地静态服务器：
   ```bash
   python -m http.server 8000
   ```
   访问 `http://localhost:8000` 即可实时查看界面。

### 方式 2：启动后端轻量联调
若需测试 API 响应与按钮请求联动：
```bash
python candidate/start.py
```
