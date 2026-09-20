# -*- coding: utf-8 -*-
"""sparkai 最小离线替身(R040 环境域依赖,登记见 reports/modules/rag__LLM-impl.md)。

原版 modules/rag/LLM.cp310-win_amd64.pyd 在模块级导入 sparkai.llm.llm 与
sparkai.core.messages,而冻结运行时不带该三方包(全树无 sparkai)。本包提供
接口面等价的离线替身(类名/方法名以 E2 串表为准),响应固定且可记录,不做任何
网络访问;放置于 modules/rag/ 下与真实应用的 sys.path 形态一致
(app_sys_path 会把 modules/<pkg> 加入 sys.path)。
"""
