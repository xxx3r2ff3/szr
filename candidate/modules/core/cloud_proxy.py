# -*- coding: utf-8 -*-
"""modules.core.cloud_proxy —— T4R 语义重建(R037)。

源 pyd:modules/core/cloud_proxy.cp310-win_amd64.pyd
        sha256 abb3216ea359838fed02e8574ccd9b62cc5a0ab971a2ac0cb7a7911cec9679cd

二进制出处(Ghidra 12.1.3 反编译,证据 runtime/ghidra/):
  * create_proxy_app @0x180005990;proxy_health_check 协程体 @0x180005630;
    proxy_middleware 协程体 @0x1800014f0(16KB,状态机)
  * run_proxy @0x180006610;模块 exec @0x180007ed0
  * 缓存常量(InitCachedConstants):500 / 502 / 504 / timeout=600 /
    __main__ 块 kwargs {target_url:<第三方上游 URL>, port:9887, host:"0.0.0.0"}
    ([COM-D002] 原 kwargs 里的第三方上游 URL 字面量已删除:target_url 改由
    SZR_PROXY_TARGET_URL 显式提供,未配置即硬失败;port/host 取值不变)

E1 探针(evidence/modules/core__cloud_proxy/runtime/):
  probeC_cloud_proxy.py / probeD_cloud_proxy_mw.py(本地 127.0.0.1 假上游,零外网)。

行为要点(探针实测):
  * app 元数据:title="HTTP Proxy API" / version="1.0.0" /
    description=f"Proxy all requests to {target_url}";路由含 /proxy-health。
  * 中间件是"全代理":仅当 path != "/proxy-health" 时自行转发(probeD
    no_route_health:删掉路由后 /proxy-health 返回 404 且上游零请求);
    其余路径不经过 call_next。
  * 请求头过滤:host/content-length/connection/accept-encoding/
    upgrade-insecure-requests 丢弃(probeD req_headers:content-encoding 与
    transfer-encoding 原样透传),随后补 headers["host"]=urlparse(target_url).netloc。
  * 转发:httpx.AsyncClient(timeout=600, verify=False) +
    client.request(method=..., url=target_path, headers=..., content=body);
    body 仅 POST/PUT/PATCH 读取,其余为 None;target_path = target_url + path
    (+ "?" + query,query 非空时)。
  * 响应头:dict(response.headers) 后 pop content-encoding / transfer-encoding,
    update {"X-Proxy-Server": "FastAPI-HTTP-Proxy", "X-Target-Server": target_url};
    返回 Response(content=response.content, status_code=response.status_code, headers=...)。
  * 异常:ConnectError→HTTPException(502, "Cannot connect to target server");
    TimeoutException→HTTPException(504, "Gateway timeout");
    其他→HTTPException(500, f"Proxy error: {e}")。
  * run_proxy 先建 app,再打四行启动横幅,最后 uvicorn.run(app, host=, port=,
    log_level="info")。
"""
import os
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import httpx
from httpx import AsyncClient
import uvicorn


def create_proxy_app(target_url: str) -> FastAPI:
    app = FastAPI(
        title="HTTP Proxy API",
        description=f"Proxy all requests to {target_url}",
        version="1.0.0",
    )

    @app.middleware("http")
    async def proxy_middleware(request: Request, call_next):
        if request.url.path != "/proxy-health":
            headers_to_remove = ["host", "content-length", "connection",
                                 "accept-encoding", "upgrade-insecure-requests"]
            headers = {key: value for key, value in request.headers.items()
                       if key.lower() not in headers_to_remove}
            headers["host"] = urlparse(target_url).netloc

            query = str(request.url.query)
            target_path = target_url + request.url.path
            if query:
                target_path = f"{target_path}?{query}"

            body = None
            if request.method in ("POST", "PUT", "PATCH"):
                body = await request.body()

            try:
                async with httpx.AsyncClient(timeout=600, verify=False) as client:
                    response = await client.request(
                        method=request.method,
                        url=target_path,
                        headers=headers,
                        content=body,
                    )
            except httpx.ConnectError:
                raise HTTPException(status_code=502, detail="Cannot connect to target server")
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="Gateway timeout")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Proxy error: {e}")

            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.update({
                "X-Proxy-Server": "FastAPI-HTTP-Proxy",
                "X-Target-Server": target_url,
            })
            return Response(content=response.content,
                            status_code=response.status_code,
                            headers=response_headers)

        return await call_next(request)

    @app.get("/proxy-health")
    async def proxy_health_check():
        return JSONResponse(content={
            "status": "healthy",
            "proxy_to": target_url,
            "service": "FastAPI HTTP Proxy",
        })

    return app


def run_proxy(target_url: str, port: int = 3000, host: str = "0.0.0.0"):
    app = create_proxy_app(target_url)
    print("Starting HTTP Proxy server...")
    print(f"Listening on: http://{host}:{port}")
    print(f"Proxying to: {target_url}")
    print(f"Health check: http://{host}:{port}/proxy-health")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # [COM-D002 语义替换] 原 __main__ 把上游 target_url 写死为第三方站点字面量。
    # 该字面量已删除:上游改为显式配置(环境变量 SZR_PROXY_TARGET_URL);未配置时
    # **硬失败**并给出明确错误,绝不回落到任何内置第三方/原厂主机。
    _target_url = os.environ.get("SZR_PROXY_TARGET_URL", "").strip()
    if not _target_url:
        raise SystemExit(
            "proxy target_url 未配置：请设置环境变量 SZR_PROXY_TARGET_URL"
            "（客户端不内置任何上游主机）。")
    run_proxy(target_url=_target_url, port=9887, host="0.0.0.0")
