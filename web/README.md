# 技术分析知识库 · Web 服务

把五本技术分析书做成可检索、可溯源的问答服务。本地直接跑，也能原样部署到云。

## 结构

```
web/
  app.py            FastAPI 服务（检索 / 问答 / 行情 / 浪型 / 静态页）
  static/index.html 单页前端，零外部依赖
  requirements.txt
  Dockerfile
```

检索逻辑复用 `../_distill/rag/`：父块 1500 个，子块 4144 个，
BM25 + bge-m3 向量 + RRF 融合。

## 语料构成

| 来源 | 块数 | 字数 | 说明 |
|---|---|---|---|
| 书籍 | 2607 | 76.7 万 | 五本经典 + 凯利公式，可按书名和页码溯源 |
| UP 主问答 | 482 | 7.7 万 | B 站充电问答，426 条已回答 |
| UP 主评论 | 554 | 16.2 万 | 主评论 + 楼中楼回复，按日期标注 |
| 社区讨论 | 501 | 35.6 万 | 其他用户发言，检索时默认不勾选 |

`/api/search` 支持 `sources` 过滤：`["book"]` / `["qa","comment"]` / `["community"]`。

## 本地启动

```bash
cd web
../.venv/Scripts/python -m uvicorn app:app --host 127.0.0.1 --port 8000
# 或用自己的虚拟环境
uvicorn app:app --reload --port 8000
```

打开 http://127.0.0.1:8000 。首次查询会加载嵌入模型，约 10–30 秒，之后正常。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/books` | 书单 |
| POST | `/api/search` | 仅检索，约 200ms |
| POST | `/api/ask` | 检索 + 模型生成答案，约 30s |
| GET | `/api/watchlist` | 自选列表（实时行情 + 日/周/月三周期结论） |
| POST/DELETE | `/api/watchlist[/{code}]` | 增加 / 移除自选 |
| GET | `/api/kline/{symbol}?period=day\|week\|month\|m60\|m30\|m15\|m5` | K 线 + MA5/10/20/60 + `wave`（多级别浪组与斐波那契位） |
| POST | `/api/wave/ai/{symbol}?period=day` | 模型在规则枚举的浪组里选主数法，返回编号、理由与切换后的斐波那契位；约 30s，按标的/周期缓存 30 分钟 |
| GET | `/api/detail/{symbol}` | 个股详情（行情 + 完整规则依据） |
| GET | `/api/plan/{symbol}` | 长线/短线条件式计划：支撑压力、加减仓区间、失效位与条件；约 1s |
| POST | `/api/review` | 调用模型，对整份自选做一次复核点评 |

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"跳空缺口有哪几种","top_k":5,"books":["b2"]}'
```

`books` 不传表示全库；`top_k` 1–20。返回结果带 `book_title / page_start / page_end`，
可直接对应到原书页码。

## 部署到腾讯云

**最小配置 2 核 4G**（模型常驻约 2.5G 内存）。需要公网访问就开安全组 8000 端口，
但更建议只开 80/443，用 Nginx 反代。

### 方式一：Docker

```bash
# 在项目根目录构建（上下文必须包含 _distill）
docker build -f web/Dockerfile -t ta-kb .
docker run -d --name ta-kb -p 8000:8000 \
  -e OCR_API_BASE="https://opencode.ai/zen/go/v1" \
  -e OCR_API_KEY="你的key" \
  -e OCR_MODEL="qwen3.8-flash" \
  ta-kb
```

### 方式二：直接跑 + systemd

```ini
# /etc/systemd/system/ta-kb.service
[Unit]
Description=TA Knowledge Base
After=network.target

[Service]
WorkingDirectory=/opt/ta/web
Environment=OCR_API_BASE=https://opencode.ai/zen/go/v1
Environment=OCR_API_KEY=你的key
Environment=OCR_MODEL=qwen3.8-flash
ExecStart=/opt/ta/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload && systemctl enable --now ta-kb
```

## 注意事项

- **密钥不要写进代码或镜像**，只通过环境变量注入；`.env` 文件不要提交。
- `/api/ask` 每次调用会走外部模型接口，约 30 秒；批量场景请用 `/api/search` 自己拼提示词。
- `/api/wave/ai` 同样会走模型接口，约 30 秒；结果按标的、周期和最后一根 K 线缓存 30 分钟，重复点击不重复计费。模型只返回候选编号，编号越界会直接报 502。
- 对外提供服务时建议加一层鉴权（Nginx Basic Auth 或网关），当前接口是开放的。
- 嵌入模型首次启动会从 HuggingFace 下载；国内机器设置 `HF_ENDPOINT=https://hf-mirror.com`。
