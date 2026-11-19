# 本次改动审阅说明

日期：2026-11-19  
范围：个股 K 线浪型叠加、AI 选浪、条件式交易计划、历史数据扩展、相关 bug 修复。  
用途：换模型做代码审阅时的入口文档；实现细节以文件内注释和测试为准。

## 0. 提交记录

| 提交 | 内容 |
|---|---|
| `fe8bc87` | 初始提交：规则回测、知识库、浪型、AI 选浪、项目文档 |
| `9f5a475` | `start.bat` 打开地址加随机参数，避免浏览器复用旧前端 |
| `41c93bd` | 新增长/短线条件式交易计划 |
| `d47f996` | 图表按需只读扩展日线历史；新增加宽候选开关 |
| `ce0e497` | 修复计划接口无下方支撑/月线缺失时的 500；窄区间改为观望 |

## 1. 如何运行与验证

```bash
# 本地服务（Windows 可直接双击 start.bat）
cd web
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# 全量测试（当前 65 项，约 20 秒）
python -m pytest tests -q
```

关键页面：

- 自选盯盘：分组、规则评分、加减手数
- 个股详情：K 线 + 多级别浪型 + 斐波那契 + 「AI 选浪」+「周期与位置建议」

## 2. 数据流

```
腾讯前复权日/周/月线 ──> kline_daily / weekly / monthly (DuckDB)
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
   /api/kline           /api/wave/ai            /api/plan
   _chart_bars          build_payload           build_levels
   _wave_overlay        validate_choice         _long_plan / _short_plan
        │                     │                      │
        └──────────────┬──────┴──────────┬───────────┘
                       ▼                 ▼
                 前端 KChart.drawWave  「周期与位置建议」卡片
```

`mtf.ensure` 写库仍按原 800 根日线；图表用的更长历史由 `_chart_bars` **只读**拉取，
不写回 `kline_daily`，因此回测样本不被改动。

## 3. 功能与实现

### 3.1 多级别浪型与斐波那契

文件：
- `quant/wave.py`
  - `zigzag()` / `clean_zigzag()`：百分比转折点
  - `_scan_pcts()`：`base = clamp(2.5 × ATR14 / 现价, 3%, 8%)`
  - `wave_groups(df, strict=True)`：小/中/大三档扫描，`DEGREE_MULTS = (1.0, 1.8, 3.2)`
  - `analyze_wave()`：评分用的单结构、近端口径（终点 ≤15 根、偏离浪5 ≤10% 等）
  - `fib_levels()` / `last_swing()`
- `web/app.py`
  - `_wave_overlay()`：把转折点索引转成时间/价格，返回 `groups`、`relaxed`、`candidate`、`fib`
  - `_chart_bars()`：图表取数，日线不够时只读翻页扩展
- `web/static/index.html`
  - `KChart.drawWave()`：一组一色、线宽表示级别；浪2/浪4 虚线；灰色候选/放宽候选；标签避让

口径：
- 同一级别内贪心分段，结构不重叠（上一组浪5 的终点是下一组起点）。
- 不同级别允许嵌套；同一组转折点在多个级别成立时保留更高级别。
- 只有通过五浪规则 + 比例校验（浪2 23.6%~95%、浪4 10%~61.8%、浪3 ≥1.2×最长推动浪）的结构才是彩色浪组。
- 未通过的结构默认不画；`放宽候选` 开关用灰色虚线补出，标注「放」，不参与评分和 Fibonacci 锚点。
- 斐波那契锚点优先级：当前有效五浪 → 最近一组有效浪 → 最近波段。

负价格：
- 高分红股前复权早期价格可能为负；ZigZag 阈值改用 `price ± pct × abs(price)`，不能写 `price × (1 ± pct)`。

### 3.2 AI 选浪

文件：
- `quant/wave_ai.py`
  - `build_payload()`：候选最多 24 组，优先保留最近；含真实转折点日期/价格、回撤、浪3 比例
  - `PROMPT` / `build_prompt()`：只允许在候选编号里选，禁止发明点位
  - `validate_choice()`：编号必须存在于候选集合，否则硬报错
  - `assemble()`：把选择转成接口返回；为选中组生成斐波那契
- `web/app.py`：`POST /api/wave/ai/{symbol}?period=...`
  - 缓存 30 分钟（按标的、周期、最后一根 K 线）
  - 模型不可用/输出越界 → 502，不静默兜底
- 前端：`runAIWave()`、`renderWaveAI()`；选中组加白色光晕、其余变淡，斐波那契锚点同步切换

约束：
- AI 只改图上的主数法与斐波那契锚点，**不进入规则评分**。
- 没有通过规则校验的候选时不调用模型，直接返回「无需选择」。

实测：`sh510300` 日线 3 组候选，模型选中当前小级别下降推动浪，耗时 31.8 秒（`deepseek-v4.1-flash`）。

### 3.3 条件式交易计划

文件：
- `quant/plan.py`
  - `build_levels()`：把浪型斐波那契、日/周均线、近 120 日摆动点、近期高低点、江恩锚点按 0.3 ATR 聚类
  - `_long_plan()`：周/月均线 + 周线浪型；回踩支撑分批加、反弹压力减、跌破失效
  - `_short_plan()`：日线浪型 + 日线 MA20/60 + 江恩窗口
  - `build()`：组装、冲突提示、条件文本
- `web/app.py`：`GET /api/plan/{symbol}`，缓存 60 秒
- 前端：`loadPlan()` / `renderPlan()`，卡片在 K 线下方

规则要点：
- 每个价位带 `sources`（来源）和 `strength`（权重和），不是单一指标。
- 加仓区/减仓区是价位 ± `max(0.25 ATR, 0.3%)`。
- 支撑/压力过近（不足 1 ATR 或现价 2%）时直接给「观望」，不生成两个重叠的区间。
- 长线偏多/短线偏空等冲突会写入 `warnings`，提示执行顺序。
- 月线数据不可用时降级为仅看周线，并写明原因，不整页 500。

## 4. 历史数据扩展与回测隔离

- `quant/data_source.py::fetch_kline_history()`：带 `end` 日期向前翻页，最多 4 页，每页 800 根；只读。
- `web/app.py::_chart_bars()`：日线不足 `limit` 时调用上面的函数；异常只记 warning 并回退到库里的数据。
- `quant/multi_tf.py::ensure()`：仍按 `day=800 / week=740 / month=175` 写库。
- 回测 `run_book_backtest.py` 直接读 `kline_daily`，因此不受图表扩展影响。

## 5. 本次修掉的 bug

| 问题 | 位置 | 修法 |
|---|---|---|
| 小周期 `int too large to convert to float` | `quant/gann7.py::_binom_two_sided` | 二项概率改对数空间计算，按总数归一 |
| 周线负价格序列几乎识别不出转折点 | `quant/wave.py::zigzag` | 阈值按绝对值计算 |
| 计划接口 `KeyError: 'low'` | `quant/plan.py::_long_plan/_short_plan` | 无下方支撑时用加仓区自身下沿作失效位 |
| 月线 501 导致计划整页 500 | `web/app.py::plan_detail` | 降级为仅看周线，写入警告 |
| 窄区间加减仓区重叠 | `quant/plan.py` 两个分歧分支 | 区间不足 1 ATR / 2% 时直接观望 |
| 浏览器复用旧前端 | `start.bat` / `web/start.bat` | 打开地址追加 `?v=%RANDOM%%RANDOM%` |

## 6. 测试

当前 65 项，相关文件：

| 文件 | 覆盖 |
|---|---|
| `tests/test_wave.py` | 推动浪/终结楔形、失效原因、负价格、多级别不重叠 |
| `tests/test_wave_ai.py` | 候选构造、编号校验、结果组装 |
| `tests/test_plan.py` | 价位真实、加仓区在现价下方、失效位、无下方支撑、窄区间 |
| `tests/test_gann7.py` | 窗口投影、因果性、二项检验小样本/大样本 |
| `tests/test_api_helpers.py` | 浪型叠加层时间对齐、`relaxed` 字段 |

## 7. 本地数据变更（不随仓库提交）

自选分组新增（写入本地 `quant/market.duckdb`，`*.duckdb` 已被 `.gitignore`）：

- 半导体：`sh688981` 中芯国际、`sz002371` 北方华创、`sh603501` 韦尔股份
- CPO：`sz300308` 中际旭创、`sz300502` 新易盛、`sz300394` 天孚通信
- 化工：`sh600309` 万华化学、`sh600426` 华鲁恒升、`sz002601` 龙佰集团

调试期间 `sh516080` 的本地日线曾被写到 1393 根（当时尚未做只读隔离）。它不在回测
`UNIVERSE` 内，回测样本不受影响；如需完全一致，可重新拉取该标的的 800 根覆盖。

## 8. 已知限制（审阅时重点关注）

1. **浪型阈值是启发式**：基础阈值用最后一根 ATR/现价校准全历史，老行情波动大时可能切得太碎或太粗；没有按滚动 ATR 分段。
2. **多级别扫描有假阳性**：`wave_groups(strict=False)` 只用于画图，切勿把它接进评分；审阅时检查是否有路径意外使用 `relaxed`。
3. **贪婪分段不是全局最优**：同一级别从最老转折点开始匹配，可能不是人眼会选的起点。
4. **负价格/跨零**：前复权序列跨零时，百分比阈值仍有边界效应。
5. **AI 选浪不可回测**：模型只在候选里选，但选择结果无法像规则一样复现；提示词依赖候选 JSON，需检查候选规模上限（24）与 token。
6. **交易计划不是回测策略**：价位权重（fib 2.0、MA 1.5、摆动点 1.0、江恩 0.8）是人工定的，没有做过收益归因；窄区间阈值（1 ATR/2%）也是经验值。
7. **历史扩展依赖上游**：腾讯接口对大 count 会截断，翻页可绕过，但仍有 501；失败时回退库里 800 根，用户可能看到不同历史长度。
8. **缓存导致改代码后需重启**：`/api/kline` 60 秒、`/api/plan` 60 秒、`/api/wave/ai` 30 分钟；调参后请重启服务。
9. **前端是单文件**：无构建、无前端测试；本次用 Playwright 做了手动回归，但没有落成自动化用例。

## 9. 建议的审阅顺序

1. `quant/wave.py::zigzag` / `wave_groups`：阈值、负价格、贪心分段、去重。
2. `web/app.py::_wave_overlay`：`groups` / `relaxed` / `candidate` 三条路径是否互不污染。
3. `quant/wave_ai.py::validate_choice` / `assemble`：越界编号、空候选、失败兜底。
4. `quant/plan.py::build_levels` / `_long_plan` / `_short_plan`：价位来源、加仓区与失效位关系、窄区间。
5. `web/app.py::_chart_bars` / `data_source.fetch_kline_history`：只读扩展是否真的不写库、翻页终止条件。
6. `quant/gann7.py::_binom_two_sided`：对数空间结果与精确二项检验的偏差。
7. 前端 `KChart.drawWave`、`renderPlan`：标签避让、AI 高亮、XSS（所有动态文本是否走 `esc`）。

## 10. 不在本次范围

- 没有改回测引擎、书籍检索、模拟盘的交易逻辑。
- 没有把新分组/标的写入仓库（属于本地数据）。
- 没有做交易计划的样本外回测，也没有把 AI 选浪接入评分。
