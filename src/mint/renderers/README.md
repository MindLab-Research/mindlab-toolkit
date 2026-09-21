# MinT 渲染器设计

本目录是 MinT 对 Tinker Cookbook 的模型族适配层。训练、校验、推理采样都应当调用这里的 renderer，不要在 MindForge / 数据集脚本里手拼 `<think>`、角色标记或工具 XML。

改渲染或 mask 行为时，**先改本文再改代码**。本文与实现冲突时，以本文为准并开同一 PR 把实现拉齐。模块 docstring 只复述本文，不另立契约。

MindForge 侧怎么选 renderer、怎么灌 tools，见仓库外文档 `MindForge/docs/sft-tool-rendering-pipeline.md`。GLM JSONL 校验契约见 `/.agents/skills/glm52-sft-validator/references/contract.md`（校验器是本目录的消费者，不是第二份渲染规范）。

## 注册名

| Cookbook 名 | 类 | 用途 |
|---|---|---|
| `MindLab/glm52` | `GLM52Renderer` | GLM-5.2/5.3，thinking 开，effort=max |
| `MindLab/glm52_high_reasoning` | 同一个 `GLM52Renderer(reasoning_effort="high")` | 仅 system 文案改成 High |
| `MindLab/glm52_disable_thinking` | `GLM52DisableThinkingRenderer` | thinking 关，header 预填 `<think></think>` |
| `MindLab/qwen35` | `Qwen35ReasoningRenderer` | Qwen3.5 格式；MindForge 用它覆盖 Qwen3.6 / 3.8 |

`import mint.renderers` 时完成注册。Qwen 工厂固定 `strip_thinking_from_history=False`，历史轮的 think 块留在序列里，与 GLM 一致。这不是 CLI 开关。

`max` / `high` **不是两套渲染或 mask**。官方 chat template 在 thinking 开启时写一行 `<|system|>Reasoning Effort: Max` 或 `… High`，模型把这当成推理强度提示（默认 max；只有显式 `high` 才降档）。线格式、think 块、工具 XML、`supervise_reasoning` 切分完全相同。之所以有两个 Cookbook 名，是因为 `get_renderer(name, tokenizer)` 只收名字、传不进构造参数；实现仍是一个类。按记录覆盖时走 `chat_template_kwargs.reasoning_effort`，或直接 `GLM52Renderer(..., reasoning_effort=...)`，不必再复制一份逻辑。

## 共用契约

### 推理从哪里来

一条 assistant 消息最多一种结构化推理。按这个顺序认：

1. 字段 `reasoning_content: str`（遗留）
2. `content` 里的 `ThinkingPart`（`{"type":"thinking","thinking":"..."}`）
3. 否则，若 string `content` 是 **serving 残片**（见下），先拆成 `reasoning_content` + 剩余 `content`，再走第 1 条
4. 都不是：无结构，`content` 整段是可见答案。list 里的裸字符串、无 `type` 的 `{"text": "..."}` 先收成标准 TextPart，再当普通答案渲染；不要恢复把完整 `<think>...</think>` 块解释成推理的旧规则。上游 VL preprocessor 读的是 `part["type"]`，不收会 `TypeError` / `KeyError`。

1 和 2 同时出现必须报错。新数据用 ThinkingPart。

**Serving 残片**（只在 1、2 都缺时才拆）：生成 prompt 已经预填了 `<think>`，模型接着吐 `REASON</think>ANSWER`。有人把这段原样塞进 `content`，没有开标签。判定：

- `content` 是字符串，或只含 string / text part 的 list（可见文本拼起来再判）
- 含 `</think>`，且**第一个** `</think>` 之前没有完整 `<think>`
- 左边可以为空：`</think>ANSWER` 就是模型不思考、直接闭合
- 右边可以为空：`REASON</think>` 仍是残片（答案丢了）；右边只剩空白（`REASON</think>\n`）同样算空答案。校验器打 warning，renderer 不拒绝

拆法：按第一个 `</think>` 切开，左边 → `reasoning_content`，右边 → 新的 `content`，然后走字段分支。不要另写一套 mask。

| 输入 `content`（无字段、无 ThinkingPart） | 结果 |
|---|---|
| `REASON</think>COMMONCONTENT` | 推理 `REASON`，答案 `COMMONCONTENT` |
| `[{"type":"text","text":"REASON</think>COMMONCONTENT"}]` | 同上（单个 text part） |
| `["REASON</think>COMMONCONTENT"]` | 同上（list 里的裸字符串） |
| `REASON\n</think>\n\nANSWER` | 推理 `REASON\n`，答案 `\n\nANSWER`（换行跟着切开） |
| `REASON</think>see <think>x</think>` | 残片：推理 `REASON`，答案是 `see <think>x</think>`。False 监督的是这段答案，不是 REASON。不要把它当成下一行的「提到标签」 |
| `see <think>x</think> please` | 第一个 `</think>` 前已有完整 `<think>`（`see <think>x`），整段当答案 |
| `<think>\nREASON\n</think>\n\nANSWER` | 同上，整段当答案（完整块抄进 content，不是残片） |
| `<think></think>ANSWER` | 同上（抄来的空块 + 答案，不提升） |
| `</think>ANSWER` | 推理空，答案 `ANSWER`（不思考直接闭合） |
| `REASON</think>` / `REASON</think>\n` | 推理 `REASON`，答案空或只剩空白（数据质量差，校验器 warning） |
| text part + image 等混装 | 不提升（拼不出一段 serving dump）。若 text 里已有 `<think>` / `</think>`，Qwen False 也不叠官方空块 |
| image 上另带 `text` 键 | 不提升（那是媒体信封，不是 serving dump；提升会丢掉图） |
| `ANSWER` | 无 `</think>`，无结构 |

已经有 `reasoning_content` 或 ThinkingPart 时，**不再**扫描 `content` 里的标签；那里的 `<think>` / `</think>` 是可见答案。

字段或 ThinkingPart **载荷内部**的 `</think>` 对 renderer 是惰性字符串，会原样包进结构化 think 块。校验器可以在这些位置判 fatal；那是数据闸门，renderer 不替它做。

### 两个正交开关

| 开关 | 改什么 | 不改什么 |
|---|---|---|
| `supervise_reasoning`（MindForge `--train-on-cot`） | SFT 的 header/output 切分，从而改 loss | 生成 prompt；结构化 CoT 是否出现在序列里 |
| GLM `enable_thinking` / `reasoning_effort` | 模板是否写 think 段、system 里的 effort 文案 | loss 切分 |
| Cookbook `TrainOnWhat` | 哪些角色的 header/output 进 loss | think 块怎么切 |

默认 `supervise_reasoning=True`：监督 think 体、`</think>`、可见答案（以及 GLM 的工具调用）。开标签 `<think>`（及 GLM 的 `<|assistant|>`）在 header 里；常用的 assistant-only `TrainOnWhat` 不给 header 加 loss——推理时生成 prompt 已经预填。`TrainOnWhat.ALL_TOKENS` 仍会监督 header。

`supervise_reasoning=False`：think 块仍留在序列里当上下文，整段（含 `</think>`）挪进 header。常用的 assistant-only `TrainOnWhat` 于是只监督可见答案 / 工具 / 结束边界。`TrainOnWhat.ALL_TOKENS` 仍会给 header（含 CoT）加 loss。

### 无结构时 True 仍监督空 `</think>`

Serving 时生成 prompt 已经以 `<think>` 结尾，模型即使「没有想法」也会先吐一个 `</think>` 再写答案。SFT 要让模型学会这个闭合，所以 **True + 无结构** 必须把空闭合放进 output 并给 loss。**False** 不需要教这个，不要为此再插入空 ThinkingPart（否则会为了对齐 token 序列而改线格式）。

以 `{"role":"assistant","content":"ANSWER"}` 为例（GLM；Qwen 只是空块多换行）：

```
序列:  …<|assistant|><think></think>ANSWER…

True  header:  …<|assistant|><think>          ← 预填，不监督
True  output:  </think>ANSWER…               ← 空闭合 + 答案，都监督

False header:  …<|assistant|><think></think>  ← 空块整段进 header
False output:  ANSWER…                       ← 只监督答案
```

有结构（字段 / ThinkingPart / 残片提升，含空推理）时走结构路径；True 监督「正文 + 闭合 + 答案」，False 只监督答案。不需要再注入空块。

### 禁止事项

- 不要把 `see <think>x</think> please` 这类「答案里提到标签」当成 CoT 去 split。
- 不要把 `REASON</think>see <think>x</think>` 当成上一行。那是残片：`REASON` 是推理，mention 在答案里。
- 已有字段或 ThinkingPart 时，不要再扫一遍 `content` 做提升。
- 不要为了让 True/False 的 `to_ints()` 相等，在 False 路径注入空 ThinkingPart。
- 不要在 Qwen False、无结构、且答案已含 `<think>` **或** `</think>` 时再叠官方空脚手架（字符串、text part、以及残片拒绝后的混装 text 同一判定）。`</think>` 并不包含子串 `<think>`。
- 不要静默丢弃 `reasoning_content`。上游 renderer 不认该字段时，训练侧必须报错。

---

## GLM-5.2

自研 `Renderer`。线格式对齐历史仓库 `MindLab-Research/agent-model-training-mono` 里的 GLM-5.2 SFT Jinja 模板（`tests/renderers/test_glm52_sft_parity.py` 对拍；默认跳过，需本地 checkout，仓库内未钉 SHA）。那是旧训练流水线的模板真值，不是运行时依赖，也不走 cookbook 的 Qwen 模板。

### 线格式

```
[gMASK]<sop>
[<|system|>Reasoning Effort: Max|High]          # 仅 enable_thinking
<|user|>…<|assistant|><think>
  {reasoning}</think>{visible}{tool_calls}{next_boundary}
```

- thinking 开：header = `<|assistant|><think>`
- thinking 关：header = `<|assistant|><think></think>`；消息若仍带结构化推理则报错
- `visible` 写入前会 `strip()`（官方模板 `visible_text` 同样 strip）
- 下一条若是 user / tool，本轮 output 末尾分别跟 `<|user|>` / `<|observation|>`
- `build_supervised_example` 在最后一条 assistant 后再追加一枚 `<|user|>` 边界 token（权重随 `TrainOnWhat`）
- 工具调用是 GLM XML：`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`
- 工具声明的唯一真值是 `create_conversation_prefix_with_tools`（system 文案 + `_mint_glm52_tools`）

输入先按「共用契约」归一：字段 / ThinkingPart / 残片提升 / 纯答案。提升之后与手写 `reasoning_content` 同一条拼接路径。

### Mask

切分在拼字符串时完成，不二次扫描 token。

| | header（通常不监督） | output（通常监督） |
|---|---|---|
| True | `<|assistant|><think>` | `{reasoning}</think>{visible}{tools}{boundary}` |
| False | `<|assistant|><think>{reasoning}</think>` | `{visible}{tools}{boundary}` |

同一条消息上 True/False 的 token 序列必须相同，只改切分。无结构化推理时 `reasoning == ""`，True 的 output 以 `</think>` 开头（空闭合仍受监督）。

`parse_response`（thinking 开）解析的是生成 prompt 之后的采样结果：去掉可选的前缀 `<think>`，再按第一个 `</think>` 切开。推理正文里再写 `</think>` 会截断——这是解析语义，也是校验器对 `reasoning_content` 判 fatal 的原因。thinking 关时只剥恰好漏进采样串的 `<think></think>` 脚手架；裸 `</think>ANSWER` 当可见答案，不当空推理。

残片提升是训练数据启发式，**不是**同一条规则：第一个 `</think>` 之前只要出现过完整 `<think>` 就不拆。因此抄进来的完整块、以及 `see <think>x</think> please`，渲染时整段当答案；不要用 `parse_response` 的行为去改提升。

---

## Qwen3.5

`Qwen3_5Renderer` 子类。上游只认 ThinkingPart，且默认监督整个 think 块。本类先按「共用契约」归一（字段和残片都折成 ThinkingPart），再在已编码的 `header` / `output` 之间搬 token。

### 线格式

生成 prompt 预填 `<think>\n`。有 ThinkingPart 时上游写出：

```
…<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{visible}<|im_end|>
```

空推理（True 注入的空 ThinkingPart，或残片 `</think>ANSWER` 提升后的空载荷）写成 serving 形态 `<think>\n</think>\n\n`，与预填的 `<think>\n` 对齐。官方「已完成的空 CoT」是 `<think>\n\n</think>\n\n`（token 271），只出现在 False + 无结构、且这是最后一条 user 之后的 assistant 轮、可见答案不含 `<think>` / `</think>`（字符串、text part，或残片拒绝后的混装 text）时。这条空块由 **本类覆写的** `_assistant_header_suffix` 放进 header；上游 cookbook 默认只要没有 ThinkingPart 就会写官方空块，带字面标签的答案会被叠第二块。这两种空块有意不对称，不要为了对齐 `to_ints()` 去抹平。

残片提升和 suffix 是两条谓词：提升看「第一个 `</think>` 前有没有完整 `<think>`」；suffix 看「可见文本里有没有任一 think 标签」。`</think>` 并不包含子串 `<think>`。`</think>ANSWER` 先被提升，suffix 不会跑；只有残片拒绝（例如带 image）时，suffix 的 `</think>` 检查才挡住叠块。

- 历史轮 think 块保留（`strip_thinking_from_history=False`）
- 可见答案 **不** `strip()`（与 GLM 不同）
- 畸形 ThinkingPart（`thinking` 非字符串）在渲染路径上抛 `RendererError`。`_assistant_header_suffix` 只看「有没有 thinking part」，不校验载荷

### Mask

有结构时先渲染，再搬 token。无结构的 False 不注入、不搬，只靠 suffix。

| | header（通常不监督） | output（通常监督） |
|---|---|---|
| True（非空推理） | 角色头 + `<think>\n` | `{reasoning}\n</think>\n\n{visible}` |
| True（空推理 / 注入空块） | 角色头 + `<think>\n` | `</think>\n\n{visible}` |
| False + 有结构 | 角色头 + 整个 `<think>…</think>\n\n` | `{visible}` |
| False + 无结构 + 答案无 `<think>` / `</think>` | 角色头 + `<think>\n\n</think>\n\n` | `{visible}` |
| False + 无结构 + 答案已含 `<think>` 或 `</think>` | 角色头（无空块） | `{visible}`（标签在答案里） |

有结构时 True/False 的 token 序列相同，只改切分。无结构时两者 **允许不同**。

True：`_mask_think_open` 只把 output 开头的 `<think>\n` 挪进 header。False + 有结构：`_mask_think_span` 按刚渲染出的完整 think **文本**找 mask 边界，reasoning 体内的 `</think>` 不算块结束。不要假设单独 `encode(think_block)` 一定是完整 output 的 token 前缀：残片切开后答案若以换行开头，会和 `_format_thinking_text` 自带的 `\n\n` 在真实 BPE 下并成更长的换行 token（Qwen3.5 上 `271` 两个换行 vs `987` 四个换行）。此时用已经合并过的 output，取 decode 后覆盖该 think 文本的最短 token 前缀；合并进 think 跨度的那一枚 token 可能带上答案侧几个换行，False 也不监督它们。字符级 tokenizer 复现不了这条，要用真实 tokenizer 或等价的合并假 tokenizer。

---

## 对照

以 `content="ANSWER"`（无结构）和 `reasoning_content="REASON"` + `content="ANSWER"` 为例。残片 `content="REASON</think>ANSWER"` 提升后与第二行相同。`content="</think>ANSWER"` 提升后与 `reasoning_content=""` + `content="ANSWER"` 相同。

**无结构**

| | GLM True | GLM False | Qwen True | Qwen False |
|---|---|---|---|---|
| 序列里的 think | `<think>` + `</think>` | 同左 | `<think>\n</think>\n\n` | `<think>\n\n</think>\n\n` |
| 监督 `</think>` | 是 | 否 | 是 | 否 |
| True/False ids | 相同 | 相同 | **不同** | **不同** |

**有结构 REASON**

| | GLM | Qwen |
|---|---|---|
| 序列 | `<think>REASON</think>ANSWER` | `<think>\nREASON\n</think>\n\nANSWER` |
| True 监督 | `REASON</think>ANSWER` | `REASON\n</think>\n\nANSWER` |
| False 监督 | `ANSWER` | `ANSWER` |
| True/False ids | 相同 | 相同 |

（GLM 可见答案会 `strip()`；Qwen 不 strip。GLM 无结构时空块是 `<think></think>`，中间没有换行。）

---

## 改动检查单

1. 更新本文对应小节（先写契约，再动代码）。
2. 断言完整序列（header+output），不要只数某一种脚手架字面量，也不要只看被监督切片。
3. True/False 各走一遍：无结构、有结构、残片 `REASON</think>ANSWER`、带换行的残片（`REASON\n</think>\n\nANSWER` 等，False 不得因 BPE 抛错）、答案含 `see <think>x</think>`、完整块的 list 裸字符串 / 无 `type` 文本对象（字符串和 text part）。
4. 畸形 ThinkingPart 必须在 `build_supervised_example` 上失败。
5. 生成 prompt 不得随 `supervise_reasoning` 变化。
6. GLM 工具路径若有改动，与校验器、MindForge `native_tools_prefix` 对拍。
