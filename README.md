# S-XGL Interpreter

This repository contains a minimal interpreter for S-XGL (Streamlined Xent Game Language), the language described the paper [Cognitive Training for Language Models: Towards General Capabilities via Cross-Entropy Games](https://arxiv.org/abs/2603.22479).

## Install

```bash
uv sync
```

## Run

Start the LLM server (defaults to `gpt2` on port 30000):

```bash
uv run llm_server.py
```

Then, in a different terminal, run a game:

```bash
uv run s_xgl.py
```

## Options

**`llm_server.py`**

| Argument | Default | Description |
|---|---|---|
| `--model` | `openai-community/gpt2` | HuggingFace model name |
| `--port` | `30000` | Port to listen on |
| `--temperature` | `1.0` | Sampling temperature |
| `--top_k` | `0` | Top-k sampling (0 = disabled) |
| `--top_p` | `1.0` | Nucleus sampling (1.0 = disabled) |
| `--greedy` | off | Use greedy decoding (deterministic) |

**`s_xgl.py`**

| Argument | Default | Description |
|---|---|---|
| `--llm_server_url` | `http://127.0.0.1:30000` | URL of the LLM server |
| `--game_path` | `games/Condense.sxgl` | Path to the `.sxgl` game file |
| `--print_strings` | off | Print strings instead of token ids |

## Games

Game programs are stored as `.sxgl` files in `games/`. Two games are included:

- `games/Pretrain.sxgl` — Pretraining game
- `games/Condense.sxgl` — A simple prompt game
