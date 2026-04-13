from contextlib import asynccontextmanager
from dataclasses import dataclass
import argparse

import torch
import uvicorn
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


MODEL_NAME = "openai-community/gpt2"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 1.0


@dataclass
class Runtime:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: torch.device
    eos_token_id: int
    bos_token_id: int
    temperature: float
    top_k: int
    top_p: float
    greedy: bool


class TokenizeRequest(BaseModel):
    text: str


class DetokenizeRequest(BaseModel):
    tokens: list[int]


class XentRequest(BaseModel):
    tokens: list[int]


class GenerateRequest(BaseModel):
    tokens: list[int]
    n: int = Field(gt=0)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_runtime(model_name: str, temperature: float, top_k: int, top_p: float, greedy: bool) -> Runtime:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = select_device()
    model.to(device)
    model.eval()

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError(f"{MODEL_NAME} does not define an EOS token.")

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = eos_token_id

    return Runtime(
        tokenizer=tokenizer,
        model=model,
        device=device,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
    )


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = load_runtime(
        model_name=app.state.args.model,
        temperature=app.state.args.temperature,
        top_k=app.state.args.top_k,
        top_p=app.state.args.top_p,
        greedy=app.state.args.greedy,
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/tokenize")
def tokenize_text(
    body: TokenizeRequest,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, list[int]]:
    return {"tokens": runtime.tokenizer.encode(body.text, add_special_tokens=False)}


@app.post("/detokenize")
def detokenize_text(
    body: DetokenizeRequest,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, str]:
    return {
        "text": runtime.tokenizer.decode(
            body.tokens,
            clean_up_tokenization_spaces=False,
        )
    }


@app.post("/xent")
def compute_xent(
    body: XentRequest,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, float]:

    tokens = [runtime.bos_token_id] + body.tokens

    if len(tokens) < 2:
        return {"xent": 0.0}

    input_ids = torch.tensor([tokens], dtype=torch.long, device=runtime.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        logits = runtime.model(input_ids=input_ids, attention_mask=attention_mask).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    next_token_ids = input_ids[:, 1:].unsqueeze(-1)
    token_log_probs = log_probs.gather(dim=-1, index=next_token_ids).squeeze(-1)
    return {"xent": float(-token_log_probs.sum().item())}


@app.post("/generate")
def generate_tokens(
    body: GenerateRequest,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, list[int]]:

    prompt_tokens = [runtime.bos_token_id] + body.tokens
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=runtime.device)
    prompt_length = len(prompt_tokens)

    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = runtime.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=not runtime.greedy,
            max_new_tokens=body.n,
            suppress_tokens=[runtime.eos_token_id],
            temperature=runtime.temperature if not runtime.greedy else None,
            top_k=runtime.top_k if not runtime.greedy else None,
            top_p=runtime.top_p if not runtime.greedy else None,
        )

    generated_tokens = output_ids[0, prompt_length:].tolist()

    return {"tokens": generated_tokens}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    app.state.args = args
    uvicorn.run(app, host=DEFAULT_HOST, port=args.port)
