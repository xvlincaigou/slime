from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool

from .bm25_retriever import get_retriever

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

QUERY_TEMPLATE_NO_GET_DOCUMENT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.
Question: {Question}
Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}""".strip()

BROWSECOMP_PLUS_CONFIGS = {
    "max_tool_calls": int(os.environ.get("BROWSECOMP_PLUS_MAX_TOOL_CALLS", "20")),
    "max_turns": int(os.environ.get("BROWSECOMP_PLUS_MAX_TURNS", "30")),
    "topk": int(os.environ.get("BROWSECOMP_PLUS_TOPK", "5")),
    "snippet_max_tokens": int(os.environ.get("BROWSECOMP_PLUS_SNIPPET_MAX_TOKENS", "512")),
    "corpus_root": os.environ.get("BROWSECOMP_PLUS_CORPUS_ROOT", "/root/browsecomp-plus"),
    "turn_hit_weight_miss": float(os.environ.get("BROWSECOMP_PLUS_TURN_HIT_WEIGHT_MISS", "0.5")),
    "turn_hit_weight_hit": float(os.environ.get("BROWSECOMP_PLUS_TURN_HIT_WEIGHT_HIT", "1.0")),
    "tool_parser": os.environ.get("BROWSECOMP_PLUS_TOOL_PARSER", "qwen25"),
}


@dataclass
class ParsedAssistantAction:
    normal_text: str
    calls: list[dict[str, Any]]


def _maybe_load_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_answer(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^0-9A-Za-z\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]+", " ", text.lower()),
    ).strip()


def exact_match(prediction: str, answer: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(answer))


def extract_final_answer(text: str) -> str:
    match = re.search(r"Exact Answer:\s*(.+?)(?:\nConfidence:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def compute_length_bonus(tool_calls: int) -> float:
    capped_calls = min(tool_calls, BROWSECOMP_PLUS_CONFIGS["max_tool_calls"])
    return max(0.0, 1.0 - (capped_calls / BROWSECOMP_PLUS_CONFIGS["max_tool_calls"]))


def compute_scheduled_reward(acc_reward: int, tool_calls: int, rollout_id: int | None, evaluation: bool) -> float:
    length_bonus = compute_length_bonus(tool_calls)
    if not evaluation and rollout_id is not None and rollout_id < 5:
        return 0.8 * acc_reward + 0.2 * length_bonus
    return 0.9 * acc_reward + 0.1 * length_bonus


def render_docs(docs, tokenizer) -> str:
    rendered = []
    for doc in docs:
        snippet_tokens = tokenizer.encode(doc.text, add_special_tokens=False)
        if len(snippet_tokens) > BROWSECOMP_PLUS_CONFIGS["snippet_max_tokens"]:
            snippet_tokens = snippet_tokens[: BROWSECOMP_PLUS_CONFIGS["snippet_max_tokens"]]
        snippet = tokenizer.decode(snippet_tokens, skip_special_tokens=True)
        rendered.append(
            {
                "docid": doc.docid,
                "score": doc.score,
                "snippet": snippet,
            }
        )
    return json.dumps(rendered, ensure_ascii=False)


def parse_tools(response: str, tools: list[dict[str, Any]], parser_name: str) -> ParsedAssistantAction:
    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools
    ]
    parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser_name)
    normal_text, calls = parser.parse_non_stream(response)
    return ParsedAssistantAction(normal_text=normal_text, calls=[call.model_dump() for call in calls])


def get_token_delta(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    curr = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

    if messages[-1]["role"] == "assistant":
        prev = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, tools=tools)
        new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
        return new_tokens, [1] * len(new_tokens)

    prev = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=False, tools=tools)
    new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
    return new_tokens, [0] * len(new_tokens)


def build_search_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Performs a search on a knowledge source. Returns the top-5 results with docid, score, and snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to send to the local BM25 retriever.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def extract_tool_query(parsed: ParsedAssistantAction) -> str | None:
    if not parsed.calls:
        return None
    tool_call = parsed.calls[0]
    if tool_call.get("name") != "search":
        return None
    try:
        args = json.loads(tool_call.get("parameters", "{}"))
    except json.JSONDecodeError:
        return None
    query = args.get("query")
    return query if isinstance(query, str) else None


def build_prompt_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question=question)}]


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported."

    sample.label = _maybe_load_json(sample.label)
    sample.metadata = _maybe_load_json(sample.metadata) or {}

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    tools = sample.metadata.get("tools") or build_search_tools()
    messages = sample.prompt if isinstance(sample.prompt, list) else build_prompt_messages(str(sample.prompt))
    prompt_text = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    advantage_weights: list[float] = []
    turn_hits: list[int] = []
    tool_calls = 0
    answered = False
    final_answer = ""
    gold_docids = set(sample.label["gold_docids"])
    retriever = get_retriever(BROWSECOMP_PLUS_CONFIGS["corpus_root"])

    for _ in range(BROWSECOMP_PLUS_CONFIGS["max_turns"]):
        text_input = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
        output = await post(
            url,
            {
                "text": text_input,
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
        )

        finish_type = output["meta_info"]["finish_reason"]["type"]
        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        response = output["text"]
        token_logprobs = output["meta_info"]["output_token_logprobs"]
        assistant_token_ids = [item[1] for item in token_logprobs]
        assistant_log_probs = [item[0] for item in token_logprobs]

        try:
            parsed = parse_tools(response, tools, BROWSECOMP_PLUS_CONFIGS["tool_parser"])
        except Exception as exc:
            logger.warning("Failed to parse BrowseComp-Plus tool call, treating as final answer: %s", exc)
            parsed = ParsedAssistantAction(normal_text=response, calls=[])

        messages.append({"role": "assistant", "content": response})
        delta_tokens, delta_loss_mask = get_token_delta(state.tokenizer, messages, tools)
        if len(delta_tokens) != len(assistant_token_ids):
            delta_tokens = assistant_token_ids
            delta_loss_mask = [1] * len(assistant_token_ids)

        response_token_ids.extend(delta_tokens)
        loss_mask.extend(delta_loss_mask)
        rollout_log_probs.extend(assistant_log_probs)

        query = extract_tool_query(parsed)
        if query is None:
            answered = True
            final_answer = extract_final_answer(parsed.normal_text.strip() or response.strip())
            advantage_weights.extend([1.0] * len(delta_tokens))
            break

        if tool_calls >= BROWSECOMP_PLUS_CONFIGS["max_tool_calls"]:
            advantage_weights.extend([1.0] * len(delta_tokens))
            break

        tool_calls += 1
        docs = retriever.search(query, topk=BROWSECOMP_PLUS_CONFIGS["topk"])
        hit = int(any(doc.docid in gold_docids for doc in docs))
        turn_hits.append(hit)
        turn_weight = (
            BROWSECOMP_PLUS_CONFIGS["turn_hit_weight_hit"]
            if hit
            else BROWSECOMP_PLUS_CONFIGS["turn_hit_weight_miss"]
        )
        advantage_weights.extend([turn_weight] * len(delta_tokens))

        tool_message = {
            "role": "tool",
            "name": "search",
            "content": render_docs(docs, state.tokenizer),
        }
        messages.append(tool_message)
        tool_delta_tokens, tool_delta_loss_mask = get_token_delta(state.tokenizer, messages, tools)
        response_token_ids.extend(tool_delta_tokens)
        loss_mask.extend(tool_delta_loss_mask)
        rollout_log_probs.extend([0.0] * len(tool_delta_tokens))
        advantage_weights.extend([1.0] * len(tool_delta_tokens))

        if finish_type == "length":
            break

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.response = "\n".join(msg["content"] for msg in messages if msg["role"] == "assistant")
    sample.train_metadata = {
        "advantage_weights": advantage_weights,
        "turn_hits": turn_hits,
        "tool_calls": tool_calls,
    }
    sample.metadata["tool_calls"] = tool_calls
    sample.metadata["turn_hits"] = turn_hits
    sample.metadata["final_answer"] = final_answer
    sample.metadata["tools"] = tools

    if len(advantage_weights) != sample.response_length:
        raise ValueError(
            f"advantage_weights length {len(advantage_weights)} != response length {sample.response_length}"
        )

    if answered:
        sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED

    return sample


async def reward_func(args, sample: Sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("sample must be a Sample instance.")

    sample.label = _maybe_load_json(sample.label)
    sample.metadata = _maybe_load_json(sample.metadata) or {}
    final_answer = extract_final_answer(sample.metadata.get("final_answer", ""))
    acc_reward = exact_match(final_answer, sample.label["answer"])
    tool_calls = int(sample.metadata.get("tool_calls", 0))
    rollout_id = kwargs.get("rollout_id")
    evaluation = bool(kwargs.get("evaluation", False))
    final_reward = compute_scheduled_reward(acc_reward, tool_calls, rollout_id, evaluation)
    sample.metadata["raw_reward"] = acc_reward
    sample.metadata["acc_reward"] = acc_reward
    sample.metadata["length_bonus"] = compute_length_bonus(tool_calls)
    sample.metadata["reward_schedule"] = (
        "warmup_0.8_0.2" if (not evaluation and rollout_id is not None and rollout_id < 5) else "main_0.9_0.1"
    )

    logger.info(
        "browsecomp_plus reward=%.4f acc=%d tool_calls=%d rollout_id=%s evaluation=%s turn_hits=%s pred=%r gt=%r",
        final_reward,
        acc_reward,
        tool_calls,
        rollout_id,
        evaluation,
        sample.metadata.get("turn_hits", []),
        final_answer,
        sample.label["answer"],
    )
    return final_reward
