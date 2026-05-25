# Adapted from generate_with_search.py
# Uses GENERATION_SYSTEM_PROMPT and RACE reward via DeepSeek V4 Flash
#
# Key differences from generate_with_search.py:
#   1. System prompt = GENERATION_SYSTEM_PROMPT (report-style, matching evaluate.py)
#   2. Reward = RACE scoring via DeepSeek judge (no EM, no FACT)
#       - Training samples: point_wise_score_prompt with static criteria → score/10
#       - Eval samples (DRB): comparative merged_score_prompt with per-query criteria
#         and reference article → target/(target+ref), aligned with evaluate.py
#   3. max_turns = 100, max_context_tokens = 65536, per-turn response ≤ 4096
#   4. --apply-chat-template should be REMOVED from the shell script;
#      this file constructs the prompt itself with the system message.
#   5. Uses Qwen3 native tool-calling format (<tool_call>/<tool_response>)
#      instead of Search-R1 tags (<search>/<information>/<answer>).

import asyncio
import json
import os
import random
import re

import tiktoken

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# ══════════════════════════════════════════════════════════════
# System prompt (same as evaluate.py)
# ══════════════════════════════════════════════════════════════
GENERATION_SYSTEM_PROMPT = (
    "You are a rigorous research assistant. Given a research question, "
    "produce a comprehensive, well-structured research article in Markdown "
    "with inline citations and a Sources section at the end."
)

# ══════════════════════════════════════════════════════════════
# Tool schemas for Qwen3 chat template (same tools as evaluate.py)
# ══════════════════════════════════════════════════════════════
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web and return ranked results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_num_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Open a webpage URL and return its text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to open.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens of page content to return.",
                        "default": 2048,
                    },
                },
                "required": ["url"],
            },
        },
    },
]

# ══════════════════════════════════════════════════════════════
# Scoring prompt — uses DRB's point_wise_score_prompt with static criteria
# (same structure as evaluate.py, but single-article instead of comparative)
# ══════════════════════════════════════════════════════════════
# Static criteria matching DeepResearchBench's generate_static_score_prompt
STATIC_CRITERIA_LIST = """\
# Comprehensiveness
[
  {{"criterion": "Information Coverage Breadth", "explanation": "Evaluates whether the article covers all key areas and aspects related to the topic without omitting important information.", "weight": 0.25}},
  {{"criterion": "Information Depth and Detail", "explanation": "Evaluates whether the article provides sufficiently detailed information rather than just surface-level overviews.", "weight": 0.25}},
  {{"criterion": "Data and Factual Support", "explanation": "Evaluates whether the article provides sufficient data, facts, cases, or evidence to support its arguments and analysis.", "weight": 0.25}},
  {{"criterion": "Multiple Perspectives and Balance", "explanation": "Evaluates whether the article considers issues from multiple angles and provides balanced viewpoints where relevant.", "weight": 0.25}}
]

# Insight
[
  {{"criterion": "Analysis Depth and Originality", "explanation": "Evaluates whether the article provides deep analysis and original insights rather than simply repeating known information.", "weight": 0.25}},
  {{"criterion": "Logical Reasoning and Causal Relationships", "explanation": "Evaluates whether the article demonstrates clear logical reasoning and effectively explains causal relationships behind phenomena.", "weight": 0.25}},
  {{"criterion": "Problem Insight and Solutions", "explanation": "Evaluates whether the article identifies key issues or challenges and provides insightful solutions or recommendations.", "weight": 0.25}},
  {{"criterion": "Forward-Looking and Inspirational Thinking", "explanation": "Evaluates whether the article demonstrates forward-looking thinking, can anticipate trends, and provides inspiring perspectives.", "weight": 0.25}}
]

# Instruction Following
[
  {{"criterion": "Response to Task Objectives", "explanation": "Evaluates whether the article directly responds to the core objectives and questions of the task.", "weight": 0.34}},
  {{"criterion": "Adherence to Scope Limitations", "explanation": "Evaluates whether the article strictly adheres to the scope limitations set in the task (e.g., geography, time period, subjects).", "weight": 0.33}},
  {{"criterion": "Complete Coverage of Task Requirements", "explanation": "Evaluates whether the article completely covers all sub-questions or aspects raised in the task without omitting important parts.", "weight": 0.33}}
]

# Readability
[
  {{"criterion": "Clear Structure and Logic", "explanation": "Evaluates whether the article has a clear structure, including appropriate introduction, body, conclusion, and logically coherent paragraph organization.", "weight": 0.25}},
  {{"criterion": "Language Expression and Fluency", "explanation": "Evaluates whether the article's language is clear, accurate, and fluent, without obvious grammatical errors or inappropriate expressions.", "weight": 0.25}},
  {{"criterion": "Appropriate Use of Technical Terms", "explanation": "Evaluates whether the article appropriately uses technical terminology and provides explanations when necessary for understanding.", "weight": 0.25}},
  {{"criterion": "Information Presentation and Visual Elements", "explanation": "Evaluates whether the article effectively uses formatting, headings, lists, emphasis, etc. to enhance readability, and appropriately uses charts or other visual elements (if any).", "weight": 0.25}}
]"""

# DRB point_wise_score_prompt (single article evaluation, same as DRB score_prompt_en.py)
RACE_SCORE_PROMPT = """\
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to thoroughly evaluate research articles, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate a research article written for this task. We will assess the article across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Article to Evaluate**
<target_article>
"{article}"
</target_article>

**Evaluation Criteria**
Now, you need to evaluate this article based on the following **evaluation criteria list**, providing analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
""" + STATIC_CRITERIA_LIST + """
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate `<target_article>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how the article fulfills the requirements of each criterion.
2.  **Analysis and Evaluation**: Analyze the article's performance on each criterion, referencing the content and criterion explanation, noting strengths and weaknesses.
3.  **Score**: Based on your analysis, score the article on each criterion (0-10 points).

**Scoring Rules**
For each criterion, score the article on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{{{{
    "comprehensiveness": [
        {{{{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Analysis],
            "target_score": [Continuous score 0-10]
        }}}},
        ...
    ],
    "insight": [
        {{{{
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Analysis],
            "target_score": [Continuous score 0-10]
        }}}},
        ...
    ],
    ...
}}}}
</output_format>

Now, please evaluate the article based on the research task and criteria, providing detailed analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
"""

# Static criteria data in the same format as DRB criteria.jsonl
# Used by _compute_weighted_score() identically to evaluate.py's calculate_weighted_scores()
STATIC_CRITERIA_DATA = {
    "dimension_weight": {
        "comprehensiveness": 0.3,
        "insight": 0.25,
        "instruction_following": 0.25,
        "readability": 0.2,
    },
    "criterions": {
        "comprehensiveness": [
            {"criterion": "Information Coverage Breadth", "weight": 0.25},
            {"criterion": "Information Depth and Detail", "weight": 0.25},
            {"criterion": "Data and Factual Support", "weight": 0.25},
            {"criterion": "Multiple Perspectives and Balance", "weight": 0.25},
        ],
        "insight": [
            {"criterion": "Analysis Depth and Originality", "weight": 0.25},
            {"criterion": "Logical Reasoning and Causal Relationships", "weight": 0.25},
            {"criterion": "Problem Insight and Solutions", "weight": 0.25},
            {"criterion": "Forward-Looking and Inspirational Thinking", "weight": 0.25},
        ],
        "instruction_following": [
            {"criterion": "Response to Task Objectives", "weight": 0.34},
            {"criterion": "Adherence to Scope Limitations", "weight": 0.33},
            {"criterion": "Complete Coverage of Task Requirements", "weight": 0.33},
        ],
        "readability": [
            {"criterion": "Clear Structure and Logic", "weight": 0.25},
            {"criterion": "Language Expression and Fluency", "weight": 0.25},
            {"criterion": "Appropriate Use of Technical Terms", "weight": 0.25},
            {"criterion": "Information Presentation and Visual Elements", "weight": 0.25},
        ],
    },
}

# ══════════════════════════════════════════════════════════════
# DRB comparative evaluation — aligned with evaluate.py
# (merged score prompts + per-query criteria + reference articles)
# ══════════════════════════════════════════════════════════════

# English merged score prompt (comparative, from DRB score_prompt_en.py)
MERGED_SCORE_PROMPT_EN = """\
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2.  **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3.  **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    ...
}}
</output_format>

Now, please evaluate the two articles based on the research task and criteria, providing detailed comparative analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
"""

# Chinese merged score prompt (comparative, from DRB score_prompt_zh.py)
MERGED_SCORE_PROMPT_ZH = """\
<system_role>你是一名严格、细致、客观的调研文章评估专家。你擅长根据具体的评估标准，深入比较两篇针对同一任务的文章，并给出精确的评分和清晰的理由。</system_role>

<user_prompt>
**任务背景**
有一个深度调研任务，你需要评估针对该任务撰写的两篇调研文章。我们会从以下四个维度评估文章：全面性、洞察力、指令遵循能力和可读性。内容如下：
<task>
"{task_prompt}"
</task>

**待评估文章**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**评估标准**
现在，你需要根据以下**评判标准列表**，逐条评估并比较这两篇文章的表现，输出对比分析，然后给出0-10的分数。每个标准都附有其解释，请仔细理解。

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**你的任务**
请严格按照 `<criteria_list>` 中的**每一条标准**，对比评估 `<article_1>` 和 `<article_2>` 在该标准上的具体表现。你需要：
1.  **逐条分析**：针对列表中的每一条标准，分别思考两篇文章是如何满足该标准要求的。
2.  **对比评估**：结合文章内容与标准解释，对比分析两篇文章在每一条标准上的表现。
3.  **分别打分**：基于你的对比分析，为两篇文章在该条标准上的表现分别打分（0-10分）。

**打分规则**
对每一条标准，分别为两篇文章打分，打分范围为 0-10 分（连续的数值）。分数高低应体现文章在该标准上表现的好坏：
*   0-2分：表现很差。几乎完全不符合标准要求。
*   2-4分：表现较差。少量符合标准要求，但有明显不足。
*   4-6分：表现中等。基本符合标准要求，不好不坏。
*   6-8分：表现较好。大部分符合标准要求，有可取之处。
*   8-10分：表现出色/极好。完全或超预期符合标准要求。

**输出格式要求**
请**严格**按照下列`<output_format>`格式输出每一条标准的评估结果，**不要包含任何其他无关内容、引言或总结**。从"标准1"开始，按顺序输出所有标准的评估：
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [全面性维度的第一条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [洞察力维度的第一条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        ...
    ],
    ...
}}
</output_format>

现在，请根据调研任务和标准，对两篇文章进行评估，并按照上述要求给出详细的对比分析和评分，请确保输出格式遵守上述`<output_format>`，而且保证其中的json格式可以解析，注意所有可能导致json解析错误的要转义的符号。
</user_prompt>
"""

# DRB data directory (relative to this script → deep_research_bench/data)
_DRB_DATA_DIR = "/home/t2vg-a100-G4-13/xietian/xl/data/drb"

_drb_data_cache = None


def _load_drb_data() -> dict:
    """Lazily load DRB criteria, reference, and query data for comparative eval."""
    global _drb_data_cache
    if _drb_data_cache is not None:
        return _drb_data_cache

    criteria_path = os.path.join(_DRB_DATA_DIR, "criteria_data", "criteria.jsonl")
    reference_path = os.path.join(_DRB_DATA_DIR, "test_data", "cleaned_data", "reference.jsonl")
    query_path = os.path.join(_DRB_DATA_DIR, "prompt_data", "query.jsonl")

    criteria_map: dict[str, dict] = {}    # prompt_text -> criteria_data
    reference_map: dict[str, str] = {}   # prompt_text -> article text
    query_map: dict[str, dict] = {}      # prompt_text -> {id, language, ...}

    for path, target_map, value_fn in [
        (criteria_path, criteria_map, lambda row: row),
        (reference_path, reference_map, lambda row: row.get("article", "")),
        (query_path, query_map, lambda row: row),
    ]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    target_map[row["prompt"]] = value_fn(row)

    _drb_data_cache = {
        "criteria_map": criteria_map,
        "reference_map": reference_map,
        "query_map": query_map,
    }
    print(
        f"[DRB] Loaded: {len(criteria_map)} criteria, "
        f"{len(reference_map)} references, {len(query_map)} queries"
    )
    return _drb_data_cache


def _format_criteria_list(criteria_data: dict) -> str:
    """Format evaluation criteria list as JSON string (no weights). Same as evaluate.py."""
    criteria_for_prompt: dict[str, list] = {}
    for dim, criterions_list in criteria_data.get("criterions", {}).items():
        if not isinstance(criterions_list, list):
            continue
        criteria_for_prompt[dim] = []
        for crit_item in criterions_list:
            if isinstance(crit_item, dict) and "criterion" in crit_item and "explanation" in crit_item:
                criteria_for_prompt[dim].append({
                    "criterion": crit_item["criterion"],
                    "explanation": crit_item["explanation"],
                })
    return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)


def _compute_comparative_score(llm_output_json: dict, criteria_data: dict) -> float:
    """Compute comparative score: target_total / (target_total + reference_total).

    Mirrors evaluate.py's calculate_weighted_scores + normalization logic.
    Returns a score in [0, 1] where 0.5 means on-par with reference.
    """
    dimension_weights = criteria_data.get("dimension_weight", {})
    criterion_weights: dict[str, dict[str, float]] = {}
    for dim, criterions in criteria_data.get("criterions", {}).items():
        criterion_weights[dim] = {c["criterion"]: c["weight"] for c in criterions}

    total_target = 0.0
    total_reference = 0.0

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            continue
        if dim not in dimension_weights or dim not in criterion_weights:
            continue

        dim_map = criterion_weights[dim]
        dim_target_sum = 0.0
        dim_ref_sum = 0.0
        dim_weight_sum = 0.0

        for score_item in scores_list:
            if not isinstance(score_item, dict):
                continue
            criterion_text = (score_item.get("criterion") or "").strip()
            art1 = score_item.get("article_1_score")
            art2 = score_item.get("article_2_score")
            if art1 is None or art2 is None or not criterion_text:
                continue
            try:
                s1, s2 = float(art1), float(art2)
            except (ValueError, TypeError):
                continue

            # Weight lookup: exact → case-insensitive → substring → average
            weight = dim_map.get(criterion_text)
            if weight is None:
                for key, val in dim_map.items():
                    if key.lower() == criterion_text.lower():
                        weight = val
                        break
            if weight is None:
                for key, val in dim_map.items():
                    if criterion_text.lower() in key.lower() or key.lower() in criterion_text.lower():
                        weight = val
                        break
            if weight is None:
                weight = sum(dim_map.values()) / max(len(dim_map), 1)

            dim_target_sum += s1 * weight
            dim_ref_sum += s2 * weight
            dim_weight_sum += weight

        if dim_weight_sum > 0:
            dim_w = dimension_weights.get(dim, 0)
            total_target += (dim_target_sum / dim_weight_sum) * dim_w
            total_reference += (dim_ref_sum / dim_weight_sum) * dim_w

    denom = total_target + total_reference
    return total_target / denom if denom > 0 else 0.0


def _compute_comparative_score_with_dims(llm_output_json: dict, criteria_data: dict) -> tuple[float, dict[str, float]]:
    """Compute comparative score + per-dimension target scores.

    Returns (score_in_0_1, {dim: target_avg_score}).
    """
    dimension_weights = criteria_data.get("dimension_weight", {})
    criterion_weights: dict[str, dict[str, float]] = {}
    for dim, criterions in criteria_data.get("criterions", {}).items():
        criterion_weights[dim] = {c["criterion"]: c["weight"] for c in criterions}

    total_target = 0.0
    total_reference = 0.0
    dim_scores: dict[str, float] = {}

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            continue
        if dim not in dimension_weights or dim not in criterion_weights:
            continue

        dim_map = criterion_weights[dim]
        dim_target_sum = 0.0
        dim_ref_sum = 0.0
        dim_weight_sum = 0.0

        for score_item in scores_list:
            if not isinstance(score_item, dict):
                continue
            criterion_text = (score_item.get("criterion") or "").strip()
            art1 = score_item.get("article_1_score")
            art2 = score_item.get("article_2_score")
            if art1 is None or art2 is None or not criterion_text:
                continue
            try:
                s1, s2 = float(art1), float(art2)
            except (ValueError, TypeError):
                continue

            weight = dim_map.get(criterion_text)
            if weight is None:
                for key, val in dim_map.items():
                    if key.lower() == criterion_text.lower():
                        weight = val
                        break
            if weight is None:
                for key, val in dim_map.items():
                    if criterion_text.lower() in key.lower() or key.lower() in criterion_text.lower():
                        weight = val
                        break
            if weight is None:
                weight = sum(dim_map.values()) / max(len(dim_map), 1)

            dim_target_sum += s1 * weight
            dim_ref_sum += s2 * weight
            dim_weight_sum += weight

        if dim_weight_sum > 0:
            dim_w = dimension_weights.get(dim, 0)
            target_avg = dim_target_sum / dim_weight_sum
            dim_scores[dim] = target_avg
            total_target += target_avg * dim_w
            total_reference += (dim_ref_sum / dim_weight_sum) * dim_w

    denom = total_target + total_reference
    score = total_target / denom if denom > 0 else 0.0
    return score, dim_scores


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════
CONFIGS = {
    # ============== Agent Loop ==============
    "max_turns": 80,
    "max_context_tokens": 65536,
    # ============== Tools ==============
    "tool_concurrency": 256,
    "tools_base_url": os.environ.get(
        "TOOLS_BASE_URL",
        "http://t2vginfra.westus2.cloudapp.azure.com/search_tool",
    ),
    "browse_max_tokens": 2048,
    # ============== Jina Reader API (browse backend) ==============
    "jina_api_key": "jina_6ab9ee4df2034c18be0ce28b69581336o3l3rpepAsTLRR5kfywF5uTz5kPa",
    # ============== Log Probability Collection ==============
    "return_logprob": True,
    # ============== Reward ==============
    "format_score": 0.1,
    # ============== Judge (RACE reward via DeepSeek) ==============
    "judge": {
        "base_url": os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.environ.get("JUDGE_API_KEY", ""),
        "model": os.environ.get("JUDGE_MODEL", "deepseek-v4-flash"),
        "timeout": 120,
        "max_retries": 3,
    },
}

TOOL_SEMAPHORE = asyncio.Semaphore(CONFIGS["tool_concurrency"])
JUDGE_SEMAPHORE = asyncio.Semaphore(64)

_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")


def _smart_truncate(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens (via tiktoken) at a natural boundary."""
    tokens = _TIKTOKEN_ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = _TIKTOKEN_ENC.decode(tokens[:max_tokens])
    return truncated + "\n...[truncated]"


async def _jina_browse(url: str, max_tokens: int = 2048) -> str:
    """Fetch page content via jina.ai reader API."""
    import httpx

    jina_key = CONFIGS.get("jina_api_key", "")
    if not jina_key:
        return ""

    # Normalize URL
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    jina_url = f"https://r.jina.ai/{url}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                jina_url,
                headers={"Authorization": f"Bearer {jina_key}"},
            )
            resp.raise_for_status()
            text = resp.text
    except Exception:
        return ""

    return _smart_truncate(text, max_tokens)


# ══════════════════════════════════════════════════════════════
# Tool dispatch (search via TOOLS_BASE_URL, browse via jina.ai)
# ══════════════════════════════════════════════════════════════
async def _dispatch_tool(name: str, args: dict) -> str:
    """Execute a tool call and return text result (same backend as evaluate.py)."""
    from custom_search_server import _raw_search

    base_url = CONFIGS["tools_base_url"]
    if name == "search":
        items = await _raw_search(
            base_url,
            args.get("query", ""),
            max_num_results=args.get("max_num_results", 5),
        )
        return json.dumps(items, ensure_ascii=False) if items else "No search results found."
    elif name == "browse":
        content = await _jina_browse(
            args.get("url", ""),
            max_tokens=args.get("max_tokens", CONFIGS["browse_max_tokens"]),
        )
        if content:
            return json.dumps({"url": args.get("url", ""), "content": content}, ensure_ascii=False)
        return f"Failed to browse {args.get('url', '')}. Webpage does not exist."
    return json.dumps({"error": f"Unknown tool: {name}"})


def _parse_tool_call(text: str):
    """Parse <tool_call>...</tool_call> from model output.

    Returns (tool_name, tool_args_dict) or (None, None) if not found / invalid.
    """
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None, None
    try:
        call = json.loads(m.group(1))
        return call.get("name"), call.get("arguments", {})
    except (json.JSONDecodeError, AttributeError):
        return None, None


def _truncate_at_tool_call(text: str) -> str:
    """Return text up to and including </tool_call>."""
    idx = text.find("</tool_call>")
    if idx >= 0:
        return text[: idx + len("</tool_call>")]
    return text


# ══════════════════════════════════════════════════════════════
# Generate — agent loop with Qwen3 tool-calling format
# ══════════════════════════════════════════════════════════════
async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Build prompt with GENERATION_SYSTEM_PROMPT + tools via chat template.
    # NOTE: --apply-chat-template must be REMOVED from shell args so that
    # sample.prompt contains the raw query text.
    raw_query = sample.prompt
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": raw_query},
    ]
    prompt_text = state.tokenizer.apply_chat_template(
        messages, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if CONFIGS["return_logprob"] else None

    last_output = None
    has_final_article = False
    final_response_text = ""

    for _turn_idx in range(CONFIGS["max_turns"]):
        # ---- context length guard ----
        total_tokens = len(prompt_tokens_ids) + len(response_token_ids)
        if total_tokens >= CONFIGS["max_context_tokens"] - 1024:
            break

        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
        }
        if CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)
        last_output = output

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]

        # ---- Detect tool call ----
        has_tool_call = "</tool_call>" in cur_response

        if CONFIGS["return_logprob"]:
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )
            cur_response_token_ids = [
                item[1] for item in output["meta_info"]["output_token_logprobs"]
            ]
            cur_response_log_probs = [
                item[0] for item in output["meta_info"]["output_token_logprobs"]
            ]
            # Truncate at </tool_call> if needed (remove trailing tokens after it)
            if has_tool_call:
                truncated = _truncate_at_tool_call(cur_response)
                if truncated != cur_response:
                    truncated_ids = state.tokenizer(
                        truncated, add_special_tokens=False
                    )["input_ids"]
                    n = len(truncated_ids)
                    cur_response_token_ids = cur_response_token_ids[:n]
                    cur_response_log_probs = cur_response_log_probs[:n]
                    cur_response = truncated
        else:
            if has_tool_call:
                cur_response = _truncate_at_tool_call(cur_response)
            cur_response_token_ids = state.tokenizer(
                cur_response, add_special_tokens=False
            )["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)

        if CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        # ---- If no tool call, model is done (final article) ----
        if not has_tool_call:
            has_final_article = True
            final_response_text = cur_response
            break

        if output["meta_info"]["finish_reason"]["type"] == "length" and not has_tool_call:
            has_final_article = True
            final_response_text = cur_response
            break

        # ---- Parse and execute tool call ----
        tool_name, tool_args = _parse_tool_call(cur_response)
        if tool_name:
            try:
                async with TOOL_SEMAPHORE:
                    result = await _dispatch_tool(tool_name, tool_args)
            except Exception:
                result = "Tool call failed. Please try a different approach."
        else:
            result = "Invalid tool call format."

        # ---- Inject tool response in Qwen3 chat format ----
        # Format: <|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n
        obs = (
            f"<|im_end|>\n<|im_start|>user\n<tool_response>\n"
            f"{result}\n"
            f"</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )
        obs_tokens_ids = state.tokenizer(obs, add_special_tokens=False)["input_ids"]

        response += obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

        if CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp mismatch: {len(response_token_ids)} vs {len(rollout_log_probs)}"

    # ---- Store results on sample ----
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    if has_final_article:
        sample.metadata["final_article"] = _clean_article(final_response_text)
    else:
        sample.metadata["final_article"] = ""

    if CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    if last_output is None:
        sample.status = Sample.Status.TRUNCATED
    else:
        match last_output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    return sample


# ══════════════════════════════════════════════════════════════
# RACE reward via DeepSeek V4 Flash
# ══════════════════════════════════════════════════════════════
def _extract_user_query(prompt_text: str) -> str:
    """Extract raw user query from Qwen-style chat-templated prompt.

    The prompt may have tools description in the system message, so we
    extract the *user* turn content.
    """
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt_text, re.DOTALL)
    return m.group(1).strip() if m else prompt_text

def _get_final_article(sample: Sample) -> str:
    article = ""
    if hasattr(sample, "metadata"):
        article = sample.metadata.get("final_article", "") or ""
    if article.strip():
        return article.strip()
    return _extract_article(sample.response)

def _clean_article(text: str) -> str:
    """Clean final assistant output: remove think blocks, tool calls, chat template tokens."""
    # Remove think blocks
    think_start = text.rfind("<think>")
    think_end = text.rfind("</think>")
    if think_start >= 0 and think_end >= 0 and think_end > think_start:
        text = text[think_end + len("</think>"):]
    elif think_start >= 0 and think_end < 0:
        text = text[:think_start]

    # Remove any stray tool calls (safety)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)

    # Remove chat template tokens
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return text


def _extract_article(response_text: str) -> str:
    """Extract the final article from the model's response.

    The model uses Qwen3 tool-calling format. The final article is in the
    last assistant turn, after the last </think> tag (if present).
    """
    # Find the last </think> and take everything after it
    idx = response_text.rfind("</think>")
    if idx >= 0:
        article = response_text[idx + len("</think>"):]
    else:
        # No thinking — use entire response (unlikely but safe)
        article = response_text
    # Clean up stray chat template tokens
    article = article.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return article


# Shared httpx client (created lazily, once per worker process)
_judge_client = None


def _get_judge_client():
    global _judge_client
    if _judge_client is None or _judge_client.is_closed:
        import httpx

        cfg = CONFIGS["judge"]
        _judge_client = httpx.AsyncClient(
            base_url=cfg["base_url"].rstrip("/"),
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            timeout=cfg["timeout"],
        )
    return _judge_client


def _extract_json_from_text(text: str) -> str | None:
    """Extract JSON from markdown code blocks or raw text (mirrors DRB extract_json_from_markdown)."""
    # Try ```json ... ```
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ``` ... ```
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try raw JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0).strip()
    return None


def _compute_weighted_score(llm_output_json: dict) -> float:
    """Compute weighted score from judge output, identical to evaluate.py's calculate_weighted_scores logic.

    Uses STATIC_CRITERIA_DATA dimension weights and per-criterion weights.
    Returns a score in [0, 10].
    """
    dimension_weights = STATIC_CRITERIA_DATA["dimension_weight"]
    criterion_weights = {}
    for dim, criterions in STATIC_CRITERIA_DATA["criterions"].items():
        criterion_weights[dim] = {c["criterion"]: c["weight"] for c in criterions}

    total_score = 0.0
    total_dim_weight = 0.0

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            continue
        if dim not in dimension_weights or dim not in criterion_weights:
            continue

        dim_criteria_map = criterion_weights[dim]
        dim_weighted_sum = 0.0
        dim_total_weight = 0.0

        for score_item in scores_list:
            if not isinstance(score_item, dict):
                continue
            criterion_text = (score_item.get("criterion") or "").strip()
            # target_score from point_wise prompt, or article_1_score as fallback
            raw_score = score_item.get("target_score") or score_item.get("article_1_score")
            if raw_score is None or not criterion_text:
                continue
            try:
                s = float(raw_score)
            except (ValueError, TypeError):
                continue

            # Exact match or fuzzy match for weight lookup
            weight = dim_criteria_map.get(criterion_text)
            if weight is None:
                # Try case-insensitive
                for key, val in dim_criteria_map.items():
                    if key.lower() == criterion_text.lower():
                        weight = val
                        break
            if weight is None:
                weight = 1.0 / max(len(dim_criteria_map), 1)

            dim_weighted_sum += s * weight
            dim_total_weight += weight

        if dim_total_weight > 0:
            dim_avg = dim_weighted_sum / dim_total_weight
            total_score += dim_avg * dimension_weights[dim]
            total_dim_weight += dimension_weights[dim]

    if total_dim_weight > 0:
        return total_score / total_dim_weight  # in [0, 10]
    return 0.0


def _compute_weighted_score_with_dims(llm_output_json: dict) -> tuple[float, dict[str, float]]:
    """Compute weighted score + per-dimension scores from judge output.

    Returns (total_score_in_0_10, {dim: avg_score}).
    """
    dimension_weights = STATIC_CRITERIA_DATA["dimension_weight"]
    criterion_weights = {}
    for dim, criterions in STATIC_CRITERIA_DATA["criterions"].items():
        criterion_weights[dim] = {c["criterion"]: c["weight"] for c in criterions}

    total_score = 0.0
    total_dim_weight = 0.0
    dim_scores: dict[str, float] = {}

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            continue
        if dim not in dimension_weights or dim not in criterion_weights:
            continue

        dim_criteria_map = criterion_weights[dim]
        dim_weighted_sum = 0.0
        dim_total_weight = 0.0

        for score_item in scores_list:
            if not isinstance(score_item, dict):
                continue
            criterion_text = (score_item.get("criterion") or "").strip()
            raw_score = score_item.get("target_score") or score_item.get("article_1_score")
            if raw_score is None or not criterion_text:
                continue
            try:
                s = float(raw_score)
            except (ValueError, TypeError):
                continue

            weight = dim_criteria_map.get(criterion_text)
            if weight is None:
                for key, val in dim_criteria_map.items():
                    if key.lower() == criterion_text.lower():
                        weight = val
                        break
            if weight is None:
                weight = 1.0 / max(len(dim_criteria_map), 1)

            dim_weighted_sum += s * weight
            dim_total_weight += weight

        if dim_total_weight > 0:
            dim_avg = dim_weighted_sum / dim_total_weight
            dim_scores[dim] = dim_avg
            total_score += dim_avg * dimension_weights[dim]
            total_dim_weight += dimension_weights[dim]

    final_score = total_score / total_dim_weight if total_dim_weight > 0 else 0.0
    return final_score, dim_scores


async def _call_judge(query: str, article: str) -> tuple[float, dict[str, float]]:
    """Call DeepSeek to score the article using DRB point_wise_score_prompt. Returns [0, 1]."""
    cfg = CONFIGS["judge"]
    if not cfg["api_key"]:
        raise ValueError("JUDGE_API_KEY environment variable not set")

    prompt = RACE_SCORE_PROMPT.format(task_prompt=query, article=article)
    client = _get_judge_client()

    for attempt in range(cfg["max_retries"]):
        try:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16384,
                    "temperature": 0.1,
                    "reasoning": {"effort": "none"},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            json_str = _extract_json_from_text(content)
            if not json_str:
                raise ValueError("Failed to extract JSON from judge response")
            llm_output = json.loads(json_str)

            # Validate expected dimensions
            expected_dims = ["comprehensiveness", "insight", "instruction_following", "readability"]
            missing = [d for d in expected_dims if d not in llm_output]
            if missing:
                raise ValueError(f"Missing dimensions: {missing}")

            # Compute weighted score (0-10) then normalize to [0, 1]
            score_0_10, dim_scores = _compute_weighted_score_with_dims(llm_output)
            return score_0_10 / 10.0, dim_scores

        except Exception as e:
            if attempt < cfg["max_retries"] - 1:
                await asyncio.sleep(2 ** attempt)

    # All retries failed — return a small default
    default_score = CONFIGS["format_score"] * 0.5
    return default_score, {}


async def _call_judge_comparative(
    query: str, article: str, reference: str,
    criteria_data: dict, language: str,
) -> float:
    """Call DeepSeek to score article vs reference using DRB merged_score_prompt.

    Uses per-query criteria and language-specific prompt.
    Returns [0, 1] where 0.5 means on-par with reference.
    """
    cfg = CONFIGS["judge"]
    if not cfg["api_key"]:
        raise ValueError("JUDGE_API_KEY environment variable not set")

    criteria_list_str = _format_criteria_list(criteria_data)
    merged_prompt = MERGED_SCORE_PROMPT_ZH if language == "zh" else MERGED_SCORE_PROMPT_EN
    prompt = merged_prompt.format(
        task_prompt=query,
        article_1=article,
        article_2=reference,
        criteria_list=criteria_list_str,
    )

    client = _get_judge_client()

    for attempt in range(cfg["max_retries"]):
        try:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16384,
                    "temperature": 0.1,
                    "reasoning": {"effort": "none"},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            json_str = _extract_json_from_text(content)
            if not json_str:
                raise ValueError("Failed to extract JSON from judge response")
            llm_output = json.loads(json_str)

            expected_dims = ["comprehensiveness", "insight", "instruction_following", "readability"]
            missing = [d for d in expected_dims if d not in llm_output]
            if missing:
                raise ValueError(f"Missing dimensions: {missing}")

            score, dim_scores = _compute_comparative_score_with_dims(llm_output, criteria_data)
            return score, dim_scores

        except Exception as e:
            if attempt < cfg["max_retries"] - 1:
                await asyncio.sleep(2 ** attempt)

    # All retries failed
    default_score = CONFIGS["format_score"] * 0.5
    return default_score, {}


async def reward_func(args, sample, **kwargs):
    """RACE reward: score the generated research article via DeepSeek judge.

    Scoring modes:
        - DRB eval samples (query in reference_map): comparative scoring
          against reference article with per-query criteria, aligned with
          evaluate.py. Score = target / (target + reference) ∈ [0, 1].
        - Training samples (no reference): point-wise scoring with static
          criteria. Score = weighted_score / 10 ∈ [0, 1].
        - Empty or very short article (< 50 chars) → format_score (0.1)
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # 1. Extract the article from the model's response
    article = _get_final_article(sample)
    if not article or len(article.strip()) < 50:
        return CONFIGS["format_score"]

    # 2. Extract the original query
    query = _extract_user_query(sample.prompt)

    # 3. Check if this is a DRB eval sample with reference data
    drb = _load_drb_data()
    reference = drb["reference_map"].get(query)
    criteria = drb["criteria_map"].get(query)
    query_info = drb["query_map"].get(query)

    do_print = random.randint(1, 32) == 1

    if reference and criteria:
        # ---- Comparative scoring (eval samples, aligned with evaluate.py) ----
        sample.metadata["race_mode"] = "comparative"
        language = query_info.get("language", "en") if query_info else "en"
        async with JUDGE_SEMAPHORE:
            score, dim_scores = await _call_judge_comparative(
                query, article, reference, criteria, language,
            )
        if do_print:
            print("--------------------------------")
            print(f"[COMPARATIVE] Query: {query[:120]}...")
            print(f"Article length: {len(article)} chars")
            print(f"Language: {language}")
            print(f"RACE score: {score:.4f} (0.5 = on-par with reference)")
            print(f"Dim scores: {dim_scores}")
    else:
        sample.metadata["race_mode"] = "pointwise"
        # ---- Point-wise scoring (training samples, no reference) ----
        async with JUDGE_SEMAPHORE:
            score, dim_scores = await _call_judge(query, article)
        if do_print:
            print("--------------------------------")
            print(f"[POINTWISE] Query: {query[:120]}...")
            print(f"Article length: {len(article)} chars")
            print(f"RACE score: {score:.4f}")
            print(f"Dim scores: {dim_scores}")

    sample.metadata["judge_scores"] = dim_scores
    return score


def _safe_tool_call_count(sample: Sample) -> int:
    resp = getattr(sample, "response", "") or ""
    return resp.count("<tool_call>")

def _append_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ══════════════════════════════════════════════════════════════
# Custom train rollout logging — upload sample trajectories to wandb
# ══════════════════════════════════════════════════════════════
_NUM_TRAIN_TRAJECTORIES = 3


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Custom train rollout log function.

    Logs a wandb.Table with up to _NUM_TRAIN_TRAJECTORIES sample trajectories
    so you can monitor actual generated text during training.
    Returns False so default logging also runs.
    """
    try:
        import wandb
    except ImportError:
        return False

    if not args.use_wandb or wandb.run is None:
        return False

    # Pick up to _NUM_TRAIN_TRAJECTORIES samples, spread across the batch
    n = min(_NUM_TRAIN_TRAJECTORIES, len(samples))
    if n == 0:
        return False

    step = max(1, len(samples) // n)
    indices = [i * step for i in range(n)]

    rows = []

    all_tool_calls = [_safe_tool_call_count(s) for s in samples]
    avg_tool_calls = sum(all_tool_calls) / max(len(all_tool_calls), 1)
    max_tool_calls = max(all_tool_calls) if all_tool_calls else 0

    wandb.log({
        "train/avg_tool_calls": avg_tool_calls,
        "train/max_tool_calls": max_tool_calls,
    }, step=rollout_id)

    for idx in indices:
        s = samples[idx]
        query = _extract_user_query(s.prompt)
        article = _get_final_article(s)
        tool_call_count = s.response.count("<tool_call>")
        reward = s.get_reward_value(args) if hasattr(s, "get_reward_value") else 0.0
        judge_scores = s.metadata.get("judge_scores", {}) if hasattr(s, "metadata") else {}

        metadata = s.metadata if hasattr(s, "metadata") else {}
        judge_scores = metadata.get("judge_scores", {})
        race_mode = metadata.get("race_mode", "")

        rows.append([
            rollout_id,
            idx,
            race_mode,
            query[:500],
            s.response[:10000],
            article[:5000],
            tool_call_count,
            len(article),
            s.response_length,
            float(reward) if reward is not None else 0.0,
            s.status.name if hasattr(s.status, "name") else str(s.status),
            float(judge_scores.get("comprehensiveness", 0.0)),
            float(judge_scores.get("insight", 0.0)),
            float(judge_scores.get("instruction_following", 0.0)),
            float(judge_scores.get("readability", 0.0)),
        ])

    table = wandb.Table(
        columns=[
            "rollout_id", "sample_idx", "race_mode", "query", "full_response",
            "final_article", "tool_calls", "article_len", "response_len", "reward", "status",
            "comprehensiveness", "insight", "instruction_following", "readability",
        ],
        data=rows,
    )
    wandb.log({"train/trajectories": table}, step=rollout_id)
    
    _append_jsonl(
        f"/home/t2vg-a100-G4-13/xietian/xl/drppl/trajectory_logs/train_rollout_{rollout_id}.jsonl",
        rows,
    )
    
    return False


# ══════════════════════════════════════════════════════════════
# Custom eval logging — upload 3 trajectories to wandb
# ══════════════════════════════════════════════════════════════
_NUM_EVAL_TRAJECTORIES = 3


def log_eval_rollout(rollout_id, args, data, extra_metrics) -> bool:
    """Custom eval rollout log function.

    Logs default metrics + a wandb.Table with up to 3 sample trajectories.
    Returns False so default logging also runs.
    """
    try:
        import wandb
    except ImportError:
        return False

    if not args.use_wandb or wandb.run is None:
        return False

    # Collect trajectory rows from all eval datasets
    rows = []
    for dataset_name, dataset_data in data.items():
        samples = dataset_data.get("samples", [])
        rewards = dataset_data.get("rewards", [])

        # Pick up to _NUM_EVAL_TRAJECTORIES samples, spread across the dataset
        n = min(_NUM_EVAL_TRAJECTORIES, len(samples))
        if n == 0:
            continue

        step = max(1, len(samples) // n)
        indices = [i * step for i in range(n)]

        tool_counts = [_safe_tool_call_count(s) for s in samples]
        avg_tool_calls = sum(tool_counts) / max(len(tool_counts), 1)
        max_tool_calls = max(tool_counts) if tool_counts else 0

        wandb.log({"eval/{dataset_name}/avg_tool_calls": avg_tool_calls}, step=rollout_id)
        wandb.log({"eval/{dataset_name}/max_tool_calls": max_tool_calls}, step=rollout_id)
        wandb.log({"eval/{dataset_name}/num_samples_for_traj": len(samples)}, step=rollout_id)


        for idx in indices:
            s = samples[idx]
            r = rewards[idx] if idx < len(rewards) else None

            query = _extract_user_query(s.prompt)
            article = _get_final_article(s)

            # Count tool calls in response
            tool_call_count = s.response.count("<tool_call>")

            metadata = s.metadata if hasattr(s, "metadata") else {}
            judge_scores = metadata.get("judge_scores", {})
            race_mode = metadata.get("race_mode", "")

            rows.append([
                dataset_name,
                idx,
                race_mode,
                query[:500],
                s.response[:10000],          # full trajectory (truncated for wandb)
                article[:5000],              # final article
                tool_call_count,
                len(article),
                float(r) if r is not None else 0.0,
                s.status.name if hasattr(s.status, "name") else str(s.status),
                float(judge_scores.get("comprehensiveness", 0.0)),
                float(judge_scores.get("insight", 0.0)),
                float(judge_scores.get("instruction_following", 0.0)),
                float(judge_scores.get("readability", 0.0)),
            ])

    if rows:
        table = wandb.Table(
            columns=[
                "dataset",
                "sample_idx",
                "race_mode",
                "query",
                "full_response",
                "final_article",
                "tool_calls",
                "article_len",
                "reward",
                "status",
                "comprehensiveness",
                "insight",
                "instruction_following",
                "readability",
            ],
            data=rows,
        )
        wandb.log({"eval/trajectories": table}, step=rollout_id)

    _append_jsonl(
        f"/home/t2vg-a100-G4-13/xietian/xl/drppl/trajectory_logs/eval_rollout_{rollout_id}.jsonl",
        rows,
    )

    return False  # let default logging also run