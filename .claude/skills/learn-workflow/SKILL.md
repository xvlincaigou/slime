---
name: learn-workflow
description: Record Q&A sessions during codebase learning. Automatically saves questions and answers to the learn/ folder with proper organization. Use when the user is asking questions about understanding the codebase.
---

# Learn Workflow

Record and organize learning Q&A sessions in the learn/ folder.

## When to Use

Use this skill when:
- User asks questions about understanding the codebase
- User wants to record learnings for future reference
- After answering a question, you need to save the Q&A to the learn folder

## Folder Structure

```
learn/
├── README.md                 # Learning progress index
├── questions/                # Individual Q&A files
│   ├── Q001-question-title.md
│   ├── Q002-another-topic.md
│   └── ...
├── topics/                   # Topic-based summaries
│   ├── architecture.md
│   ├── rollout.md
│   ├── reward.md
│   └── ...
└── summaries/                # Weekly/monthly summaries
    └── 2026-04-summary.md
```

## Recording Format

Each Q&A file should follow this format:

```markdown
---
date: 2026-04-04
question_id: Q001
topics: ["rollout", "sglang"]
related_files:
  - slime/rollout/sglang_rollout.py
  - slime/rollout/base_types.py
---

# Question

用户的问题内容...

# Answer

回答内容...

## Key Points

- 要点1
- 要点2
- 要点3

## Code References

```python
# 相关代码片段
```

## Follow-up Questions

- [ ] 待深入的问题1
- [ ] 待深入的问题2
```

## Workflow

### Step 1: Answer the Question

First, answer the user's question thoroughly. Include:
- Direct answer
- Code references with file paths
- Key concepts explained

### Step 2: Create Q&A Record

1. Determine the next question ID by checking existing files in `learn/questions/`
2. Create a new file: `learn/questions/{ID}-{short-title}.md`
3. Fill in the frontmatter with date, topics, and related files
4. Write the question and answer content

### Step 3: Update Topic Files (if needed)

If the question relates to specific topics, update or create topic files in `learn/topics/`:

```markdown
# Topic: Rollout

## Overview

Brief description of the topic.

## Related Questions

- [Q001: What is rollout?](questions/Q001-what-is-rollout.md)
- [Q005: How to customize rollout?](questions/Q005-custom-rollout.md)

## Key Concepts

### Concept 1

Explanation...

## Code References

- `slime/rollout/sglang_rollout.py` - Main rollout implementation
```

### Step 4: Update README

Keep `learn/README.md` updated with:
- Current progress
- List of topics covered
- Outstanding questions

## Best Practices

- Use Chinese for content (as user prefers)
- Include file paths and line numbers for code references
- Tag related topics for cross-referencing
- Mark follow-up questions for future exploration
- Update topic summaries when new insights emerge
