# ComfyUI-QwenVL-Utils / lib / output_cleaner.py
# Clean model output by removing thinking blocks, code fences, role prefixes, etc.

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OutputCleanConfig:
    mode: str = "prompt"
    strip_think: bool = True
    strip_code_fences: bool = True
    strip_role_prefixes: bool = True
    strip_json_wrappers: bool = True
    strip_leading_preamble: bool = True
    strip_planning: bool = True
    keep_first_paragraph_only: bool = False


_ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|final|output|response|result|prompt)\s*:\s*", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*$", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think[^>]*>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_MARKER_RE = re.compile(r"(?im)^\s*(final|final answer|answer|output|result|prompt)\s*[:\-]\s*")
_IM_TOKEN_RE = re.compile(r"(?i)<\|?im_(start|end)\|?>|<im_(start|end)>|<\|endoftext\|>")
_PLANNING_RE = re.compile(
    r"(?is)\b("
    r"i\s+(should|need|must|will|want|am\s+going\s+to|have\s+to)\b|"
    r"let's\b|first\b|next\b|then\b|wait\b|"
    r"so\s+i\s+need\s+to\b|i\s+should\s+focus\s+on\b"
    r")"
)


def clean_model_output(text: str, config: Optional[OutputCleanConfig] = None) -> str:
    if not text:
        return ""
    cfg = config or OutputCleanConfig()
    cleaned = text.strip()

    cleaned = _IM_TOKEN_RE.sub("", cleaned).strip()

    if cfg.strip_think:
        cleaned = _THINK_BLOCK_RE.sub("", cleaned)
        cleaned = _THINK_CLOSE_RE.sub("", cleaned)
        if _THINK_OPEN_RE.search(cleaned):
            cleaned = _THINK_OPEN_RE.sub("", cleaned)
            parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
            if len(parts) == 2:
                cleaned = parts[1]
        cleaned = cleaned.strip()

    cleaned = _IM_TOKEN_RE.sub("", cleaned).strip()

    if cfg.strip_code_fences and "```" in cleaned:
        cleaned = "\n".join(ln for ln in cleaned.splitlines() if not _CODE_FENCE_RE.match(ln)).strip()

    if cfg.strip_json_wrappers:
        extracted = _extract_from_json(cleaned, cfg.mode)
        if extracted is not None:
            cleaned = extracted.strip()

    if cfg.strip_leading_preamble:
        cleaned = _drop_preamble(cleaned).strip()

    if cfg.strip_planning and cfg.mode == "prompt":
        without = _strip_planning_paragraphs(cleaned)
        if without:
            cleaned = without

    if cfg.strip_role_prefixes:
        lines = cleaned.splitlines()
        if lines:
            lines[0] = _ROLE_PREFIX_RE.sub("", lines[0])
        cleaned = "\n".join(lines).strip()

    cleaned = _MARKER_RE.sub("", cleaned).strip()

    if cfg.keep_first_paragraph_only:
        cleaned = re.split(r"\n\s*\n", cleaned, maxsplit=1)[0].strip()

    return cleaned


def _extract_from_json(text: str, mode: str) -> Optional[str]:
    candidate = text.strip()
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    keys = (["prompt", "final", "output", "text", "content"] if mode == "prompt"
            else ["text", "content", "output", "final"])
    for k in keys:
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _drop_preamble(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    keep_from = 0
    for i, ln in enumerate(lines[:20]):
        if _MARKER_RE.match(ln):
            keep_from = i
        if re.search(r"(?i)\bhere(?:'s| is)\b", ln) and i < 6:
            keep_from = max(keep_from, i + 1)
    return "\n".join(lines[keep_from:]).strip()


def _strip_planning_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paragraphs:
        return ""
    kept = []
    dropping = True
    for p in paragraphs:
        if dropping and _PLANNING_RE.search(p):
            continue
        dropping = False
        kept.append(p)
    return "\n\n".join(kept).strip() if kept else text.strip()
