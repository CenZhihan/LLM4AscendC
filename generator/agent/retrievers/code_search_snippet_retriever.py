"""
Restricted local snippet search for curated Ascend C sources.

Unlike ``code_rag``, this retriever does not search the general indexed code corpus.
It first mirrors the allowed sources into ``generator/agent/Knowledge/code_search_snippet``
and then searches only that local Knowledge tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from generator.config import (
    agent_code_search_snippet_knowledge_root,
    agent_code_search_snippet_max_chars,
    agent_code_search_snippet_source_asc_devkit_examples,
    agent_code_search_snippet_source_cann_skills_root,
    agent_code_search_snippet_top_k,
)


_MIN_SNIPPET_CHARS = 80
_MAX_SNIPPET_CHARS = 5000
_CODE_EXTENSIONS = {
    ".asc": "cpp",
    ".c": "c",
    ".cc": "cpp",
    ".cmake": "cmake",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".json": "json",
    ".py": "python",
    ".sh": "bash",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
}
_TEXTUAL_SCRIPT_EXTENSIONS = frozenset({".py", ".sh"})
_DIRECT_COPY_EXTENSIONS = frozenset({
    ".asc", ".c", ".cc", ".cmake", ".cpp", ".cxx", ".h", ".hpp",
    ".json", ".py", ".sh", ".txt", ".yaml", ".yml",
})
_FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)\n(?P<code>[\s\S]*?)```", re.MULTILINE)
_DEF_RE = re.compile(
    r"(^\s*(?:template\s*<[^>]+>\s*)?(?:class|struct)\s+\w+[^\n{]*\{)"
    r"|(^\s*extern\s+\"C\"\s+.*?\b\w+\s*\([^;]*\)\s*\{)"
    r"|(^\s*(?:__aicore__\s+)?(?:inline\s+)?[\w:<>~*&\s]+\b\w+\s*\([^;]*\)\s*(?:const\s*)?\{)",
    re.MULTILINE,
)
_CANN_SKILLS_EXCLUDED = frozenset({
    "ascendc-api-best-practices",
    "ascendc-code-review",
    "ascendc-docs-search",
    "ascendc-env-check",
    "ascendc-npu-arch",
    "ascendc-task-focus",
    "ascendc-tiling-design",
})
_LANGUAGE_TO_EXTENSION = {
    "asc": ".asc",
    "bash": ".sh",
    "c": ".c",
    "cmake": ".cmake",
    "cpp": ".cpp",
    "c++": ".cpp",
    "cc": ".cpp",
    "cxx": ".cpp",
    "h": ".h",
    "hpp": ".hpp",
    "json": ".json",
    "python": ".py",
    "py": ".py",
    "shell": ".sh",
    "sh": ".sh",
    "yaml": ".yaml",
    "yml": ".yaml",
}


@dataclass(frozen=True)
class _SnippetRecord:
    source: str
    file_path: str
    relative_path: str
    start_line: int
    end_line: int
    language: str
    text: str


@dataclass(frozen=True)
class CodeSearchSnippetResult:
    source: str
    file_path: str
    relative_path: str
    start_line: int
    end_line: int
    language: str
    text: str
    score: float


def _normalize_source_root(path_text: str) -> Optional[Path]:
    if not path_text:
        return None
    root = Path(path_text).expanduser()
    return root.resolve() if root.exists() else None


def _normalize_target_root(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def _slugify_path_stem(path: Path) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", path.stem)


def _guess_snippet_extension(language: str, code: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in _LANGUAGE_TO_EXTENSION:
        return _LANGUAGE_TO_EXTENSION[normalized]

    stripped = code.lstrip()
    if "#include" in code or "__aicore__" in code or "GM_ADDR" in code:
        return ".cpp"
    if stripped.startswith("import ") or stripped.startswith("from "):
        return ".py"
    if stripped.startswith("{") or stripped.startswith("["):
        return ".json"
    if stripped.startswith("#!/bin/bash") or stripped.startswith("#!/usr/bin/env bash"):
        return ".sh"
    if "cmake_minimum_required" in code or "project(" in code:
        return ".cmake"
    return ".txt"


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", " ", (text or "").lower())
    return [tok for tok in cleaned.split() if len(tok) >= 2]


def _extract_code_chunks(text: str) -> List[Tuple[int, int, str]]:
    matches = list(_DEF_RE.finditer(text))
    if not matches:
        return []

    chunks: List[Tuple[int, int, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        snippet = text[start:end].strip()
        if _MIN_SNIPPET_CHARS <= len(snippet) <= _MAX_SNIPPET_CHARS:
            start_line = text[:start].count("\n") + 1
            end_line = text[:end].count("\n") + 1
            chunks.append((start_line, end_line, snippet))
    return chunks


def _extract_fixed_line_chunks(text: str, lines_per_chunk: int = 40) -> List[Tuple[int, int, str]]:
    lines = text.splitlines()
    chunks: List[Tuple[int, int, str]] = []
    for start in range(0, len(lines), lines_per_chunk):
        end = min(start + lines_per_chunk, len(lines))
        snippet = "\n".join(lines[start:end]).strip()
        if _MIN_SNIPPET_CHARS <= len(snippet) <= _MAX_SNIPPET_CHARS:
            chunks.append((start + 1, end, snippet))
    return chunks


def _iter_cann_skills_files(root: Path) -> Iterable[Tuple[str, Path]]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        suffix = path.suffix.lower()
        if rel.name == "SKILL.md":
            yield "cann_skills", path
            continue
        if "references" in rel.parts or "templates" in rel.parts:
            if suffix == ".md":
                yield "cann_skills", path
            continue
        if "scripts" in rel.parts and suffix in _TEXTUAL_SCRIPT_EXTENSIONS:
            yield "cann_skills", path


def _iter_asc_devkit_files(root: Path) -> Iterable[Tuple[str, Path]]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _CODE_EXTENSIONS:
            yield "asc_devkit", path


def _iter_curated_knowledge_files(root: Path) -> Iterable[Tuple[str, Path]]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _CODE_EXTENSIONS:
            yield "cann_skills", path


def _score_record(record: _SnippetRecord, query: str, tokens: Sequence[str]) -> float:
    haystack = f"{record.relative_path}\n{record.text}".lower()
    query_lower = query.lower().strip()
    if not haystack:
        return 0.0

    score = 0.0
    if query_lower and query_lower in haystack:
        score += 8.0

    unique_tokens = list(dict.fromkeys(tokens))
    matched_tokens = 0
    for token in unique_tokens:
        if token in haystack:
            matched_tokens += 1
            score += 1.5
            if token in record.relative_path.lower():
                score += 1.0

    if unique_tokens and matched_tokens == len(unique_tokens):
        score += 3.0

    if record.source == "asc_devkit":
        score += 0.5
    if "tiling" in record.relative_path.lower() and "tiling" in unique_tokens:
        score += 1.0
    if "host" in record.relative_path.lower() and "host" in unique_tokens:
        score += 0.5
    if "kernel" in record.relative_path.lower() and "kernel" in unique_tokens:
        score += 0.5
    return score


class CodeSearchSnippetRetriever:
    """Search curated local snippets mirrored into the agent Knowledge tree."""

    def __init__(
        self,
        cann_skills_root: Optional[str] = None,
        asc_devkit_examples_root: Optional[str] = None,
        knowledge_root: Optional[str] = None,
        top_k: Optional[int] = None,
        max_chars: Optional[int] = None,
    ):
        self.source_cann_skills_root = _normalize_source_root(
            cann_skills_root or agent_code_search_snippet_source_cann_skills_root
        )
        self.source_asc_devkit_examples_root = _normalize_source_root(
            asc_devkit_examples_root or agent_code_search_snippet_source_asc_devkit_examples
        )
        self.knowledge_root = _normalize_target_root(
            knowledge_root or agent_code_search_snippet_knowledge_root
        )
        self.knowledge_cann_skills_root = self.knowledge_root / "cann_skills"
        self.knowledge_asc_devkit_root = self.knowledge_root / "asc_devkit_examples"
        self.top_k = top_k or agent_code_search_snippet_top_k
        self.max_chars = max_chars or agent_code_search_snippet_max_chars
        self._records: Optional[List[_SnippetRecord]] = None

    def _copy_if_needed(self, source_path: Path, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            try:
                src_stat = source_path.stat()
                dst_stat = target_path.stat()
                if (
                    src_stat.st_size == dst_stat.st_size
                    and src_stat.st_mtime_ns == dst_stat.st_mtime_ns
                ):
                    return
            except OSError:
                pass
        shutil.copy2(source_path, target_path)

    def _sync_source_tree(
        self,
        source_root: Optional[Path],
        target_root: Path,
        iterator,
    ) -> Optional[Path]:
        if source_root is None:
            return target_root if target_root.exists() else None

        copied_any = False
        for _, source_path in iterator(source_root):
            relative_path = source_path.relative_to(source_root)
            self._copy_if_needed(source_path, target_root / relative_path)
            copied_any = True

        if copied_any:
            return target_root
        return target_root if target_root.exists() else None

    def _materialize_markdown_snippets(
        self,
        source_path: Path,
        target_dir: Path,
    ) -> int:
        try:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return 0

        snippet_count = 0
        for index, match in enumerate(_FENCE_RE.finditer(text), start=1):
            snippet = match.group("code").strip()
            if len(snippet) < _MIN_SNIPPET_CHARS:
                continue
            extension = _guess_snippet_extension(match.group("lang") or "", snippet)
            target_name = f"{_slugify_path_stem(source_path)}__{index:03d}{extension}"
            target_path = target_dir / target_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(snippet + "\n", encoding="utf-8")
            snippet_count += 1
        return snippet_count

    def _materialize_cann_skills_knowledge(self) -> Optional[Path]:
        if self.source_cann_skills_root is None:
            return self.knowledge_cann_skills_root if self.knowledge_cann_skills_root.exists() else None

        if self.knowledge_cann_skills_root.exists():
            shutil.rmtree(self.knowledge_cann_skills_root)
        self.knowledge_cann_skills_root.mkdir(parents=True, exist_ok=True)

        copied_any = False
        for skill_dir in sorted(self.source_cann_skills_root.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name in _CANN_SKILLS_EXCLUDED:
                continue

            skill_name = skill_dir.name
            skill_md = skill_dir / "SKILL.md"
            if skill_md.is_file():
                copied_any |= bool(
                    self._materialize_markdown_snippets(
                        skill_md,
                        self.knowledge_cann_skills_root / "snippets" / skill_name / "skill",
                    )
                )

            for group_name in ("references", "templates"):
                group_root = skill_dir / group_name
                if not group_root.is_dir():
                    continue
                for path in sorted(group_root.rglob("*")):
                    if not path.is_file():
                        continue
                    relative_path = path.relative_to(group_root)
                    suffix = path.suffix.lower()
                    if suffix == ".md":
                        copied_any |= bool(
                            self._materialize_markdown_snippets(
                                path,
                                self.knowledge_cann_skills_root / group_name / skill_name / relative_path.parent,
                            )
                        )
                    elif suffix in _DIRECT_COPY_EXTENSIONS:
                        self._copy_if_needed(
                            path,
                            self.knowledge_cann_skills_root / group_name / skill_name / relative_path,
                        )
                        copied_any = True

            scripts_root = skill_dir / "scripts"
            if scripts_root.is_dir():
                for path in sorted(scripts_root.rglob("*")):
                    if path.is_file() and path.suffix.lower() in _DIRECT_COPY_EXTENSIONS:
                        relative_path = path.relative_to(scripts_root)
                        self._copy_if_needed(
                            path,
                            self.knowledge_cann_skills_root / "scripts" / skill_name / relative_path,
                        )
                        copied_any = True

        return self.knowledge_cann_skills_root if copied_any else None

    def _prepare_knowledge_roots(self) -> Tuple[Optional[Path], Optional[Path]]:
        self.knowledge_root.mkdir(parents=True, exist_ok=True)
        cann_root = self._materialize_cann_skills_knowledge()
        asc_root = self._sync_source_tree(
            self.source_asc_devkit_examples_root,
            self.knowledge_asc_devkit_root,
            _iter_asc_devkit_files,
        )
        return cann_root, asc_root

    def _load_records(self) -> List[_SnippetRecord]:
        if self._records is not None:
            return self._records

        records: List[_SnippetRecord] = []
        knowledge_cann_skills_root, knowledge_asc_devkit_root = self._prepare_knowledge_roots()

        def add_markdown_records(source: str, root: Path, path: Path) -> None:
            rel = path.relative_to(root).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return
            for match in _FENCE_RE.finditer(text):
                snippet = match.group("code").strip()
                if not (_MIN_SNIPPET_CHARS <= len(snippet) <= _MAX_SNIPPET_CHARS):
                    continue
                start_line = text[:match.start("code")].count("\n") + 1
                end_line = start_line + snippet.count("\n")
                language = (match.group("lang") or "text").strip().lower() or "text"
                records.append(
                    _SnippetRecord(
                        source=source,
                        file_path=str(path),
                        relative_path=rel,
                        start_line=start_line,
                        end_line=end_line,
                        language=language,
                        text=snippet,
                    )
                )

        def add_code_records(source: str, root: Path, path: Path) -> None:
            rel = path.relative_to(root).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return

            chunks = _extract_code_chunks(text) or _extract_fixed_line_chunks(text)
            language = _CODE_EXTENSIONS.get(path.suffix.lower(), "text")
            for start_line, end_line, snippet in chunks:
                records.append(
                    _SnippetRecord(
                        source=source,
                        file_path=str(path),
                        relative_path=rel,
                        start_line=start_line,
                        end_line=end_line,
                        language=language,
                        text=snippet,
                    )
                )

        if knowledge_cann_skills_root is not None:
            for source, path in _iter_curated_knowledge_files(knowledge_cann_skills_root):
                add_code_records(source, knowledge_cann_skills_root, path)

        if knowledge_asc_devkit_root is not None:
            for source, path in _iter_asc_devkit_files(knowledge_asc_devkit_root):
                add_code_records(source, knowledge_asc_devkit_root, path)

        self._records = records
        return records

    def is_available(self) -> bool:
        return bool(self._load_records())

    def available_sources(self) -> Dict[str, str]:
        return {
            "knowledge_root": str(self.knowledge_root),
            "cann_skills": str(self.source_cann_skills_root) if self.source_cann_skills_root else "",
            "asc_devkit": str(self.source_asc_devkit_examples_root) if self.source_asc_devkit_examples_root else "",
        }

    def build_query(self, op_name: str, category: str, extra_context: Optional[str] = None) -> str:
        parts: List[str] = []
        if extra_context:
            parts.append(extra_context)
        if op_name:
            parts.append(op_name.replace("_", " "))
        if category:
            parts.append(category)
        parts.append("Ascend C")
        return " | ".join(part for part in parts if part).strip()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        source: str = "all",
    ) -> List[CodeSearchSnippetResult]:
        records = self._load_records()
        if not records:
            return []

        requested_source = (source or "all").strip().lower()
        tokens = _tokenize(query)
        results: List[CodeSearchSnippetResult] = []
        for record in records:
            if requested_source not in {"", "all"} and record.source != requested_source:
                continue
            score = _score_record(record, query, tokens)
            if score <= 0:
                continue
            results.append(
                CodeSearchSnippetResult(
                    source=record.source,
                    file_path=record.file_path,
                    relative_path=record.relative_path,
                    start_line=record.start_line,
                    end_line=record.end_line,
                    language=record.language,
                    text=record.text,
                    score=score,
                )
            )

        k = top_k or self.top_k
        results.sort(key=lambda item: (-item.score, item.source, item.relative_path, item.start_line))
        return results[:k]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source: str = "all",
    ) -> List[str]:
        matches = self.search(query=query, top_k=top_k, source=source)
        if not matches:
            sources = self.available_sources()
            return [
                "[code_search_snippet] No matching snippets found. "
                f"Available roots: cann_skills={sources['cann_skills'] or 'missing'}, "
                f"asc_devkit={sources['asc_devkit'] or 'missing'}"
            ]

        formatted: List[str] = []
        total_chars = 0
        for index, match in enumerate(matches, start=1):
            header = (
                f"### Snippet {index} (score: {match.score:.2f}, source: {match.source}, "
                f"file: {match.relative_path}:{match.start_line}-{match.end_line})\n\n"
            )
            body = f"```{match.language}\n{match.text}\n```\n\n"
            part = header + body
            if total_chars + len(part) > self.max_chars:
                remaining = self.max_chars - total_chars
                if remaining > 120:
                    formatted.append((part[:remaining]).rstrip() + "\n\n(... truncated ...)")
                break
            formatted.append(part)
            total_chars += len(part)
        return formatted