"""
KB Shell Search Retriever for Ascend C kernel development agent.

Searches Knowledge/ directory documents using shell tools (grep/find)
for structured knowledge base queries.

Input: knowledge_category + operator_name
Output: Matching file paths + content snippets
"""
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Structured result types
# ============================================================

@dataclass
class KBShellSearchResult:
    """Result of KB shell search."""
    query: str
    category: str
    operator_name: str
    matches: List[dict]             # [{file, line, content}]
    total_matches: int
    files_searched: int
    details: str                    # Human-readable summary


# ============================================================
# Knowledge directory mapping
# ============================================================

# Maps category names to Knowledge/ subdirectories
_CATEGORY_DIRS: Dict[str, str] = {
    "api": "api",
    "api-docs": "api",
    "api_docs": "api",
    "tiling": "tiling",
    "tiling-design": "tiling",
    "tiling_design": "tiling",
    "reduction": "tiling/reduction",
    "index-tracking": "tiling/index-tracking",
    "arch": "arch",
    "architecture": "arch",
    "npu-arch": "arch",
    "npu_arch": "arch",
    "code-review": "code-review",
    "code_review": "code-review",
    "code-style": "code-review",
    "security": "code-review",
    "all": "",  # Search all subdirectories
}

# Knowledge base root
def _get_knowledge_root() -> Optional[str]:
    """Get the Knowledge/ directory root path."""
    this_dir = Path(__file__).parent
    knowledge = this_dir.parent / "Knowledge"
    if knowledge.is_dir():
        return str(knowledge)
    return None


# ============================================================
# Search implementation
# ============================================================

def search_kb(
    category: str = "all",
    operator_name: str = "",
    query: str = "",
) -> KBShellSearchResult:
    """
    Search Knowledge/ directory using grep/find.

    Args:
        category: Knowledge category (api/tiling/arch/code-review/all)
        operator_name: Operator name to search for
        query: Additional query text

    Returns:
        KBShellSearchResult with matches
    """
    knowledge_root = _get_knowledge_root()
    if not knowledge_root:
        return KBShellSearchResult(
            query=query,
            category=category,
            operator_name=operator_name,
            matches=[],
            total_matches=0,
            files_searched=0,
            details="知识库目录未找到。请确认 generator/agent/Knowledge/ 目录存在。",
        )

    # Resolve category directory
    cat_dir = _CATEGORY_DIRS.get(category.lower(), "")
    if cat_dir:
        search_dir = os.path.join(knowledge_root, cat_dir) if cat_dir else knowledge_root
    else:
        search_dir = knowledge_root

    if not os.path.isdir(search_dir):
        return KBShellSearchResult(
            query=query,
            category=category,
            operator_name=operator_name,
            matches=[],
            total_matches=0,
            files_searched=0,
            details=f"知识库分类目录 '{category}' 未找到: {search_dir}",
        )

    # Build search terms
    search_terms = []
    if operator_name:
        search_terms.append(operator_name)
    if query:
        search_terms.append(query)

    if not search_terms:
        return KBShellSearchResult(
            query=query,
            category=category,
            operator_name=operator_name,
            matches=[],
            total_matches=0,
            files_searched=0,
            details="请提供 operator_name 或 query 参数以执行搜索。",
        )

    # Search using grep
    matches: List[dict] = []
    files_searched = 0

    for term in search_terms:
        try:
            # grep -rn --include='*.md' -i 'term' search_dir
            cmd = ["grep", "-rn", "--include=*.md", "-i", "-C", "1", term, search_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            if result.stdout:
                # Parse grep output: file:line:content
                current_file = None
                for raw_line in result.stdout.splitlines():
                    # Match file:line:content or file:line-content or --separator--
                    if raw_line.startswith("--"):
                        continue

                    m = re.match(r"^(.+?):(\d+):\s*(.*)", raw_line)
                    if m:
                        filepath = m.group(1)
                        line_num = int(m.group(2))
                        content = m.group(3).strip()
                        files_searched += 1

                        # Limit content length
                        if len(content) > 200:
                            content = content[:200] + "..."

                        matches.append({
                            "file": os.path.relpath(filepath, knowledge_root),
                            "line": line_num,
                            "content": content,
                            "match_term": term,
                        })
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    # Deduplicate by (file, line, term)
    seen = set()
    unique_matches = []
    for m in matches:
        key = (m["file"], m["line"], m["match_term"])
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    details = (
        f"搜索完成: category='{category}', operator='{operator_name}', query='{query}'\n"
        f"搜索目录: {os.path.relpath(search_dir, knowledge_root) or 'Knowledge/'}\n"
        f"匹配结果: {len(unique_matches)} 条\n"
        f"涉及文件: {len(set(m['file'] for m in unique_matches))} 个"
    )

    return KBShellSearchResult(
        query=query,
        category=category,
        operator_name=operator_name,
        matches=unique_matches[:50],  # Limit output
        total_matches=len(unique_matches),
        files_searched=files_searched,
        details=details,
    )


# ============================================================
# Retriever class
# ============================================================

class KBShellSearchRetriever:
    """
    Knowledge base shell search retriever.

    Uses grep/find to search Knowledge/ directory documents
    by category and operator name.
    """

    def __init__(self):
        self._knowledge_root = _get_knowledge_root()

    def is_available(self) -> bool:
        """Check if Knowledge directory exists."""
        return self._knowledge_root is not None

    def search(
        self,
        category: str = "all",
        operator_name: str = "",
        query: str = "",
    ) -> KBShellSearchResult:
        """
        Search Knowledge/ directory documents.

        Args:
            category: Knowledge category
                      (api/tiling/arch/code-review/reduction/index-tracking/all)
            operator_name: Operator name (e.g., "gelu", "softmax")
            query: Additional search query text

        Returns:
            KBShellSearchResult with matching documents
        """
        return search_kb(category, operator_name, query)

    def list_categories(self) -> List[str]:
        """List available knowledge categories."""
        return [k for k in _CATEGORY_DIRS if k != "all"]
