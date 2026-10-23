"""FirstPR Coach logic. Gemma calls happen through gemma_client.GemmaClient."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from github_client import (
    GitHubClient,
    GitHubError,
    IssueData,
    RepoData,
    load_repo,
    parse_issue_url,
    parse_repo,
)
from gemma_client import GemmaClient
import prompts

# Generous limits on purpose: Gemma 4 has a long context window.
MAX_README = 12000
MAX_CONTRIBUTING = 8000
MAX_ISSUE_BODY = 6000
MAX_COMMENT = 900
MAX_TREE_PATHS = 600
MAX_FILE_LINES = 250
MAX_FILE_CHARS = 14000

CLAIM_PATTERNS = [
    r"\bi('?d| would) (like|love) to (work|take|fix|try|pick|tackle)",
    r"\bcan i (work|take|try|pick|have|be assigned|tackle)",
    r"\bmay i (work|take|try)",
    r"\bi('?ll| will) (work|take|fix|try|pick|submit|open|send|tackle)",
    r"\b(working|work) on (this|it)\b",
    r"\bassign (this |the )?(issue )?to me\b",
    r"\bi('?m| am) (working|on it|taking|picking)",
    r"\bclaim(ed|ing)?\b",
]
LINKED_PR_RE = re.compile(r"(?i)\b(pr|pull request|fixes|closes|resolves)\s*#?\s*\d+")


# ------------------------------------------------------------------ preflight
def claim_signals(comments: list[dict]) -> list[str]:
    hits = []
    for c in comments:
        body = c.get("body", "")
        if any(re.search(p, body, re.IGNORECASE) for p in CLAIM_PATTERNS) or LINKED_PR_RE.search(body):
            hits.append(c.get("author", "someone"))
    return hits


def _days_since(iso: str) -> int | None:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - dt).days


def preflight(issue: IssueData, repo: RepoData) -> list[tuple[str, str]]:
    """Deterministic checks (no AI). Returns [(level, message)] with level in !, i, ok."""
    notes: list[tuple[str, str]] = []
    if issue.state != "open":
        notes.append(("!", "This issue is CLOSED. Pick an open one."))
    if issue.assignees:
        notes.append(("!", f"Assigned to {', '.join('@' + a for a in issue.assignees)}. Ask before starting."))
    hits = claim_signals(issue.comments)
    if hits:
        who = ", ".join(sorted({'@' + h for h in hits}))
        notes.append(("!", f"{len(hits)} comment(s) look like someone claimed it or linked a PR ({who}). Read them first."))
    age = _days_since(issue.updated_at)
    if age is not None and age > 365:
        notes.append(("!", f"No activity for {age} days. Maintainers may not respond; the issue may be stale."))
    if repo.contributing_path:
        notes.append(("i", f"Contribution rules found: {repo.contributing_path}"))
    else:
        notes.append(("i", "No CONTRIBUTING file found. Check the README for rules."))
    if not any(level == "!" for level, _ in notes):
        notes.append(("ok", "No red flags found: open, unassigned, no claim-like comments."))
    return notes


def preflight_text(notes: list[tuple[str, str]]) -> str:
    return "\n".join(f"- [{lvl}] {msg}" for lvl, msg in notes)


# ------------------------------------------------------------ tools for Gemma
TOOL_DECLARATIONS = [
    {
        "name": "find_files",
        "description": "Search file paths in the repository by keyword (case-insensitive). Use it to locate code related to the issue.",
        "parameters": {
            "type": "object",
            "properties": {"keyword": {"type": "string", "description": "Word to look for in file paths, e.g. 'login' or 'parser'"}},
            "required": ["keyword"],
        },
    },
    {
        "name": "list_repo_files",
        "description": "List the files and folders directly inside a directory of the repository. Use an empty string for the top level.",
        "parameters": {
            "type": "object",
            "properties": {"directory": {"type": "string", "description": "Directory such as 'src/utils', or '' for the root"}},
            "required": ["directory"],
        },
    },
    {
        "name": "read_repo_file",
        "description": f"Read a text file from the repository. Returns numbered lines (at most {MAX_FILE_LINES} lines per call; use start_line to continue).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path exactly as in the repository"},
                "start_line": {"type": "integer", "description": "Optional 1-based line to start from"},
            },
            "required": ["path"],
        },
    },
]


class RepoTools:
    """The functions Gemma is allowed to call. Read-only."""

    def __init__(self, gh: GitHubClient, repo: RepoData):
        self.gh = gh
        self.repo = repo
        self._cache: dict[str, str] = {}

    def functions(self) -> dict:
        return {
            "find_files": self.find_files,
            "list_repo_files": self.list_repo_files,
            "read_repo_file": self.read_repo_file,
        }

    def find_files(self, keyword: str) -> str:
        kw = str(keyword).lower().strip()
        if not kw:
            return "Give a non-empty keyword."
        matches = [p for p in self.repo.tree_paths if kw in p.lower()]
        if not matches:
            return f"No file paths contain '{keyword}'."
        shown = matches[:60]
        extra = f"\n(+{len(matches) - 60} more, use a narrower keyword)" if len(matches) > 60 else ""
        return "\n".join(shown) + extra

    def list_repo_files(self, directory: str = "") -> str:
        d = str(directory).strip().strip("/")
        prefix = f"{d}/" if d else ""
        children: set[str] = set()
        for p in self.repo.tree_paths:
            if p.startswith(prefix):
                rest = p[len(prefix):]
                head, sep, _ = rest.partition("/")
                children.add(head + ("/" if sep else ""))
        if not children:
            return f"Nothing found under '{directory}'."
        return "\n".join(sorted(children)[:200])

    def read_repo_file(self, path: str, start_line: int = 1) -> str:
        path = str(path).strip().lstrip("/")
        if path not in self.repo.tree_paths and not self.repo.tree_truncated:
            return f"'{path}' is not in the repository file list. Use find_files first."
        try:
            start = max(1, int(start_line or 1))
        except (TypeError, ValueError):
            start = 1
        if path not in self._cache:
            try:
                self._cache[path] = self.gh.file_text(self.repo.owner, self.repo.name, path, self.repo.default_branch)
            except GitHubError as e:
                return f"Could not read file: {e}"
        text = self._cache[path]
        if "\x00" in text[:2000]:
            return "That looks like a binary file, skipping."
        lines = text.splitlines()
        chunk = lines[start - 1 : start - 1 + MAX_FILE_LINES]
        out = "\n".join(f"{i}: {line}" for i, line in enumerate(chunk, start))
        if len(out) > MAX_FILE_CHARS:
            out = out[:MAX_FILE_CHARS] + "\n...(truncated)"
        more = len(lines) - (start - 1 + len(chunk))
        if more > 0:
            out += f"\n...({more} more lines; call again with start_line={start + len(chunk)})"
        return out or "(empty file)"


# --------------------------------------------------------------- prompt build
def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n...(clipped)"


def build_plan_prompt(issue: IssueData, repo: RepoData, notes, skills: str, level: str,
                      lang: str, has_image: bool, tool_steps: int) -> str:
    comments = "\n".join(
        f"- @{c['author']}: {_clip(c['body'], MAX_COMMENT)}" for c in issue.comments[:12]
    ) or "(no comments)"
    tree = "\n".join(repo.tree_paths[:MAX_TREE_PATHS])
    if len(repo.tree_paths) > MAX_TREE_PATHS or repo.tree_truncated:
        tree += "\n...(file list clipped; use find_files / list_repo_files)"
    image_line = (
        "\nThe learner attached a screenshot (error, failing test or UI). Read it and use what it shows.\n"
        if has_image else ""
    )
    return f"""{prompts.language_line(lang)}

LEARNER: skills = {skills or 'not given'}; level = {level}.
REPOSITORY: {repo.owner}/{repo.name}. {repo.description} (main language: {repo.language})

PREFLIGHT FACTS (computed by code, trust these):
{preflight_text(notes)}

ISSUE #{issue.number}: {issue.title}
State: {issue.state} | Labels: {', '.join(issue.labels) or 'none'} | Assignees: {', '.join(issue.assignees) or 'none'}
{_clip(issue.body, MAX_ISSUE_BODY)}

COMMENTS:
{comments}

README:
{_clip(repo.readme, MAX_README) or '(none)'}

CONTRIBUTING ({repo.contributing_path or 'not found'}):
{_clip(repo.contributing, MAX_CONTRIBUTING) or '(none)'}

FILE LIST:
{tree}
{image_line}
TASK: First use your tools to look at the files most likely involved (at most {tool_steps} tool calls). Then write the coaching plan. Remember: coach, never write the fix.

{prompts.PLAN_FORMAT}"""


# ------------------------------------------------------------------ hint split
HINTS_HEADING = re.compile(r"(?mi)^##\s*hints?\b.*$")
NEXT_HEADING = re.compile(r"(?m)^##\s")
HINT_SPLIT = re.compile(r"(?mi)^\s*[*_]*hint\s*(\d+)[*_]*\s*[:.\-\u2013]\s*[*_]*")


def split_hints(plan: str) -> tuple[str, list[str]]:
    """Pull the HINT lines out so the CLI can reveal them one at a time.

    Returns (plan_without_hint_text, hints). If parsing fails, returns (plan, []).
    """
    m = HINTS_HEADING.search(plan)
    if not m:
        return plan, []
    end_m = NEXT_HEADING.search(plan, m.end())
    end = end_m.start() if end_m else len(plan)
    block = plan[m.end():end]
    pieces = HINT_SPLIT.split(block)
    # split() with one capture group -> [pre, num, text, num, text, ...]
    hints = [pieces[i + 1].strip() for i in range(1, len(pieces) - 1, 2)]
    hints = [h for h in hints if h]
    if len(hints) < 2:
        return plan, []
    stub = "## Hints\n(Hidden so you can try first. They unlock one at a time below.)\n\n"
    return plan[: m.start()] + stub + plan[end:], hints


# ---------------------------------------------------------------------- flows
@dataclass
class PlanResult:
    issue: IssueData
    repo: RepoData
    notes: list[tuple[str, str]]
    plan_text: str            # hints hidden
    hints: list[str] = field(default_factory=list)
    full_text: str = ""       # everything, hints included


def make_plan(gh: GitHubClient, gemma: GemmaClient, url: str, skills: str, level: str,
              lang: str, image_path: str | None = None, say=print) -> PlanResult:
    owner, name, number = parse_issue_url(url)
    say(f"Reading {owner}/{name} issue #{number} from GitHub ...")
    repo = load_repo(gh, owner, name)
    issue = gh.issue(owner, name, number)
    notes = preflight(issue, repo)

    tools = RepoTools(gh, repo)
    prompt = build_plan_prompt(issue, repo, notes, skills, level, lang, bool(image_path), gemma.max_tool_steps)
    say(f"Asking {gemma.model} to study the repo (it can read files with function calling) ...")

    def on_tool(tool_name: str, args: dict):
        shown = ", ".join(f"{k}={v}" for k, v in args.items())
        say(f"   Gemma calls {tool_name}({shown})")

    text = gemma.run_with_tools(
        prompt=prompt,
        system=prompts.COACH_SYSTEM,
        declarations=TOOL_DECLARATIONS,
        functions=tools.functions(),
        image_path=image_path,
        on_tool=on_tool,
    )
    hidden, hints = split_hints(text)
    return PlanResult(issue=issue, repo=repo, notes=notes, plan_text=hidden, hints=hints, full_text=text)


def find_issues(gh: GitHubClient, gemma: GemmaClient, repo_text: str, skills: str,
                level: str, lang: str, say=print) -> str:
    owner, name = parse_repo(repo_text)
    say(f"Looking for beginner-friendly issues in {owner}/{name} ...")
    issues, label = gh.beginner_issues(owner, name)
    if not issues:
        return f"No open issues labelled good first issue / help wanted in {owner}/{name}. Try another repo."
    rows = []
    for i in issues:
        assignees = [a["login"] for a in i.get("assignees", [])]
        body = _clip(i.get("body") or "", 700).replace("\n", " ")
        rows.append(
            f"#{i['number']} | {i['title']} | labels: {', '.join(l['name'] for l in i.get('labels', []))} | "
            f"comments: {i.get('comments', 0)} | assigned: {', '.join(assignees) or 'no'} | "
            f"updated: {i.get('updated_at', '')[:10]} | {i.get('html_url', '')}\n    {body}"
        )
    prompt = f"""{prompts.language_line(lang)}
LEARNER: skills = {skills or 'not given'}; level = {level}.
REPOSITORY: {owner}/{name}. These open issues were found with the label "{label}":

""" + "\n".join(rows) + f"\n\n{prompts.FIND_FORMAT}"
    say(f"Asking {gemma.model} to rank {len(issues)} issues for you ...")
    return gemma.ask(prompt, system=prompts.FIND_SYSTEM)


def check_understanding(gemma: GemmaClient, plan_text: str, explanation: str, lang: str) -> str:
    prompt = f"""{prompts.language_line(lang)}

THE COACHING PLAN YOU GAVE (hints included):
{plan_text}

THE LEARNER'S OWN EXPLANATION OF WHAT THEY WILL CHANGE:
{explanation}
"""
    return gemma.ask(prompt, system=prompts.EXPLAIN_BACK_SYSTEM)
