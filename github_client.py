"""Tiny GitHub REST helper (uses only `requests`, no GitHub SDK).

Everything here reads PUBLIC data. A GITHUB_TOKEN is optional: without one
GitHub allows about 60 requests per hour, which is enough for a few plans.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import requests

API = "https://api.github.com"
PLACEHOLDER_TOKENS = {"", "optional", "your_github_token_here", "paste_token_here"}

ISSUE_URL_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s]+)/issues/(\d+)")
REPO_RE = re.compile(r"^(?:https?://)?(?:www\.)?(?:github\.com/)?([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$")


class GitHubError(Exception):
    """Raised with a friendly message when GitHub says no."""


def parse_issue_url(url: str) -> tuple[str, str, int]:
    m = ISSUE_URL_RE.search(url.strip())
    if not m:
        raise GitHubError(
            "That does not look like a GitHub issue link. Expected something like "
            "https://github.com/OWNER/REPO/issues/123"
        )
    return m.group(1), m.group(2), int(m.group(3))


def parse_repo(text: str) -> tuple[str, str]:
    m = REPO_RE.match(text.strip())
    if not m:
        raise GitHubError("Give the repo as OWNER/REPO or a github.com link.")
    return m.group(1), m.group(2)


@dataclass
class IssueData:
    owner: str
    repo: str
    number: int
    title: str
    body: str
    state: str
    url: str
    labels: list[str]
    assignees: list[str]
    created_at: str
    updated_at: str
    comments: list[dict] = field(default_factory=list)  # {"author", "body", "created_at"}


@dataclass
class RepoData:
    owner: str
    name: str
    default_branch: str
    description: str
    language: str
    tree_paths: list[str]
    tree_truncated: bool
    readme: str
    contributing: str
    contributing_path: str


class GitHubClient:
    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "firstpr-coach",
            }
        )
        token = (token if token is not None else os.getenv("GITHUB_TOKEN", "")).strip()
        if token.lower() not in PLACEHOLDER_TOKENS:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.requests_made = 0

    # ------------------------------------------------------------------ core
    def _get(self, path: str, params: dict | None = None, raw: bool = False):
        headers = {"Accept": "application/vnd.github.raw+json"} if raw else None
        try:
            r = self.session.get(API + path, params=params, headers=headers, timeout=25)
        except requests.RequestException as e:  # offline, DNS, timeout...
            raise GitHubError(f"Could not reach GitHub: {e}") from e
        self.requests_made += 1
        if r.status_code == 404:
            raise GitHubError(f"Not found on GitHub: {path} (private repo or wrong link?)")
        if r.status_code in (403, 429):
            if r.headers.get("X-RateLimit-Remaining") == "0" or r.status_code == 429:
                raise GitHubError(
                    "GitHub rate limit reached. Wait a while, or add a free GITHUB_TOKEN "
                    "to your .env file to raise the limit."
                )
            raise GitHubError(f"GitHub refused the request (403): {r.text[:200]}")
        if not r.ok:
            raise GitHubError(f"GitHub error {r.status_code}: {r.text[:200]}")
        return r.text if raw else r.json()

    # --------------------------------------------------------------- repo bits
    def repo_info(self, owner: str, name: str) -> dict:
        return self._get(f"/repos/{owner}/{name}")

    def tree(self, owner: str, name: str, branch: str) -> tuple[list[str], bool]:
        data = self._get(f"/repos/{owner}/{name}/git/trees/{branch}", params={"recursive": "1"})
        paths = [t["path"] for t in data.get("tree", []) if t.get("type") == "blob"]
        return paths, bool(data.get("truncated"))

    def readme_text(self, owner: str, name: str) -> str:
        try:
            return self._get(f"/repos/{owner}/{name}/readme", raw=True)
        except GitHubError:
            return ""

    def file_text(self, owner: str, name: str, path: str, ref: str | None = None) -> str:
        params = {"ref": ref} if ref else None
        return self._get(f"/repos/{owner}/{name}/contents/{path}", params=params, raw=True)

    # -------------------------------------------------------------- issue bits
    def issue(self, owner: str, name: str, number: int) -> IssueData:
        data = self._get(f"/repos/{owner}/{name}/issues/{number}")
        if "pull_request" in data:
            raise GitHubError("That link is a pull request, not an issue. Paste an issue link.")
        comments_raw = self._get(
            f"/repos/{owner}/{name}/issues/{number}/comments", params={"per_page": 30}
        )
        comments = [
            {
                "author": (c.get("user") or {}).get("login", "unknown"),
                "body": c.get("body") or "",
                "created_at": c.get("created_at", ""),
            }
            for c in comments_raw
        ]
        return IssueData(
            owner=owner,
            repo=name,
            number=number,
            title=data.get("title", ""),
            body=data.get("body") or "",
            state=data.get("state", "open"),
            url=data.get("html_url", f"https://github.com/{owner}/{name}/issues/{number}"),
            labels=[l["name"] for l in data.get("labels", []) if isinstance(l, dict)],
            assignees=[a["login"] for a in data.get("assignees", [])],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            comments=comments,
        )

    def beginner_issues(self, owner: str, name: str, limit: int = 15) -> tuple[list[dict], str]:
        """Open issues carrying a beginner-friendly label. Returns (issues, label_used)."""
        for label in ("good first issue", "good-first-issue", "help wanted", "beginner", "easy"):
            data = self._get(
                f"/repos/{owner}/{name}/issues",
                params={"state": "open", "labels": label, "per_page": limit, "sort": "updated"},
            )
            issues = [i for i in data if "pull_request" not in i]
            if issues:
                return issues[:limit], label
        return [], ""


# ------------------------------------------------------------ bundle builder
CONTRIBUTING_RE = re.compile(r"(^|/)contributing(\.(md|rst|txt|markdown))?$", re.IGNORECASE)


def find_contributing(paths: list[str]) -> str:
    candidates = [p for p in paths if CONTRIBUTING_RE.search(p)]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda p: (p.count("/"), len(p)))[0]


def load_repo(gh: GitHubClient, owner: str, name: str) -> RepoData:
    info = gh.repo_info(owner, name)
    branch = info.get("default_branch", "main")
    paths, truncated = gh.tree(owner, name, branch)
    contributing_path = find_contributing(paths)
    contributing = ""
    if contributing_path:
        try:
            contributing = gh.file_text(owner, name, contributing_path, branch)
        except GitHubError:
            contributing = ""
    return RepoData(
        owner=owner,
        name=name,
        default_branch=branch,
        description=info.get("description") or "",
        language=info.get("language") or "unknown",
        tree_paths=paths,
        tree_truncated=truncated,
        readme=gh.readme_text(owner, name),
        contributing=contributing,
        contributing_path=contributing_path,
    )
