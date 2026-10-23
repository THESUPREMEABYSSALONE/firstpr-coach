"""Run with:  python -m unittest discover -s tests -v
No API key or internet needed: Gemma and GitHub are faked."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types  # noqa: E402

import coach  # noqa: E402
from gemma_client import GemmaClient, GemmaError, resolve_model  # noqa: E402
from github_client import GitHubError, IssueData, RepoData, find_contributing, parse_issue_url, parse_repo  # noqa: E402


def make_repo(paths=None):
    return RepoData(
        owner="o", name="r", default_branch="main", description="demo repo", language="Python",
        tree_paths=paths or ["README.md", "src/app.py", "src/utils/helpers.py", "docs/CONTRIBUTING.md"],
        tree_truncated=False, readme="# demo", contributing="rules", contributing_path="docs/CONTRIBUTING.md",
    )


def make_issue(**kw):
    base = dict(owner="o", repo="r", number=7, title="Fix typo", body="There is a bug", state="open",
                url="https://github.com/o/r/issues/7", labels=["good first issue"], assignees=[],
                created_at="2026-10-01T00:00:00Z", updated_at="2026-10-20T00:00:00Z", comments=[])
    base.update(kw)
    return IssueData(**base)


class FakeGitHub:
    requests_made = 0

    def file_text(self, owner, name, path, ref=None):
        return "\n".join(f"line {i}" for i in range(1, 400))


class ParsingTests(unittest.TestCase):
    def test_issue_url(self):
        self.assertEqual(parse_issue_url("https://github.com/a/b/issues/12"), ("a", "b", 12))
        with self.assertRaises(GitHubError):
            parse_issue_url("https://github.com/a/b/pull/12")

    def test_repo(self):
        self.assertEqual(parse_repo("a/b"), ("a", "b"))
        self.assertEqual(parse_repo("https://github.com/a/b.git"), ("a", "b"))

    def test_contributing_finder(self):
        self.assertEqual(find_contributing(["a.py", ".github/CONTRIBUTING.md", "CONTRIBUTING.md"]), "CONTRIBUTING.md")
        self.assertEqual(find_contributing(["a.py"]), "")

    def test_model_alias(self):
        self.assertEqual(resolve_model("31b"), "gemma-4-31b-it")
        self.assertEqual(resolve_model(None), "gemma-4-26b-a4b-it")


class PreflightTests(unittest.TestCase):
    def test_clean_issue(self):
        notes = coach.preflight(make_issue(), make_repo())
        self.assertTrue(any(l == "ok" for l, _ in notes))

    def test_assigned_closed_claimed(self):
        issue = make_issue(state="closed", assignees=["sam"],
                           comments=[{"author": "kay", "body": "Can I work on this?", "created_at": ""}])
        text = coach.preflight_text(coach.preflight(issue, make_repo()))
        self.assertIn("CLOSED", text)
        self.assertIn("@sam", text)
        self.assertIn("@kay", text)

    def test_stale(self):
        issue = make_issue(updated_at="2020-01-01T00:00:00Z")
        self.assertIn("stale", coach.preflight_text(coach.preflight(issue, make_repo())))


class HintTests(unittest.TestCase):
    PLAN = ("## Where to look\n- a\n\n## Hints\nHINT 1: Ask why.\nHINT 2: Look at the parser,\nthen check the loop.\n"
            "**HINT 3:** Describe it in words.\n\n## Before you open the pull request\n- test\n")

    def test_split(self):
        hidden, hints = coach.split_hints(self.PLAN)
        self.assertEqual(len(hints), 3)
        self.assertTrue(hints[1].startswith("Look at the parser"))
        self.assertNotIn("Ask why", hidden)
        self.assertIn("Before you open the pull request", hidden)

    def test_no_hints_returns_original(self):
        text = "## Hints\nnothing useful here"
        self.assertEqual(coach.split_hints(text), (text, []))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tools = coach.RepoTools(FakeGitHub(), make_repo())

    def test_find(self):
        self.assertIn("src/utils/helpers.py", self.tools.find_files("helper"))
        self.assertIn("No file paths", self.tools.find_files("zzz"))

    def test_list(self):
        out = self.tools.list_repo_files("src")
        self.assertIn("app.py", out)
        self.assertIn("utils/", out)

    def test_read_paginates(self):
        out = self.tools.read_repo_file("src/app.py")
        self.assertIn("1: line 1", out)
        self.assertIn("start_line=251", out)
        self.assertIn("251: line 251", self.tools.read_repo_file("src/app.py", 251))

    def test_read_unknown_file(self):
        self.assertIn("not in the repository", self.tools.read_repo_file("nope.py"))


class FakeModels:
    """Pretends to be client.models: first asks for a tool, then answers."""

    def __init__(self):
        self.requests = []

    def generate_content(self, model, contents, config):
        self.requests.append((model, contents, config))
        if len(self.requests) == 1:
            fc = types.FunctionCall(name="find_files", args={"keyword": "app"}, id="1")
            content = types.Content(role="model", parts=[types.Part(function_call=fc)])
            return SimpleNamespace(function_calls=[fc], candidates=[SimpleNamespace(content=content)], text=None)
        return SimpleNamespace(function_calls=None, candidates=[], text="## Final plan")


class GemmaLoopTests(unittest.TestCase):
    def test_tool_loop_runs_function_then_answers(self):
        fake = SimpleNamespace(models=FakeModels(), files=None)
        gemma = GemmaClient(api_key="x", client=fake, sleep=lambda s: None)
        seen = []
        out = gemma.run_with_tools(
            prompt="go", system="sys", declarations=coach.TOOL_DECLARATIONS,
            functions=coach.RepoTools(FakeGitHub(), make_repo()).functions(),
            on_tool=lambda n, a: seen.append(n),
        )
        self.assertEqual(out, "## Final plan")
        self.assertEqual(seen, ["find_files"])
        self.assertEqual(gemma.calls, 2)
        model, contents, config = fake.models.requests[1]
        self.assertEqual(model, "gemma-4-26b-a4b-it")
        self.assertEqual(len(contents), 3)  # user, model call, tool result
        self.assertEqual(config.thinking_config.thinking_level.value.lower(), "high")

    def test_empty_answer_is_an_error(self):
        class Empty:
            def generate_content(self, **kw):
                return SimpleNamespace(function_calls=None, candidates=[], text="")
        gemma = GemmaClient(api_key="x", client=SimpleNamespace(models=Empty()))
        with self.assertRaises(GemmaError):
            gemma.ask("hi")

    def test_missing_key_message(self):
        import os
        from gemma_client import get_api_key
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(k, None)
        with self.assertRaises(GemmaError) as ctx:
            get_api_key()
        self.assertIn(".env", str(ctx.exception))



class FakeResponse:
    def __init__(self, payload=None, text="", status=200, headers=None):
        self._payload, self.text, self.status_code, self.headers = payload, text, status, headers or {}
        self.ok = status < 400

    def json(self):
        return self._payload


class FakeSession:
    """Pretends to be requests.Session so GitHubClient can be tested offline."""

    def __init__(self, routes):
        self.headers, self.routes, self.seen = {}, routes, []

    def get(self, url, params=None, headers=None, timeout=None):
        path = url.replace("https://api.github.com", "")
        self.seen.append((path, params, headers))
        for prefix, resp in self.routes.items():
            if path == prefix or path.startswith(prefix):
                return resp
        return FakeResponse(status=404, text="nope")


class GitHubClientTests(unittest.TestCase):
    def test_load_repo_and_issue(self):
        from github_client import GitHubClient, load_repo
        routes = {
            "/repos/o/r/git/trees/main": FakeResponse({"truncated": False, "tree": [
                {"type": "blob", "path": "README.md"}, {"type": "tree", "path": "src"},
                {"type": "blob", "path": "CONTRIBUTING.md"}, {"type": "blob", "path": "src/a.py"}]}),
            "/repos/o/r/readme": FakeResponse(text="# Hello"),
            "/repos/o/r/contents/CONTRIBUTING.md": FakeResponse(text="Be nice"),
            "/repos/o/r/issues/5/comments": FakeResponse([{"user": {"login": "kay"}, "body": "I'll take this", "created_at": "2026-10-01T00:00:00Z"}]),
            "/repos/o/r/issues/5": FakeResponse({"title": "Bug", "body": "broken", "state": "open", "html_url": "u",
                                                 "labels": [{"name": "good first issue"}], "assignees": [],
                                                 "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-02T00:00:00Z"}),
            "/repos/o/r": FakeResponse({"default_branch": "main", "description": "demo", "language": "Python"}),
        }
        gh = GitHubClient(token="", session=FakeSession(routes))
        repo = load_repo(gh, "o", "r")
        self.assertEqual(repo.tree_paths, ["README.md", "CONTRIBUTING.md", "src/a.py"])  # folders excluded
        self.assertEqual(repo.contributing, "Be nice")
        issue = gh.issue("o", "r", 5)
        self.assertEqual(issue.labels, ["good first issue"])
        self.assertEqual(coach.claim_signals(issue.comments), ["kay"])

    def test_pull_request_link_is_rejected(self):
        from github_client import GitHubClient
        routes = {"/repos/o/r/issues/9": FakeResponse({"pull_request": {}, "title": "x"})}
        with self.assertRaises(GitHubError):
            GitHubClient(token="", session=FakeSession(routes)).issue("o", "r", 9)

    def test_beginner_issues_skip_prs_and_fall_back_to_next_label(self):
        from github_client import GitHubClient

        class Sess(FakeSession):
            def get(self, url, params=None, headers=None, timeout=None):
                label = (params or {}).get("labels")
                if label == "good first issue":
                    return FakeResponse([{"number": 1, "pull_request": {}}])  # only a PR -> treated as empty
                if label == "good-first-issue":
                    return FakeResponse([])
                return FakeResponse([{"number": 2, "title": "real"}])  # "help wanted"
        issues, label = GitHubClient(token="", session=Sess({})).beginner_issues("o", "r")
        self.assertEqual((label, [i["number"] for i in issues]), ("help wanted", [2]))

    def test_rate_limit_message(self):
        from github_client import GitHubClient
        sess = FakeSession({"/repos/o/r": FakeResponse(status=403, text="limit", headers={"X-RateLimit-Remaining": "0"})})
        with self.assertRaises(GitHubError) as ctx:
            GitHubClient(token="", session=sess).repo_info("o", "r")
        self.assertIn("GITHUB_TOKEN", str(ctx.exception))

    def test_raw_header_used_for_file_text(self):
        from github_client import GitHubClient
        sess = FakeSession({"/repos/o/r/contents/a.py": FakeResponse(text="print(1)")})
        GitHubClient(token="", session=sess).file_text("o", "r", "a.py", "main")
        self.assertEqual(sess.seen[0][2], {"Accept": "application/vnd.github.raw+json"})
        self.assertEqual(sess.seen[0][1], {"ref": "main"})


if __name__ == "__main__":
    unittest.main()
