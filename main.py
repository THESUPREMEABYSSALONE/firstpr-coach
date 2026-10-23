"""FirstPR Coach: a Gemma 4 mentor that helps you make your first open-source pull request.

    python main.py check
    python main.py find OWNER/REPO --skills "python, html"
    python main.py plan https://github.com/OWNER/REPO/issues/123 --image error.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from THIS folder, wherever you run the command from.
load_dotenv(Path(__file__).resolve().parent / ".env")

from coach import check_understanding, find_issues, make_plan, preflight_text  # noqa: E402
from gemma_client import GemmaClient, GemmaError  # noqa: E402
from github_client import GitHubClient, GitHubError  # noqa: E402

BAR = "=" * 64


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skills", default="", help='what you already know, e.g. "python, html"')
    p.add_argument("--level", default="beginner", choices=["beginner", "intermediate", "advanced"])
    p.add_argument("--lang", default="English", help="language for the explanations (default English)")
    p.add_argument("--model", default=None, help="26b (default, faster) or 31b (stronger reasoning)")
    p.add_argument("--fast", action="store_true", help="turn thinking mode down to save time and quota")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="firstpr", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="test your API key and confirm Gemma 4 answers")
    c.add_argument("--model", default=None)

    f = sub.add_parser("find", help="rank a repo's beginner issues for YOU")
    f.add_argument("repo", help="OWNER/REPO or a github.com link")
    add_common(f)

    p = sub.add_parser("plan", help="turn one issue into a coaching plan")
    p.add_argument("issue_url", help="https://github.com/OWNER/REPO/issues/123")
    p.add_argument("--image", default=None, help="screenshot of an error / failing test / UI (png, jpg, webp)")
    p.add_argument("--save", default=None, help="save the full plan to this .md file")
    p.add_argument("--hints-all", action="store_true", help="show every hint at once")
    p.add_argument("--no-explain-back", action="store_true", help="skip the understanding check")
    p.add_argument("--max-tool-steps", type=int, default=6, help="max repo-reading tool calls (default 6)")
    add_common(p)
    return parser


def make_gemma(args, tool_steps: int = 6) -> GemmaClient:
    thinking = "minimal" if getattr(args, "fast", False) else "high"
    return GemmaClient(model=args.model, thinking=thinking, max_tool_steps=tool_steps)


def footer(gemma: GemmaClient, gh: GitHubClient | None = None) -> None:
    print(f"\n{BAR}")
    print(f"Gemma 4 calls this run: {gemma.calls}  |  model: {gemma.model}  |  via Gemini API")
    if gh is not None:
        print(f"GitHub requests this run: {gh.requests_made}")
    for note in gemma.notes:
        print(f"Note: {note}")


def cmd_check(args) -> None:
    gemma = GemmaClient(model=args.model, thinking="minimal")
    reply = gemma.check()
    print(f"OK. {gemma.model} answered: {reply!r}")
    print("Your API key works and Gemma 4 is reachable through the Gemini API.")


def cmd_find(args) -> None:
    gemma, gh = make_gemma(args), GitHubClient()
    result = find_issues(gh, gemma, args.repo, args.skills, args.level, args.lang)
    print(f"\n{BAR}\nBEST ISSUES FOR YOU\n{BAR}\n{result}")
    footer(gemma, gh)


def cmd_plan(args) -> None:
    gemma, gh = make_gemma(args, args.max_tool_steps), GitHubClient()
    res = make_plan(gh, gemma, args.issue_url, args.skills, args.level, args.lang, args.image)

    print(f"\n{BAR}\n#{res.issue.number} {res.issue.title}\n{res.issue.url}\n{BAR}")
    print("PREFLIGHT (checked by code, not AI)")
    print(preflight_text(res.notes))
    print(f"{BAR}\n{res.plan_text}")

    interactive = sys.stdin.isatty()
    if res.hints:
        if args.hints_all or not interactive:
            for i, h in enumerate(res.hints, 1):
                print(f"\nHINT {i}: {h}")
        else:
            print(f"\n{BAR}\nTry it yourself first. Press Enter to unlock a hint.")
            for i, h in enumerate(res.hints, 1):
                input(f"  [Enter] for hint {i} of {len(res.hints)} ... ")
                print(f"\nHINT {i}: {h}\n")

    feedback = ""
    if interactive and not args.no_explain_back:
        print(f"{BAR}\nUNDERSTANDING CHECK (optional)")
        print("In your own words, what will you change and why? Press Enter on an empty line to skip.")
        mine = input("> ").strip()
        if mine:
            feedback = check_understanding(gemma, res.full_text, mine, args.lang)
            print(f"\n{feedback}")

    if args.save:
        out = Path(args.save)
        out.write_text(
            f"# #{res.issue.number} {res.issue.title}\n{res.issue.url}\n\n"
            f"## Preflight\n{preflight_text(res.notes)}\n\n{res.full_text}\n"
            + (f"\n## Understanding check\n{feedback}\n" if feedback else ""),
            encoding="utf-8",
        )
        print(f"\nSaved to {out}")
    footer(gemma, gh)


def main(argv: list[str] | None = None) -> int:
    # Make Windows consoles happy with any characters the model returns.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        {"check": cmd_check, "find": cmd_find, "plan": cmd_plan}[args.cmd](args)
    except (GemmaError, GitHubError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
