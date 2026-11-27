"""Watch ODE live: one line per request the gateway handles, as it happens (demo view).

Run it in a second terminal while you use `tokencut` with TOKENCUT_ODE=on (or shadow):

    python -m eval.ode_watch                 # follows the newest gateway log
    python -m eval.ode_watch --replay        # also prints what is already in it
    python -m eval.ode_watch --log-dir logs/ode-live/on/gateway --replay

For each typed task: the router's guess, ODE's choice and the per-arm scores behind it. For
the agent's tool steps: the arm kept and the model the request really went to. Side requests
(titles, summaries) are shown as untouched. Costs are list prices from the billed usage.
Reads the numbers-only log; never prints prompts or code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tokencut.ode.arms import arm as arm_info
from tokencut.ode.reward import model_for, request_cost, usage_from_dict

DIM, BOLD, GREEN, YELLOW, CYAN, MAGENTA, RESET = ("\033[2m", "\033[1m", "\033[32m", "\033[33m",
                                                  "\033[36m", "\033[35m", "\033[0m")


def default_log_dir() -> Path:
    home = os.environ.get("TOKENCUT_HOME") or str(Path.home() / ".tokencut")
    return Path(os.environ.get("TOKENCUT_LOG_DIR") or Path(home) / "logs")


def short(model: str) -> str:
    for name in ("haiku", "sonnet", "opus", "gpt"):
        if name in model.lower():
            return name.capitalize() if name != "gpt" else "GPT"
    return model or "?"


def cost_of(rec: dict, model: str) -> float | None:
    usage = rec.get("usage")
    if not usage or not model:
        return None
    try:
        return request_cost(model_for(model), usage_from_dict(usage))
    except ValueError:
        return None


class View:
    def __init__(self, color: bool):
        self.color = color
        self.total = 0.0
        self.tasks = 0

    def c(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def show(self, rec: dict) -> None:
        if rec.get("method") != "POST":
            return
        ode = rec.get("ode") or {}
        choice = ode.get("choice")
        model = ode.get("model_sent") or (rec.get("analysis") or {}).get("model", "")
        cost = cost_of(rec, model)
        if cost is not None:
            self.total += cost
        clock = (rec.get("ts") or "")[11:19]
        money = f"${cost:.4f}" if cost is not None else ""
        if choice is None:
            print(self.c(DIM, f"{clock}  side request        -> {short(model):6}  untouched   "
                              f"{money}"))
            return
        if choice.get("decided"):
            self.tasks += 1
            numbers = ode.get("task_numbers") or {}
            guess = f"{numbers.get('task')} {numbers.get('p', 0):.2f}"
            picked = choice["arm"]
            info = arm_info(picked)
            tag = self.c(YELLOW, " (random try)") if choice.get("explored") else ""
            ranked = sorted(choice.get("scores", {}).items(), key=lambda kv: -kv[1]["total"])
            kept = ""
            if choice.get("previous") == picked:
                kept = " (same model as last task)"
                if ranked and ranked[0][0] != picked and not choice.get("explored"):
                    gap = ranked[0][1]["total"] - choice["scores"][picked]["total"]
                    kept = (f" (kept: {ranked[0][0]} was only {gap:.2f} ahead, "
                            f"under the 0.02 switch margin)")
            print()
            print(self.c(BOLD, f"{clock}  NEW TASK #{self.tasks}  router: {guess}"))
            print(f"          ODE picks {self.c(GREEN, picked)}{tag}  = "
                  f"{short(info.model.model_id)}, {info.trim} trimming{kept}")
            print(self.c(DIM, "          scores: " + "  ".join(
                f"{a} {s['total']:.2f}" for a, s in ranked)))
            if ode.get("mode") == "shadow":
                print(self.c(MAGENTA, "          (shadow mode: logged only, request unchanged)"))
        where = f"-> {self.c(CYAN, short(model)):6}" if self.color else f"-> {short(model):6}"
        step = "task request" if choice.get("decided") else "agent step  "
        if (ode.get("task_numbers") or {}).get("machine") == "prompt_suggestion":
            step = "next-prompt hint"  # Claude Code's own request, not a task: arm kept
        print(f"{clock}  {step}  arm {choice.get('arm') or '-':18} {where}  {money}")

    def footer(self) -> None:
        if self.total:
            print(self.c(BOLD, f"\n  {self.tasks} tasks, ${self.total:.4f} list price"))


def newest(log_dir: Path) -> Path | None:
    files = sorted(log_dir.glob("requests-*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=default_log_dir())
    parser.add_argument("--replay", action="store_true", help="print the existing lines too")
    parser.add_argument("--once", action="store_true", help="print what is there and stop")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    if os.name == "nt":
        os.system("")  # turn on ANSI colours in the Windows console
    sys.stdout.reconfigure(encoding="utf-8")
    view = View(color=not args.no_color)
    print(f"Watching ODE in {args.log_dir}  (Ctrl+C to stop)")
    path, pos = newest(args.log_dir), 0
    if path is not None and not (args.replay or args.once):
        pos = path.stat().st_size
    try:
        while True:
            latest = newest(args.log_dir)
            if latest is not None and latest != path:
                path, pos = latest, 0  # a new day's log
            if path is not None:
                with path.open("rb") as f:
                    f.seek(pos)
                    while True:
                        line = f.readline()
                        if not line.endswith(b"\n"):
                            break  # nothing new, or a line still being written
                        pos = f.tell()
                        if line.strip():
                            try:
                                view.show(json.loads(line))
                            except (ValueError, KeyError):
                                pass
            if args.once:
                break
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    view.footer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
