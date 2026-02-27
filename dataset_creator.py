import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme(
    {
        "header": "bold bright_cyan",
        "task_id": "bold yellow",
        "section": "bold magenta",
        "hint": "dim italic",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "muted": "dim",
        "action": "bold bright_white on blue",
    }
)

console = Console(theme=THEME, highlight=True)


def clear():
    console.clear()


def header_bar(current: int, total: int, output_path: str):
    pct = int(current / total * 100) if total else 0
    filled = int(pct / 2)
    bar = "[green]" + "█" * filled + "[/green]" + "[dim]" + "░" * (50 - filled) + "[/dim]"

    left = Text.assemble(
        ("  AlphaD-RL  ", "bold bright_cyan on dark_blue"),
        ("  HumanEval Dataset Creator  ", "bold white on grey23"),
    )
    right = Text.assemble(
        (f" {current}/{total} ", "bold white on dark_green"),
        (f"  {pct}%  ", "bold yellow"),
    )

    info_row = Table.grid(expand=True)
    info_row.add_column(justify="left")
    info_row.add_column(justify="center")
    info_row.add_column(justify="right")
    info_row.add_row(
        left,
        bar,
        right,
    )
    console.print(info_row)
    console.print(
        f"  [muted]Output:[/muted] [cyan]{output_path}[/cyan]"
        f"  [muted]│[/muted]  [muted]Press [/muted][bold]S[/bold][muted] to skip · "
        f"[bold]Q[/bold][muted] to quit & save · "
        f"[bold]E[/bold][muted] to enter solution[/muted]\n"
    )


def render_task(row: dict, idx: int, total: int, canonical_visible: bool = False):
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column(style="bold")
    meta.add_row("Task ID", f"[task_id]{row['task_id']}[/task_id]")
    meta.add_row("Entry point", f"[cyan]{row['entry_point']}[/cyan]")

    console.print(Panel(meta, title="[section]Problem Metadata[/section]", border_style="blue", expand=False))

    prompt_syntax = Syntax(
        row["prompt"],
        "python",
        theme="monokai",
        line_numbers=True,
        word_wrap=True,
    )
    console.print(Panel(prompt_syntax, title="[section]Prompt[/section]", border_style="cyan"))

    test_syntax = Syntax(
        row["test"],
        "python",
        theme="monokai",
        line_numbers=True,
        word_wrap=True,
    )
    console.print(Panel(test_syntax, title="[section]Test Cases[/section]", border_style="magenta"))

    if canonical_visible:
        canon_syntax = Syntax(
            row["canonical_solution"],
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        console.print(
            Panel(canon_syntax, title="[warning]Canonical Solution (peek)[/warning]", border_style="yellow")
        )


def get_solution_inline(prefill: str) -> str | None:
    console.print("[hint]Type your solution below. When done, enter a blank line followed by [bold]END[/bold] and press Enter.[/hint]")
    console.print("[dim]─────────────────────────────────────────[/dim]")
    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
    except EOFError:
        pass
    console.print("[dim]─────────────────────────────────────────[/dim]")
    result = "\n".join(lines).strip()
    return result if result else None


def solution_preview(solution: str) -> Panel:
    return Panel(
        Syntax(solution, "python", theme="monokai", line_numbers=True, word_wrap=True),
        title="[success]Your Solution[/success]",
        border_style="green",
    )


def action_menu(console: Console) -> str:
    console.print(Rule(style="dim"))
    choice = Prompt.ask(
        "\n[bold]Action[/bold]",
        choices=["e", "s", "q", "p", "a"],
        default="e",
        show_choices=False,
        console=console,
    )
    return choice.lower()


def print_legend():
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("key", style="bold bright_white")
    table.add_column("action", style="dim")
    table.add_row("[e]", "Enter solution in terminal")
    table.add_row("[s]", "Skip this problem")
    table.add_row("[q]", "Quit and save progress")
    table.add_row("[p]", "Peek canonical solution")
    table.add_row("[a]", "Accept current solution as-is (reuse last)")
    console.print(Align.center(table))


def load_existing(output_path: str) -> dict[str, str]:
    p = Path(output_path)
    if not p.exists():
        return {}
    try:
        if output_path.endswith(".parquet"):
            df = pd.read_parquet(output_path)
        else:
            df = pd.read_csv(output_path)
        if "best_solution" in df.columns and "task_id" in df.columns:
            filled = df.dropna(subset=["best_solution"])
            return dict(zip(filled["task_id"], filled["best_solution"]))
    except Exception as e:
        console.print(f"[warning]Could not load existing file: {e}[/warning]")
    return {}


def save_dataset(rows: list[dict], output_path: str):
    df = pd.DataFrame(rows)
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False, engine="pyarrow")
    else:
        df.to_csv(output_path, index=False)
    console.print(f"\n[success]✓ Saved {len(df)} rows → {output_path}[/success]")

def main():
    parser = argparse.ArgumentParser(description="Interactive HumanEval dataset creator")
    parser.add_argument("--output", default="humaneval_best_solutions.parquet",
                        help="Output file (parquet or csv)")
    parser.add_argument("--start", type=int, default=None,
                        help="Task index to start from (0-based)")
    parser.add_argument("--format", choices=["parquet", "csv"], default=None,
                        help="Force output format (overrides extension detection)")
    args = parser.parse_args()

    output_path = args.output
    if args.format == "csv" and not output_path.endswith(".csv"):
        output_path = Path(output_path).stem + ".csv"
    elif args.format == "parquet" and not output_path.endswith(".parquet"):
        output_path = Path(output_path).stem + ".parquet"
    output_path = str(Path(output_path).resolve())

    clear()
    with console.status("[bold cyan]Loading HumanEval from HuggingFace...[/bold cyan]"):
        try:
            hf_dataset = load_dataset("openai/openai_humaneval", split="test")
        except Exception:
            hf_dataset = load_dataset("openai_humaneval", split="test")

    rows_hf = list(hf_dataset)
    total = len(rows_hf)

    console.print(f"[success]✓ Loaded {total} HumanEval problems[/success]")

    existing = load_existing(output_path)
    if existing:
        console.print(
            f"[hint]  Resuming: {len(existing)} problems already solved in {output_path}[/hint]"
        )

    rows: list[dict] = []
    for r in rows_hf:
        rows.append(
            {
                "task_id":           r["task_id"],
                "prompt":            r["prompt"],
                "entry_point":       r["entry_point"],
                "canonical_solution":r["canonical_solution"],
                "test":              r["test"],
                "best_solution":     existing.get(r["task_id"], None),
            }
        )

    if args.start is not None:
        start_idx = args.start
    else:
        start_idx = next(
            (i for i, r in enumerate(rows) if r["best_solution"] is None), total
        )

    if start_idx >= total:
        console.print("[success]All problems already have solutions! Dataset is complete.[/success]")
        save_dataset(rows, output_path)
        return

    console.print(f"[hint]  Starting from index {start_idx} (task {rows[start_idx]['task_id']})[/hint]")
    console.print()
    input("Press Enter to begin...")

    current_solution: str | None = None
    show_canonical = False
    i = start_idx

    while i < total:
        row = rows[i]
        done = sum(1 for r in rows if r["best_solution"] is not None)
        show_canonical = False

        clear()
        header_bar(done, total, output_path)
        print_legend()
        console.print()
        render_task(row, i, total, canonical_visible=show_canonical)

        if row["best_solution"] is not None:
            console.print(solution_preview(row["best_solution"]))
            console.print(
                "[hint]This problem is already solved. "
                "[bold]e[/bold]=redo · [bold]s[/bold]=keep & next · [bold]q[/bold]=quit[/hint]"
            )
            ch = Prompt.ask("Action", choices=["e", "s", "q"], default="s")
            if ch == "q":
                break
            elif ch == "s":
                i += 1
                continue

        while True:
            choice = action_menu(console)

            if choice == "q":
                # save and exit
                save_dataset(rows, output_path)
                console.print("\n[success]Progress saved. Goodbye![/success]")
                return

            elif choice == "s":
                console.print("[muted]Skipped.[/muted]")
                i += 1
                break

            elif choice == "p":
                clear()
                header_bar(done, total, output_path)
                print_legend()
                console.print()
                render_task(row, i, total, canonical_visible=True)
                if row["best_solution"]:
                    console.print(solution_preview(row["best_solution"]))

            elif choice == "a":
                if current_solution is None:
                    console.print("[warning]No solution in buffer yet. Use [bold]e[/bold] first.[/warning]")
                    continue
                rows[i]["best_solution"] = current_solution
                console.print(solution_preview(current_solution))
                console.print(f"[success]✓ Accepted buffer solution for {row['task_id']}[/success]")

                done_now = sum(1 for r in rows if r["best_solution"] is not None)
                if done_now % 5 == 0:
                    save_dataset(rows, output_path)
                i += 1
                break

            elif choice == "e":
                prefill = row["best_solution"] or ""
                console.print(f"\n[hint]Enter solution for [bold]{row['task_id']}[/bold] – {row['entry_point']}[/hint]\n")
                solution = get_solution_inline(prefill)

                if solution is None or solution.strip() == "":
                    console.print("[warning]Empty or cancelled – try again.[/warning]")
                    continue

                current_solution = solution
                clear()
                header_bar(done, total, output_path)
                console.print(solution_preview(solution))

                confirmed = Confirm.ask(
                    "\n[bold]Accept this solution?[/bold]", default=True
                )
                if confirmed:
                    rows[i]["best_solution"] = solution
                    console.print(f"[success]✓ Saved solution for {row['task_id']}[/success]")

                    done_now = sum(1 for r in rows if r["best_solution"] is not None)
                    if done_now % 5 == 0:
                        save_dataset(rows, output_path)
                    i += 1
                    break
                else:
                    console.print("[hint]Not accepted – you can edit again.[/hint]")

    clear()
    done = sum(1 for r in rows if r["best_solution"] is not None)
    console.print(
        Panel(
            f"[success]Session complete![/success]\n\n"
            f"  Problems solved : [bold]{done}[/bold] / {total}\n"
            f"  Output file     : [cyan]{output_path}[/cyan]",
            title="[bold]Summary[/bold]",
            border_style="green",
            expand=False,
        )
    )
    save_dataset(rows, output_path)


if __name__ == "__main__":
    main()
