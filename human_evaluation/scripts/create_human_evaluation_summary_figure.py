#!/usr/bin/env python3
"""
Emit LaTeX/PGFPlots code for the human-evaluation summary forest plot
(figure_human_evaluation_summary.tex), using the same scores and bootstrap
intervals as create_results_table.py.
"""

import argparse
import os
import sys
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.bootstrap import bootstrap_confidence_interval
from metrics.dataloader import add_pairwise_dataset_cli_arguments, load_metric_dataset
from metrics.sqm import calculate_sqm_scores


# Row label centres (ytick positions); fluency/accuracy markers sit ±0.14 apart.
ROW_CENTRE_Y_VALUES = (1.0, 2.0, 3.0, 3.75)
VERTICAL_JITTER = 0.14

# Same systems and order as create_results_table.py / paper figure labels.
SYSTEM_ORDER = (
    "reference",
    "Gemini-3-Pro",
    "romansh-nllb-1.3b",
    "romansh-nllb-1.3b-dict-prompting",
)

FIGURE_TEMPLATE = r"""% Axis width = column minus label column (text width + gap). TikZ font=\footnotesize; do not wrap in \resizebox.
\definecolor{forestplotbackground}{RGB}{245,245,245}% HTML whitesmoke #F5F5F5
\begin{tikzpicture}[font=\footnotesize]
\begin{axis}[
    width=\dimexpr\columnwidth-29mm\relax,
    height=4.8cm,
    scale only axis,
    xmin=-1.0, xmax=0.75,
    ymin=0.5, ymax=4.25,
    xtick={-1.0,-0.5,0,0.5},
    xlabel={$z$-normalized human quality rating},
    xlabel style={yshift=2pt},
    % Fourth row centre at $y=3.75$ (between tight $3.5$ and uniform $4$).
    ytick={1,2,3,3.75},
    yticklabels={
      {\mbox{Human reference}},
      {Gemini 3 Pro \mbox{(preview)}},
      {LR$\rightarrow$HR augmented NLLB},
      {\begin{minipage}[t]{2.4cm}\raggedright\leftskip=1em\relax\noindent + Dictionary prompting\end{minipage}},
    },
    yticklabel style={
      align=left,
      anchor=east,
      text width=2.45cm,
    },
    y dir=reverse,
    axis y line=left,
    axis x line=bottom,
    tick align=outside,
    tick pos=left,
    enlarge y limits=false,
    axis background/.append style={fill=forestplotbackground},
    xmajorgrids,
    grid style={dashed, gray!30},
    legend style={
        legend columns=1,
        font=\footnotesize,
        draw=black!55,
        line width=0.4pt,
        fill=white,
        fill opacity=1,
        draw opacity=1,
        inner xsep=4pt,
        inner ysep=2pt,
        outer sep=0pt,
        column sep=2pt,
        row sep=1pt,
        cells={anchor=west},
        legend cell align=left,
        every legend image column/.append style={column sep=2pt},
        at={(rel axis cs:0.06,1.03)},
        anchor=north west,
    },
    legend image post style={scale=0.9},
    clip=false,
]

% Midlines between Human--Gemini and Gemini--NLLB; extend left into label band (shift matches axis width reserve).
\draw[gray!65, line width=0.55pt] ([xshift=-29mm]axis cs:-1.0,1.5) -- (axis cs:0.75,1.5);
\draw[gray!65, line width=0.55pt] ([xshift=-29mm]axis cs:-1.0,2.5) -- (axis cs:0.75,2.5);

\addplot+[
    only marks,
    mark=*, mark size=2.2pt,
    color=orange!85!black,
    error bars/.cd, x dir=both, x explicit,
    error bar style={line width=0.7pt, gray!55},
    error mark options={rotate=90, mark size=3pt, line width=0.7pt, draw=gray!55, fill=gray!55},
] table[x=x, y=y, x error minus=em, x error plus=ep, col sep=comma] {
    x,     em,   ep,   y
{fluency_rows}
};
\addlegendentry{Fluency}

\addplot+[
    only marks,
    mark=square*, mark size=2pt,
    color=blue!70!black,
    error bars/.cd, x dir=both, x explicit,
    error bar style={line width=0.7pt, gray!55},
    error mark options={rotate=90, mark size=3pt, line width=0.7pt, draw=gray!55, fill=gray!55},
] table[x=x, y=y, x error minus=em, x error plus=ep, col sep=comma] {
    x,     em,   ep,   y
{accuracy_rows}
};
\addlegendentry{Accuracy}

\end{axis}
\end{tikzpicture}
"""


def confidence_error_distances(
    score: float,
    confidence_interval: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if confidence_interval is None:
        return None
    lower, upper = confidence_interval
    return score - lower, upper - score


def format_plot_table_rows(
    coordinates: list[tuple[float, float, float, float]],
) -> str:
    """
    Format lines ``    x,  em,  ep,  y`` with spacing similar to the paper mock-up.
    """
    lines = []
    for mean_score, error_minus, error_plus, plot_y in coordinates:
        mean_text = f"{mean_score:.2f}"
        leading_spaces = "   " if mean_text.startswith("-") else "    "
        lines.append(
            f"{leading_spaces}{mean_text},  {error_minus:.2f}, {error_plus:.2f}, {plot_y:.2f}"
        )
    return "\n".join(lines)


def collect_coordinates(
    scores: dict[str, float | None],
    confidence_intervals: dict[str, tuple[float, float] | None],
    *,
    vertical_offsets: tuple[float, ...],
    jitter_down: bool,
) -> list[tuple[float, float, float, float]]:
    coordinates: list[tuple[float, float, float, float]] = []
    for row_index, system_name in enumerate(SYSTEM_ORDER):
        score = scores.get(system_name)
        interval = confidence_intervals.get(system_name)

        centre_y = vertical_offsets[row_index]
        plot_y = centre_y - VERTICAL_JITTER if jitter_down else centre_y + VERTICAL_JITTER

        if score is None:
            print(
                f"Warning: missing score for system {system_name!r}; "
                "omitting point from figure data.",
                file=sys.stderr,
            )
            continue
        error_distances = confidence_error_distances(score, interval)
        if error_distances is None:
            print(
                f"Warning: missing bootstrap CI for system {system_name!r}; "
                "using error bar distances 0.0.",
                file=sys.stderr,
            )
            error_distances = (0.0, 0.0)

        error_minus, error_plus = error_distances
        coordinates.append((score, error_minus, error_plus, plot_y))
    return coordinates


def write_output(content: str, output_path: Path) -> None:
    if output_path.parent.exists():
        with output_path.open("w", encoding="utf-8") as output_file:
            output_file.write(content)
        print(f"\nFigure LaTeX written to {output_path}", file=sys.stderr)
    else:
        print(
            f"\nWarning: Output directory does not exist: {output_path.parent}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_pairwise_dataset_cli_arguments(parser)
    args = parser.parse_args()

    print("Loading datasets...", file=sys.stderr)
    segment_fluency = load_metric_dataset(
        args.dataset, "segment_fluency", hub_revision=args.revision
    )
    document_accuracy = load_metric_dataset(
        args.dataset, "document_accuracy", hub_revision=args.revision
    )

    print("Calculating fluency SQM scores...", file=sys.stderr)
    fluency_sqm_scores = calculate_sqm_scores(segment_fluency)

    print("Calculating accuracy SQM scores...", file=sys.stderr)
    accuracy_sqm_scores = calculate_sqm_scores(document_accuracy)

    print(
        "Calculating bootstrap CIs for fluency SQM (segment-level)...",
        file=sys.stderr,
    )
    fluency_sqm_confidence_intervals = bootstrap_confidence_interval(
        segment_fluency,
        calculate_sqm_scores,
        resampling_unit="segment",
        n_resamples=1000,
        random_seed=42,
    )

    print(
        "Calculating bootstrap CIs for accuracy SQM (document-level)...",
        file=sys.stderr,
    )
    accuracy_sqm_confidence_intervals = bootstrap_confidence_interval(
        document_accuracy,
        calculate_sqm_scores,
        resampling_unit="document",
        n_resamples=1000,
        random_seed=42,
    )

    fluency_rows = format_plot_table_rows(
        collect_coordinates(
            fluency_sqm_scores,
            fluency_sqm_confidence_intervals,
            vertical_offsets=ROW_CENTRE_Y_VALUES,
            jitter_down=True,
        )
    )
    accuracy_rows = format_plot_table_rows(
        collect_coordinates(
            accuracy_sqm_scores,
            accuracy_sqm_confidence_intervals,
            vertical_offsets=ROW_CENTRE_Y_VALUES,
            jitter_down=False,
        )
    )

    figure_body = (
        FIGURE_TEMPLATE.replace("{fluency_rows}", fluency_rows).replace(
            "{accuracy_rows}", accuracy_rows
        )
    )

    print(figure_body)

    dotenv.load_dotenv()
    paper_directory = os.getenv("PAPER_DIR")
    if paper_directory is not None:
        output_path = Path(paper_directory) / "include" / "figure_human_evaluation_summary.tex"
        write_output(figure_body, output_path)


if __name__ == "__main__":
    main()
