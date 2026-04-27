from pathlib import Path
import base64

import nbformat
import pandas as pd
import plotly.io as pio
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "HAR_model_static_export.ipynb"


FIGURES = [
    ("realized_vol", "Daily Realized Volatility"),
    ("oil_drivers", "Oil Drivers Overview"),
    ("correlation", "Monthly Correlation Structure"),
    ("iv_scatter", "IV vs Next-Day Log Realized Volatility"),
    ("ovx_scatter", "OVX vs Next-Month Log Realized Volatility"),
    ("tosi_scatter", "TOSI vs Next-Month Log Realized Volatility"),
    ("ols_sharpe", "OLS Sharpe by Feature Specification"),
    ("iv_lift", "IV Lift Comparison"),
    ("dm_heatmap", "Diebold-Mariano Significance"),
    ("jets_forecast", "JETS Forecast vs Actual"),
    ("jets_pnl", "JETS Cumulative Straddle P&L"),
    ("feature_importance_jets", "Tree-Based Feature Importance for JETS"),
]


def build_notebook():
    exec_count = 1

    def next_count():
        nonlocal exec_count
        cur = exec_count
        exec_count += 1
        return cur

    def html_output(html):
        return [new_output(output_type="display_data", data={"text/html": html, "text/plain": html})]

    def figure_html_output(fig_name, include_plotlyjs):
        fig = pio.from_json((ROOT / "data" / "processed" / "figures" / f"{fig_name}.json").read_text(encoding="utf-8"))
        fig.update_layout(width=1150)
        html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            config={"displayModeBar": False, "responsive": True},
        )
        return [new_output(output_type="display_data", data={"text/html": html, "text/plain": f"<Figure: {fig_name}>"} )]

    summary = pd.read_parquet(ROOT / "data" / "processed" / "summary.parquet")
    best_models = pd.read_parquet(ROOT / "data" / "processed" / "best_models.parquet")

    cells = [
        new_markdown_cell(
            "# HAR Model Static Export\n\n"
            "This companion notebook is for **PDF submission**. It leaves `HAR_model.ipynb` unchanged "
            "and renders the saved Plotly figures as static images so they survive notebook-to-PDF export."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import plotly.io as pio\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "ROOT = Path.cwd()\n"
            "FIG_DIR = ROOT / 'data' / 'processed' / 'figures'\n\n"
            "def load_figure(name):\n"
            "    return pio.from_json((FIG_DIR / f'{name}.json').read_text(encoding='utf-8'))\n\n"
            "def show_static_figure(name, width=1400, height=None, scale=2):\n"
            "    fig = load_figure(name)\n"
            "    if width is not None:\n"
            "        fig.update_layout(width=width)\n"
            "    if height is not None:\n"
            "        fig.update_layout(height=height)\n"
            "    png = fig.to_image(format='png', scale=scale)\n"
            "    display(Image(data=png))\n"
        ,
            execution_count=next_count(),
        ),
        new_markdown_cell("## Summary Tables"),
        new_code_cell(
            "summary = pd.read_parquet(ROOT / 'data' / 'processed' / 'summary.parquet')\n"
            "best_models = pd.read_parquet(ROOT / 'data' / 'processed' / 'best_models.parquet')\n\n"
            "display(Markdown('### Best Model per Ticker'))\n"
            "display(best_models[['Ticker', 'Feature_Spec', 'Model_Family', 'RMSE', 'Sharpe_Straddle', 'Directional_Acc']])\n\n"
            "display(Markdown('### Average Metrics by Feature Specification and Model Family'))\n"
            "display(summary)\n"
        ,
            execution_count=next_count(),
            outputs=html_output(
                "<h3>Best Model per Ticker</h3>"
                + best_models[
                    ["Ticker", "Feature_Spec", "Model_Family", "RMSE", "Sharpe_Straddle", "Directional_Acc"]
                ].to_html(index=False)
                + "<h3>Average Metrics by Feature Specification and Model Family</h3>"
                + summary.to_html(index=False)
            ),
        ),
    ]

    for idx, (fig_name, title) in enumerate(FIGURES):
        cells.append(new_markdown_cell(f"## {title}"))
        cells.append(
            new_code_cell(
                f"show_static_figure('{fig_name}')",
                execution_count=next_count(),
                outputs=figure_html_output(fig_name, include_plotlyjs=True if idx == 0 else False),
            )
        )

    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
    )
    return nb


def main():
    nb = build_notebook()
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote {NOTEBOOK_PATH.name}")


if __name__ == "__main__":
    main()
