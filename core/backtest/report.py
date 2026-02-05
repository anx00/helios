"""
Report Generator Module

Generates HTML and Markdown reports for backtest results:
- Summary statistics
- Calibration curves
- PnL charts
- Day-by-day breakdown
- Regime analysis
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Any
from pathlib import Path
import json

from .engine import BacktestResult, DayResult
from .metrics import CalibrationMetrics
from .calibration import CalibrationResult

logger = logging.getLogger("backtest.report")


class ReportGenerator:
    """
    Generates reports from backtest results.
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html(
        self,
        result: BacktestResult,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate HTML report.

        Args:
            result: BacktestResult to report
            filename: Output filename (optional)

        Returns:
            Path to generated file
        """
        if filename is None:
            filename = (
                f"report_{result.station_id}_"
                f"{result.start_date.isoformat()}_"
                f"{result.end_date.isoformat()}.html"
            )

        html = self._render_html(result)

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Generated HTML report: {filepath}")
        return str(filepath)

    def generate_markdown(
        self,
        result: BacktestResult,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate Markdown report.

        Args:
            result: BacktestResult to report
            filename: Output filename (optional)

        Returns:
            Path to generated file
        """
        if filename is None:
            filename = (
                f"report_{result.station_id}_"
                f"{result.start_date.isoformat()}_"
                f"{result.end_date.isoformat()}.md"
            )

        md = self._render_markdown(result)

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"Generated Markdown report: {filepath}")
        return str(filepath)

    def _render_html(self, result: BacktestResult) -> str:
        """Render HTML report."""
        metrics = result.aggregated_metrics

        # Build reliability chart data
        reliability_json = "[]"
        if metrics and metrics.reliability_data:
            reliability_json = json.dumps(metrics.reliability_data)

        # Build PnL series
        pnl_series = []
        cumulative = 0.0
        for day in result.day_results:
            cumulative += day.pnl_net
            pnl_series.append({
                "date": day.market_date.isoformat(),
                "pnl": day.pnl_net,
                "cumulative": cumulative
            })
        pnl_json = json.dumps(pnl_series)

        # Day-by-day table rows
        day_rows = ""
        for day in result.day_results:
            status = "correct" if day.predicted_winner == day.actual_winner else "wrong"
            day_rows += f"""
            <tr class="{status}">
                <td>{day.market_date.isoformat()}</td>
                <td>{day.predicted_winner or '-'}</td>
                <td>{day.actual_winner or '-'}</td>
                <td>{day.predicted_tmax or 0:.1f}°F</td>
                <td>{day.actual_tmax or 0:.1f}°F</td>
                <td>${day.pnl_net:.2f}</td>
                <td>{day.nowcast_count}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HELIOS Backtest Report - {result.station_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --text-main: #e5e7eb;
            --text-muted: #9ca3af;
            --accent-blue: #0078ff;
            --accent-green: #10b981;
            --accent-red: #ef4444;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-main);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: var(--bg-secondary);
            border-radius: 12px;
        }}

        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: var(--text-muted);
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}

        .stat-card .label {{
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-card .value {{
            font-size: 28px;
            font-weight: 700;
            margin-top: 5px;
        }}

        .stat-card .value.green {{ color: var(--accent-green); }}
        .stat-card .value.red {{ color: var(--accent-red); }}
        .stat-card .value.blue {{ color: var(--accent-blue); }}

        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .section h2 {{
            font-size: 18px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .chart-container {{
            position: relative;
            height: 300px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}

        th {{
            color: var(--text-muted);
            font-weight: 500;
            text-transform: uppercase;
            font-size: 11px;
        }}

        tr.correct {{
            background: rgba(16, 185, 129, 0.1);
        }}

        tr.wrong {{
            background: rgba(239, 68, 68, 0.1);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}

        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>HELIOS Backtest Report</h1>
            <div class="subtitle">
                {result.station_id} | {result.start_date.isoformat()} to {result.end_date.isoformat()} |
                Mode: {result.mode.value} | Policy: {result.policy_name}
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Total PnL</div>
                <div class="value {'green' if result.total_pnl_net >= 0 else 'red'}">
                    ${result.total_pnl_net:.2f}
                </div>
            </div>
            <div class="stat-card">
                <div class="label">Win Rate</div>
                <div class="value blue">{result.win_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">{result.sharpe_ratio:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="label">Max Drawdown</div>
                <div class="value red">{result.max_drawdown:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">Total Fills</div>
                <div class="value">{result.total_fills}</div>
            </div>
            <div class="stat-card">
                <div class="label">Days Tested</div>
                <div class="value">{result.days_with_data}</div>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="section">
                <h2>Calibration Metrics</h2>
                <table>
                    <tr>
                        <td>Brier Score</td>
                        <td><strong>{metrics.brier_global:.4f if metrics else 'N/A'}</strong></td>
                    </tr>
                    <tr>
                        <td>Log Loss</td>
                        <td><strong>{metrics.log_loss_global:.4f if metrics else 'N/A'}</strong></td>
                    </tr>
                    <tr>
                        <td>ECE</td>
                        <td><strong>{metrics.ece:.4f if metrics else 'N/A'}</strong></td>
                    </tr>
                    <tr>
                        <td>Sharpness</td>
                        <td><strong>{metrics.sharpness:.4f if metrics else 'N/A'}</strong></td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2>Point Prediction Metrics</h2>
                <table>
                    <tr>
                        <td>Tmax MAE</td>
                        <td><strong>{metrics.tmax_mae:.2f}°F</strong></td>
                    </tr>
                    <tr>
                        <td>Tmax RMSE</td>
                        <td><strong>{metrics.tmax_rmse:.2f}°F</strong></td>
                    </tr>
                    <tr>
                        <td>Tmax Bias</td>
                        <td><strong>{metrics.tmax_bias:+.2f}°F</strong></td>
                    </tr>
                    <tr>
                        <td>Avg Churn</td>
                        <td><strong>{metrics.avg_churn:.4f if metrics else 'N/A'}</strong></td>
                    </tr>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>Cumulative PnL</h2>
            <div class="chart-container">
                <canvas id="pnlChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>Reliability Diagram</h2>
            <div class="chart-container">
                <canvas id="reliabilityChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>Day-by-Day Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Predicted</th>
                        <th>Actual</th>
                        <th>Pred Tmax</th>
                        <th>Actual Tmax</th>
                        <th>PnL</th>
                        <th>Nowcasts</th>
                    </tr>
                </thead>
                <tbody>
                    {day_rows}
                </tbody>
            </table>
        </div>

        <div class="section" style="text-align: center; color: var(--text-muted);">
            Generated by HELIOS Phase 5 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>

    <script>
        // PnL Chart
        const pnlData = {pnl_json};
        new Chart(document.getElementById('pnlChart'), {{
            type: 'line',
            data: {{
                labels: pnlData.map(d => d.date),
                datasets: [{{
                    label: 'Cumulative PnL',
                    data: pnlData.map(d => d.cumulative),
                    borderColor: '#0078ff',
                    backgroundColor: 'rgba(0,120,255,0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af', maxRotation: 45 }}
                    }}
                }}
            }}
        }});

        // Reliability Chart
        const relData = {reliability_json};
        if (relData.bin_centers) {{
            new Chart(document.getElementById('reliabilityChart'), {{
                type: 'scatter',
                data: {{
                    datasets: [
                        {{
                            label: 'Actual Frequency',
                            data: relData.bin_centers.map((x, i) => ({{
                                x: x,
                                y: relData.actual_freqs[i]
                            }})).filter(d => d.y !== null),
                            backgroundColor: '#0078ff',
                            pointRadius: 6
                        }},
                        {{
                            label: 'Perfect Calibration',
                            data: [{{x: 0, y: 0}}, {{x: 1, y: 1}}],
                            type: 'line',
                            borderColor: 'rgba(255,255,255,0.3)',
                            borderDash: [5, 5],
                            pointRadius: 0
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            min: 0, max: 1,
                            title: {{ display: true, text: 'Predicted Probability', color: '#9ca3af' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }},
                            ticks: {{ color: '#9ca3af' }}
                        }},
                        y: {{
                            min: 0, max: 1,
                            title: {{ display: true, text: 'Observed Frequency', color: '#9ca3af' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }},
                            ticks: {{ color: '#9ca3af' }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""
        return html

    def _render_markdown(self, result: BacktestResult) -> str:
        """Render Markdown report."""
        metrics = result.aggregated_metrics

        # Day-by-day table
        day_table = "| Date | Predicted | Actual | Pred Tmax | Actual Tmax | PnL |\n"
        day_table += "|------|-----------|--------|-----------|-------------|-----|\n"

        for day in result.day_results:
            status = "✓" if day.predicted_winner == day.actual_winner else "✗"
            day_table += (
                f"| {day.market_date} | {day.predicted_winner or '-'} | "
                f"{day.actual_winner or '-'} | {day.predicted_tmax or 0:.1f}°F | "
                f"{day.actual_tmax or 0:.1f}°F | ${day.pnl_net:.2f} {status} |\n"
            )

        md = f"""# HELIOS Backtest Report

**Station:** {result.station_id}
**Period:** {result.start_date} to {result.end_date}
**Mode:** {result.mode.value}
**Policy:** {result.policy_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total PnL (Net) | **${result.total_pnl_net:.2f}** |
| Win Rate | {result.win_rate:.1%} |
| Sharpe Ratio | {result.sharpe_ratio:.2f} |
| Max Drawdown | {result.max_drawdown:.1%} |
| Total Fills | {result.total_fills} |
| Days Tested | {result.days_with_data} |

---

## Calibration Metrics

| Metric | Value |
|--------|-------|
| Brier Score | {metrics.brier_global:.4f if metrics else 'N/A'} |
| Log Loss | {metrics.log_loss_global:.4f if metrics else 'N/A'} |
| ECE | {metrics.ece:.4f if metrics else 'N/A'} |
| Sharpness | {metrics.sharpness:.4f if metrics else 'N/A'} |

---

## Point Prediction Metrics

| Metric | Value |
|--------|-------|
| Tmax MAE | {metrics.tmax_mae:.2f}°F |
| Tmax RMSE | {metrics.tmax_rmse:.2f}°F |
| Tmax Bias | {metrics.tmax_bias:+.2f}°F |
| Avg Churn | {metrics.avg_churn:.4f if metrics else 'N/A'} |
| Avg Flips | {metrics.avg_flips:.1f if metrics else 'N/A'} |

---

## Day-by-Day Results

{day_table}

---

*Report generated by HELIOS Phase 5 Backtesting Module*
"""
        return md

    def generate_calibration_report(
        self,
        result: CalibrationResult,
        filename: Optional[str] = None
    ) -> str:
        """Generate report for calibration results."""
        if filename is None:
            filename = (
                f"calibration_report_{result.station_id}_"
                f"{result.train_start.isoformat()}.md"
            )

        # Sort results by validation score
        sorted_results = sorted(
            result.all_results,
            key=lambda x: x.get("val_score", 0),
            reverse=True
        )

        # Top 10 table
        top_table = "| Rank | Parameters | Train Score | Val Score |\n"
        top_table += "|------|------------|-------------|------------|\n"

        for i, r in enumerate(sorted_results[:10], 1):
            params_str = ", ".join(f"{k}={v:.3f}" for k, v in r["params"].items())
            top_table += (
                f"| {i} | {params_str} | "
                f"{r['train_score']:.4f} | {r['val_score']:.4f} |\n"
            )

        md = f"""# HELIOS Calibration Report

**Station:** {result.station_id}
**Training Period:** {result.train_start} to {result.train_end}
**Validation Period:** {result.val_start} to {result.val_end}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Best Parameters

```json
{json.dumps(result.best_params, indent=2)}
```

**Training Score:** {result.best_train_score:.4f}
**Validation Score:** {result.best_val_score:.4f}

---

## Calibration Summary

| Metric | Value |
|--------|-------|
| Total Combinations | {result.total_combinations} |
| Combinations Tested | {result.combinations_tested} |

---

## Top 10 Parameter Sets

{top_table}

---

*Report generated by HELIOS Phase 5 Calibration Module*
"""

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"Generated calibration report: {filepath}")
        return str(filepath)
