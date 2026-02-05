"""
Automated Backtesting Module

Generates nightly reports comparing HELIOS predictions vs actual temps
with virtual P&L calculations.
"""
from datetime import date, datetime, timedelta
from typing import List, Tuple, Dict, Optional
import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).parent.parent / "helios_weather.db"


def get_verified_predictions(station_id: str, days: int = 7) -> List[Dict]:
    """Get verified predictions for the last N days."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT *
            FROM predictions
            WHERE station_id = ?
              AND verified_at IS NOT NULL
              AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (station_id, cutoff_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    except:
        return []


def calculate_virtual_pnl(
    prediction_f: float,
    actual_f: float,
    bet_amount: float = 10.0
) -> Tuple[float, str]:
    """
    Simulate a $10 bet on the closest range to our prediction.
    
    Simplified model: If within ±2°F, win $8. Otherwise lose $10.
    
    Returns:
        (pnl, outcome_str)
    """
    error = abs(prediction_f - actual_f)
    
    if error <= 1.0:
        # Very accurate, good payout
        return 8.0, f"WIN (+$8, error={error:.1f}°F)"
    elif error <= 2.0:
        # Decent, small win
        return 2.0, f"WIN (+$2, error={error:.1f}°F)"
    else:
        # Miss, lose bet
        return -bet_amount, f"LOSS (-${bet_amount}, error={error:.1f}°F)"


def generate_daily_report(station_id: str, days: int = 7) -> str:
    """
    Generate comprehensive daily report.
    
    Shows:
    - Prediction vs Actual comparison
    - HRRR error vs HELIOS error
    - Virtual P&L
    """
    predictions = get_verified_predictions(station_id, days)
    
    if not predictions:
        return f"No verified predictions for {station_id} in last {days} days"
    
    lines = []
    lines.append("=" * 75)
    lines.append(f"  REPORTE DE BACKTESTING - {station_id}")
    lines.append(f"  Últimos {days} días")
    lines.append("=" * 75)
    lines.append("")
    
    lines.append("┌" + "─" * 11 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┬" + "─" * 12 + "┬" + "─" * 16 + "┐")
    lines.append("│    Día    │  Real    │  HRRR    │  Helios  │ Error HRRR │   P&L Virtual  │")
    lines.append("├" + "─" * 11 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼" + "─" * 12 + "┼" + "─" * 16 + "┤")
    
    total_hrrr_error = 0.0
    total_helios_error = 0.0
    total_pnl = 0.0
    count = 0
    
    for pred in predictions:
        try:
            day = datetime.fromisoformat(pred['timestamp']).strftime('%Y-%m-%d')
            actual = pred.get('real_max_verified_f', 0)
            hrrr = pred.get('hrrr_max_raw_f', 0)
            helios = pred.get('final_prediction_f', 0)
            
            if actual == 0 or hrrr == 0 or helios == 0:
                continue
            
            hrrr_error = abs(hrrr - actual)
            helios_error = abs(helios - actual)
            
            pnl, outcome = calculate_virtual_pnl(helios, actual)
            
            total_hrrr_error += hrrr_error
            total_helios_error += helios_error
            total_pnl += pnl
            count += 1
            
            pnl_str = f"+${pnl:.1f}" if pnl > 0 else f"-${abs(pnl):.1f}"
            
            lines.append(f"│ {day} │ {actual:6.1f}°F │ {hrrr:6.1f}°F │ {helios:6.1f}°F │  {hrrr_error:+5.1f}°F  │  {pnl_str:12s}  │")
        except:
            continue
    
    lines.append("└" + "─" * 11 + "┴" + "─" * 10 + "┴" + "─" * 10 + "┴" + "─" * 10 + "┴" + "─" * 12 + "┴" + "─" * 16 + "┘")
    
    if count > 0:
        avg_hrrr = total_hrrr_error / count
        avg_helios = total_helios_error / count
        improvement = avg_hrrr - avg_helios
        roi = (total_pnl / (count * 10)) * 100
        
        lines.append("")
        lines.append(f"Promedio Error HRRR:   {avg_hrrr:.2f}°F")
        lines.append(f"Promedio Error Helios: {avg_helios:.2f}°F")
        
        if improvement > 0:
            lines.append(f"Mejora:                +{improvement:.2f}°F ✅")
        else:
            lines.append(f"Mejora:                {improvement:.2f}°F ⚠️")
        
        lines.append("")
        lines.append(f"P&L Total:             ${total_pnl:+.2f}")
        lines.append(f"ROI:                   {roi:+.1f}%")
    
    lines.append("")
    lines.append("=" * 75)
    
    return "\n".join(lines)


def run_nightly_report():
    """Generate report for all stations."""
    from config import STATIONS
    
    print("\n" + "=" * 75)
    print("  REPORTE NOCTURNO DE BACKTESTING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 75 + "\n")
    
    for station_id in STATIONS.keys():
        report = generate_daily_report(station_id, days=7)
        print(report)
        print()


if __name__ == "__main__":
    # Test report generation
    run_nightly_report()
