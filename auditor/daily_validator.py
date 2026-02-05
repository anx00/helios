"""
Helios Weather Lab - Daily Validator (Protocol FORTRESS)
Comparative verification: AI vs HRRR vs GFS with winner determination.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from database import (
    get_unverified_predictions,
    update_verification,
    get_accuracy_report,
)
from collector.metar_fetcher import fetch_metar_history
from config import STATIONS


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round((celsius * 9/5) + 32, 1)


def determine_winner(ai_error: float, hrrr_error: float, gfs_error: Optional[float]) -> str:
    """
    Determine which prediction was most accurate.
    
    Returns 'AI', 'HRRR', or 'GFS'
    """
    errors = {"AI": abs(ai_error) if ai_error is not None else float('inf')}
    errors["HRRR"] = abs(hrrr_error) if hrrr_error is not None else float('inf')
    
    if gfs_error is not None:
        errors["GFS"] = abs(gfs_error)
    
    return min(errors, key=errors.get)


async def get_actual_max_temperature(station_id: str, date: datetime) -> Optional[Dict[str, float]]:
    """Get the actual maximum temperature for a station on a specific date."""
    try:
        observations = await fetch_metar_history(station_id, hours=24)
        
        if not observations:
            print(f"⚠ No historical data for {station_id} on {date.date()}")
            return None
        
        target_date = date.date()
        day_observations = [
            obs for obs in observations
            if obs.observation_time.date() == target_date
            and obs.temp_c is not None
        ]
        
        if not day_observations:
            print(f"⚠ No observations for {station_id} on {target_date}")
            return None
        
        max_temp_c = max(obs.temp_c for obs in day_observations)
        
        return {
            "max_c": max_temp_c,
            "max_f": celsius_to_fahrenheit(max_temp_c),
            "observation_count": len(day_observations),
        }
        
    except Exception as e:
        print(f"✗ Error getting actual max for {station_id}: {e}")
        return None


async def verify_station_predictions(station_id: str, date: datetime) -> Dict[str, Any]:
    """
    Verify all predictions for a station (FORTRESS: comparative analysis).
    """
    predictions = get_unverified_predictions(station_id, date)
    
    if not predictions:
        return {
            "station_id": station_id,
            "date": date.date().isoformat(),
            "status": "no_predictions",
            "verified_count": 0,
        }
    
    actual_max = await get_actual_max_temperature(station_id, date)
    
    if not actual_max:
        return {
            "station_id": station_id,
            "date": date.date().isoformat(),
            "status": "no_actual_data",
            "predictions_pending": len(predictions),
        }
    
    verified_count = 0
    ai_errors = []
    hrrr_errors = []
    gfs_errors = []
    winners = {"AI": 0, "HRRR": 0, "GFS": 0}
    
    for pred in predictions:
        ai_pred_f = pred.get("ai_prediction_f")
        hrrr_pred_f = pred.get("hrrr_max_forecast_f")
        gfs_pred_f = pred.get("gfs_max_forecast_f")
        
        # Calculate errors
        ai_error = (ai_pred_f - actual_max["max_f"]) if ai_pred_f else None
        hrrr_error = (hrrr_pred_f - actual_max["max_f"]) if hrrr_pred_f else None
        gfs_error = (gfs_pred_f - actual_max["max_f"]) if gfs_pred_f else None
        
        # MODULE D: Rule 2 - Calculate Alpha Generated
        # Alpha = Error_Modelo - Error_Helios
        alpha = None
        if ai_error is not None and hrrr_error is not None:
            alpha = abs(hrrr_error) - abs(ai_error)
        
        # Determine winner for this prediction
        winner = determine_winner(ai_error, hrrr_error, gfs_error)
        winners[winner] = winners.get(winner, 0) + 1
        
        # Update database
        update_verification(
            prediction_id=pred["id"],
            actual_max_c=actual_max["max_c"],
            actual_max_f=actual_max["max_f"],
            ai_error_f=ai_error,
            hrrr_error_f=hrrr_error,
            gfs_error_f=gfs_error,
            daily_alpha=alpha,
            daily_winner=winner,
        )
        
        verified_count += 1
        
        if ai_error is not None:
            ai_errors.append(abs(ai_error))
        if hrrr_error is not None:
            hrrr_errors.append(abs(hrrr_error))
        if gfs_error is not None:
            gfs_errors.append(abs(gfs_error))
    
    # Determine overall daily winner
    daily_winner = max(winners, key=winners.get)
    
    return {
        "station_id": station_id,
        "date": date.date().isoformat(),
        "status": "verified",
        "verified_count": verified_count,
        "actual_max_f": actual_max["max_f"],
        "avg_ai_error": sum(ai_errors) / len(ai_errors) if ai_errors else None,
        "avg_hrrr_error": sum(hrrr_errors) / len(hrrr_errors) if hrrr_errors else None,
        "avg_gfs_error": sum(gfs_errors) / len(gfs_errors) if gfs_errors else None,
        "daily_winner": daily_winner,
        "winner_counts": winners,
    }


async def run_daily_audit(target_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Run the FORTRESS daily audit for all stations.
    Comparative analysis: AI vs HRRR vs GFS.
    """
    if target_date is None:
        target_date = datetime.now() - timedelta(days=1)
    
    print(f"\n{'#'*65}")
    print(f"🏰 FORTRESS DAILY AUDIT: {target_date.date()}")
    print(f"{'#'*65}\n")
    
    results = {}
    overall_winners = {"AI": 0, "HRRR": 0, "GFS": 0}
    
    for station_id in STATIONS.keys():
        print(f"Auditing {station_id}...")
        result = await verify_station_predictions(station_id, target_date)
        results[station_id] = result
        
        if result["status"] == "verified":
            print(f"  ✓ Verified {result['verified_count']} predictions")
            print(f"    Actual Max: {result['actual_max_f']}°F")
            
            if result['avg_ai_error']:
                print(f"    AI Error:   ±{result['avg_ai_error']:.1f}°F")
            if result['avg_hrrr_error']:
                print(f"    HRRR Error: ±{result['avg_hrrr_error']:.1f}°F")
            if result['avg_gfs_error']:
                print(f"    GFS Error:  ±{result['avg_gfs_error']:.1f}°F")
            
            winner = result.get('daily_winner', 'N/A')
            print(f"    🏆 Station Winner: {winner}")
            
            if winner in overall_winners:
                overall_winners[winner] += 1
        else:
            print(f"  ⚠ Status: {result['status']}")
    
    # Overall winner
    if sum(overall_winners.values()) > 0:
        overall_winner = max(overall_winners, key=overall_winners.get)
        print(f"\n{'='*65}")
        print(f"🏆 GANADOR DEL DÍA: {overall_winner}")
        print(f"   Puntuación: AI={overall_winners['AI']}, HRRR={overall_winners['HRRR']}, GFS={overall_winners['GFS']}")
        print(f"{'='*65}\n")
    
    print(f"{'#'*65}\n")
    
    return results


def generate_accuracy_report(days: int = 7) -> str:
    """
    Generate FORTRESS accuracy report comparing AI vs HRRR vs GFS.
    """
    report_data = get_accuracy_report(days)
    
    lines = [
        "",
        "=" * 65,
        f"🏰 FORTRESS ACCURACY REPORT (Last {days} days)",
        "=" * 65,
        "",
    ]
    
    if not report_data:
        lines.append("No verified predictions available yet.")
    else:
        total_ai_wins = 0
        total_hrrr_wins = 0
        total_gfs_wins = 0
        
        for station_id, stats in report_data.items():
            station = STATIONS.get(station_id)
            station_name = station.name if station else station_id
            
            lines.append(f"📍 {station_name} ({station_id})")
            lines.append(f"   Predictions: {stats['total_predictions']}")
            
            if stats.get('avg_ai_error_f'):
                lines.append(f"   AI Error:   +/-{stats['avg_ai_error_f']:.1f}F ({stats.get('ai_wins', 0)} wins)")
            if stats.get('avg_hrrr_error_f'):
                lines.append(f"   HRRR Error: +/-{stats['avg_hrrr_error_f']:.1f}F ({stats.get('hrrr_wins', 0)} wins)")
            if stats.get('avg_gfs_error_f'):
                lines.append(f"   GFS Error:  +/-{stats['avg_gfs_error_f']:.1f}F ({stats.get('gfs_wins', 0)} wins)")
            
            if stats.get('avg_alpha') is not None:
                alpha = stats['avg_alpha']
                status = "[ADDING VALUE]" if alpha > 0 else "[DESTROYING VALUE]"
                lines.append(f"   Alpha Gen:  {alpha:+.2f}F {status}")
            
            # GROK FILTER STATS
            proposed = stats.get('ai_proposed_changes', 0)
            adjustments = stats.get('ai_adjustments', 0)
            abstentions = stats.get('ai_abstentions', 0)
            
            if proposed > 0:
                abstention_rate = (abstentions / proposed) * 100 if proposed > 0 else 0
                lines.append(f"   --- GROK FILTER ---")
                lines.append(f"   AI Proposed Changes: {proposed}")
                lines.append(f"   Changes Applied:     {adjustments}")
                lines.append(f"   Changes Ignored:     {abstentions}")
                lines.append(f"   Abstention Rate:     {abstention_rate:.1f}%")
            
            total_ai_wins += stats.get('ai_wins', 0)
            total_hrrr_wins += stats.get('hrrr_wins', 0)
            total_gfs_wins += stats.get('gfs_wins', 0)
            
            lines.append("")
        
        # Overall summary
        lines.append("-" * 65)
        lines.append(f"OVERALL WINS: AI={total_ai_wins}, HRRR={total_hrrr_wins}, GFS={total_gfs_wins}")
        
        if total_ai_wins >= total_hrrr_wins and total_ai_wins >= total_gfs_wins:
            lines.append("   -> AI is performing best!")
        elif total_hrrr_wins >= total_gfs_wins:
            lines.append("   -> HRRR model is most accurate")
        else:
            lines.append("   -> GFS model is most accurate")
    
    lines.append("=" * 65)
    lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(run_daily_audit())
    print(generate_accuracy_report())
