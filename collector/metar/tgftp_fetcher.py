"""
NOAA Racing Fetcher - TG-FTP (Text) Source
Fetches METAR data from NOAA TG-FTP server (plain text files).
"""

import httpx
from typing import Optional
import logging

from collector.metar.temperature_parser import decode_temperature_from_raw

logger = logging.getLogger("metar_tgftp")

async def fetch_metar_tgftp(station_id: str) -> Optional[dict]:
    """
    Fetch METAR from NOAA TG-FTP (HTTPS Direct).
    URL: https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station_id}.TXT
    """
    url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station_id}.TXT"
    
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Format is:
            # 2024/01/12 21:00
            # KATL 122052Z 31008KT 10SM FEW250 09/01 A3012 RMK AO2 SLP198 T00890006
            
            lines = response.text.strip().splitlines()
            if len(lines) < 2:
                return None
                
            obs_time_header = lines[0].strip() # 2024/01/12 21:00
            raw_ob = lines[1].strip()
            
            parsed = decode_temperature_from_raw(raw_ob)

            return {
                "station_id": station_id,
                "raw_ob": raw_ob,
                "obs_time": obs_time_header, # Use the header time as it's more complete
                "temp": parsed["temp_c"],
                "dewp": parsed["dewp_c"],
                "temp_c_low": parsed["temp_c_low"],
                "temp_c_high": parsed["temp_c_high"],
                "temp_f_low": parsed["temp_f_low"],
                "temp_f_high": parsed["temp_f_high"],
                "settlement_f_low": parsed["settlement_f_low"],
                "settlement_f_high": parsed["settlement_f_high"],
                "has_t_group": parsed["has_t_group"],
                "source": "NOAA_TG_FTP_TXT"
            }
            
    except Exception as e:
        logger.debug(f"NOAA TG-FTP error for {station_id}: {e}")
        return None
