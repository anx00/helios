"""
NOAA Racing Fetcher - TG-FTP (Text) Source
Fetches METAR data from NOAA TG-FTP server (plain text files).
"""

import httpx
from datetime import datetime, timezone
from typing import Optional
import logging
import re

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
            
            # Parse simple values from raw METAR if needed, but for Racing we mostly care about TEMP
            # Raw METAR parsing is complex, but NOAA METARs have a standard T-group for precise temp
            # RMK ... T00890006 -> 8.9C / 0.6C
            
            temp_c = None
            dewp_c = None
            
            t_group = re.search(r' T([01])(\d{3})([01])(\d{3})', raw_ob)
            if t_group:
                # 1 = negative, 0 = positive
                t_sign = -1 if t_group.group(1) == '1' else 1
                t_val = int(t_group.group(2)) / 10.0
                temp_c = t_sign * t_val
                
                d_sign = -1 if t_group.group(3) == '1' else 1
                d_val = int(t_group.group(4)) / 10.0
                dewp_c = d_sign * d_val
            else:
                # Fallback to standard parsing (e.g. 09/01)
                std_temp = re.search(r' (M?\d{2})/(M?\d{2}| )', raw_ob)
                if std_temp:
                    t_str = std_temp.group(1).replace('M', '-')
                    temp_c = float(t_str)
                    d_str = std_temp.group(2).replace('M', '-')
                    if d_str.strip():
                        dewp_c = float(d_str)

            return {
                "station_id": station_id,
                "raw_ob": raw_ob,
                "obs_time": obs_time_header, # Use the header time as it's more complete
                "temp": temp_c,
                "dewp": dewp_c,
                "source": "NOAA_TG_FTP_TXT"
            }
            
    except Exception as e:
        logger.debug(f"NOAA TG-FTP error for {station_id}: {e}")
        return None
