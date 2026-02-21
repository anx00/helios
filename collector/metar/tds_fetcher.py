"""
NOAA Racing Fetcher - AWC TDS (XML) Source
Fetches METAR data from NOAA Aviation Weather Center Text Data Server (XML).
"""

import httpx
import xml.etree.ElementTree as ET
from typing import Optional
import logging

from collector.metar.temperature_parser import decode_temperature_from_raw

logger = logging.getLogger("metar_tds")

async def fetch_metar_tds(station_id: str) -> Optional[dict]:
    """
    Fetch METAR from AWC TDS XML feed.
    """
    url = f"https://aviationweather.gov/api/data/metar?ids={station_id}&format=xml"
    
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            root = ET.fromstring(response.text)
            
            # TDS XML structure: <METAR> <raw_text>...</raw_text> <temp_c>...</temp_c> ... </METAR>
            metar_elem = root.find(".//METAR")
            if metar_elem is None:
                return None
                
            def get_val(tag):
                elem = metar_elem.find(tag)
                return elem.text if elem is not None else None

            raw_ob = get_val("raw_text")
            obs_time_str = get_val("observation_time")
            temp_c = get_val("temp_c")
            dewp_c = get_val("dewpoint_c")
            wdir = get_val("wind_dir_degrees")
            wspd = get_val("wind_speed_kt")
            altim = get_val("altim_in_hg")
            visib = get_val("visibility_statute_mi")
            
            # Sky conditions
            sky_conds = []
            for sky in metar_elem.findall("sky_condition"):
                cover = sky.get("sky_cover")
                base = sky.get("cloud_base_ft_agl")
                if cover:
                    sky_conds.append({"cover": cover, "base": base})

            def safe_float(val):
                if not val: return None
                try:
                    # Handle "10+" -> 10.0
                    clean = val.replace("+", "").replace("SM", "")
                    return float(clean)
                except:
                    return None

            def safe_int(val):
                if not val: return None
                try:
                    return int(val)
                except:
                    return None

            parsed = decode_temperature_from_raw(
                raw_ob or "",
                fallback_temp_c=safe_float(temp_c),
                fallback_dewp_c=safe_float(dewp_c),
            )

            return {
                "station_id": station_id,
                "raw_ob": raw_ob,
                "obs_time": obs_time_str,
                "temp": parsed["temp_c"],
                "dewp": parsed["dewp_c"],
                "temp_c_low": parsed["temp_c_low"],
                "temp_c_high": parsed["temp_c_high"],
                "temp_f_low": parsed["temp_f_low"],
                "temp_f_high": parsed["temp_f_high"],
                "settlement_f_low": parsed["settlement_f_low"],
                "settlement_f_high": parsed["settlement_f_high"],
                "has_t_group": parsed["has_t_group"],
                "wdir": safe_int(wdir) if wdir != "VRB" else None,
                "wspd": safe_int(wspd),
                "altim": safe_float(altim),
                "visib": safe_float(visib),
                "clouds": sky_conds,
                "source": "AWC_TDS_XML"
            }
            
    except Exception as e:
        logger.debug(f"AWC TDS XML error for {station_id}: {e}")
        return None
