from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from zoneinfo import ZoneInfo

import aiohttp

from collector.metar_fetcher import fetch_metar_race
from collector.pws_fetcher import fetch_and_publish_pws
from config import STATIONS, get_active_stations
from core.world import get_world
from database import get_latest_prediction, get_latest_prediction_for_date

logger = logging.getLogger("telegram_bot")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_allowed_chat_ids(raw: str) -> Set[str]:
    values: Set[str] = set()
    for part in (raw or "").split(","):
        item = part.strip()
        if item:
            values.add(item)
    return values


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, decimals: int = 2, suffix: str = "") -> str:
    num = _safe_float(value)
    if num is None:
        return "n/a"
    return f"{num:.{decimals}f}{suffix}"


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if _safe_float(v) is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


@dataclass(frozen=True)
class TelegramBotSettings:
    token: str
    allowed_chat_ids: Optional[Set[str]]
    poll_timeout_seconds: int = 25
    max_message_chars: int = 3900

    @classmethod
    def from_env(cls) -> Optional["TelegramBotSettings"]:
        token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        if not token:
            return None

        if not _env_bool("TELEGRAM_BOT_ENABLED", True):
            return None

        allowed_raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
        allowed = _parse_allowed_chat_ids(allowed_raw)
        return cls(
            token=token,
            allowed_chat_ids=allowed if allowed else None,
            poll_timeout_seconds=max(10, int(os.getenv("TELEGRAM_POLL_TIMEOUT_SECONDS", "25"))),
            max_message_chars=max(1000, int(os.getenv("TELEGRAM_MAX_MESSAGE_CHARS", "3900"))),
        )


class HeliosTelegramBot:
    """Telegram bot with station selection and HELIOS mobile summaries."""

    def __init__(self, settings: TelegramBotSettings):
        self.settings = settings
        self._api_base_url = f"https://api.telegram.org/bot{settings.token}"
        self._offset: Optional[int] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._chat_station: Dict[str, str] = {}
        self._unauthorized_warned: Set[str] = set()

    async def run_forever(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.settings.poll_timeout_seconds + 15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._session = session
            me = await self._safe_get_me()
            if me:
                logger.info("Telegram bot online as @%s", me.get("username", "unknown"))
            else:
                logger.info("Telegram bot online")

            while True:
                try:
                    updates = await self._fetch_updates()
                    for update in updates:
                        update_id = update.get("update_id")
                        if isinstance(update_id, int):
                            self._offset = update_id + 1
                        await self._handle_update(update)
                except asyncio.CancelledError:
                    logger.info("Telegram bot loop cancelled")
                    raise
                except Exception as exc:
                    logger.warning("Telegram polling error: %s", exc)
                    await asyncio.sleep(3)

    async def _safe_get_me(self) -> Optional[Dict[str, Any]]:
        try:
            result = await self._api_post("getMe", {})
            return result if isinstance(result, dict) else None
        except Exception as exc:
            logger.warning("Telegram getMe failed: %s", exc)
            return None

    async def _fetch_updates(self) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "timeout": self.settings.poll_timeout_seconds,
            "allowed_updates": ["message", "edited_message"],
        }
        if self._offset is not None:
            payload["offset"] = self._offset
        result = await self._api_post(
            "getUpdates",
            payload,
            timeout_seconds=self.settings.poll_timeout_seconds + 10,
        )
        if not isinstance(result, list):
            return []
        return [item for item in result if isinstance(item, dict)]

    async def _api_post(
        self,
        method: str,
        payload: Dict[str, Any],
        timeout_seconds: Optional[int] = None,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("Telegram session not initialized")

        url = f"{self._api_base_url}/{method}"
        timeout = aiohttp.ClientTimeout(total=timeout_seconds) if timeout_seconds else None
        async with self._session.post(url, json=payload, timeout=timeout) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"Telegram HTTP {response.status}: {body[:240]}")
        data = json.loads(body)
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error in {method}: {data}")
        return data.get("result")

    async def _handle_update(self, update: Dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            return

        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", "")).strip()
        if not chat_id:
            return
        if not self._is_chat_allowed(chat_id):
            if chat_id not in self._unauthorized_warned:
                self._unauthorized_warned.add(chat_id)
                await self._send_text(chat_id, "Chat no autorizado para este bot.")
            return

        text = str(message.get("text") or "").strip()
        if not text:
            return
        await self._dispatch_text(chat_id, text)

    def _is_chat_allowed(self, chat_id: str) -> bool:
        allowed = self.settings.allowed_chat_ids
        return not allowed or chat_id in allowed

    async def _dispatch_text(self, chat_id: str, text: str) -> None:
        active_ids = list(get_active_stations().keys())
        upper_text = text.upper()
        if upper_text in active_ids:
            self._chat_station[chat_id] = upper_text
            await self._send_text(
                chat_id,
                f"Estacion activa: {upper_text} ({STATIONS[upper_text].name})",
                reply_markup=self._station_keyboard(active_ids),
            )
            return

        parts = text.split()
        command = parts[0].split("@", 1)[0].lower()
        args = parts[1:]

        if command in {"/start", "/help", "/ayuda"}:
            await self._cmd_start(chat_id, active_ids)
            return
        if command in {"/stations", "/estaciones"}:
            await self._cmd_stations(chat_id, active_ids)
            return
        if command in {"/station", "/estacion"}:
            await self._cmd_station(chat_id, args, active_ids)
            return
        if command == "/helios":
            await self._cmd_helios(chat_id, args)
            return
        if command == "/metar":
            await self._cmd_metar(chat_id, args)
            return
        if command == "/pws":
            await self._cmd_pws(chat_id, args)
            return
        if command in {"/resumen", "/status"}:
            await self._cmd_resumen(chat_id, args)
            return

        await self._send_text(
            chat_id,
            "Comando no reconocido. Usa /help para ver comandos.",
            reply_markup=self._station_keyboard(active_ids),
        )

    async def _cmd_start(self, chat_id: str, active_ids: Sequence[str]) -> None:
        station_id = self._current_station_for_chat(chat_id, active_ids)
        msg = (
            "HELIOS Telegram Bot\n\n"
            "Comandos:\n"
            "/helios [ICAO] - Prediccion actual HELIOS\n"
            "/pws [ICAO] - PWS completo (pesos, medias, metricas y estaciones)\n"
            "/metar [ICAO] - NOAA METAR actual\n"
            "/resumen [ICAO] - HELIOS + METAR + resumen PWS\n"
            "/stations - Lista de estaciones\n"
            "/station ICAO - Cambiar estacion\n\n"
            f"Estacion actual: {station_id}\n"
            "Tambien puedes tocar un boton de estacion."
        )
        await self._send_text(chat_id, msg, reply_markup=self._station_keyboard(active_ids))

    async def _cmd_stations(self, chat_id: str, active_ids: Sequence[str]) -> None:
        selected = self._current_station_for_chat(chat_id, active_ids)
        lines = ["Estaciones activas:"]
        for sid in active_ids:
            station = STATIONS[sid]
            marker = "->" if sid == selected else "  "
            lines.append(f"{marker} {sid}: {station.name} [{station.timezone}]")
        await self._send_text(chat_id, "\n".join(lines), reply_markup=self._station_keyboard(active_ids))

    async def _cmd_station(self, chat_id: str, args: Sequence[str], active_ids: Sequence[str]) -> None:
        if not args:
            current = self._current_station_for_chat(chat_id, active_ids)
            await self._send_text(
                chat_id,
                f"Estacion actual: {current}\nUsa /station ICAO para cambiarla.",
                reply_markup=self._station_keyboard(active_ids),
            )
            return

        wanted = str(args[0]).upper()
        if wanted not in active_ids:
            await self._send_text(
                chat_id,
                f"Estacion invalida: {wanted}. Usa /stations para ver opciones.",
                reply_markup=self._station_keyboard(active_ids),
            )
            return

        self._chat_station[chat_id] = wanted
        await self._send_text(
            chat_id,
            f"Estacion cambiada a {wanted} ({STATIONS[wanted].name}).",
            reply_markup=self._station_keyboard(active_ids),
        )

    async def _cmd_helios(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones activas disponibles.")
            return

        station = STATIONS[station_id]
        now_local = datetime.now(ZoneInfo(station.timezone))
        requested_target_date = now_local.date().isoformat()

        pred = (
            get_latest_prediction_for_date(station_id, requested_target_date)
            or get_latest_prediction(station_id)
        )
        if not pred:
            await self._send_text(chat_id, f"No hay predicciones HELIOS para {station_id}.")
            return

        pred_target = str(pred.get("target_date") or requested_target_date)
        lines = [
            f"HELIOS - {station_id} ({station.name})",
            f"Target date local: {requested_target_date}",
            f"Target date pred:  {pred_target}",
            f"Timestamp DB:      {pred.get('timestamp', 'n/a')}",
            "",
            f"Final prediction: {_fmt(pred.get('final_prediction_f'), 1, 'F')}",
            f"HRRR base:        {_fmt(pred.get('hrrr_max_raw_f'), 1, 'F')}",
            f"Deviation now:    {_fmt(pred.get('current_deviation_f'), 1, 'F')}",
            f"Physics delta:    {_fmt(pred.get('physics_adjustment_f'), 1, 'F')}",
            f"Delta weight:     {_fmt(pred.get('delta_weight'), 3)}",
            f"Confidence score: {_fmt(pred.get('confidence_score'), 3)}",
        ]
        await self._send_text(chat_id, "\n".join(lines))

    async def _cmd_metar(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones activas disponibles.")
            return

        station = STATIONS[station_id]
        metar = await fetch_metar_race(station_id)
        if not metar:
            await self._send_text(chat_id, f"No se pudo obtener METAR NOAA para {station_id}.")
            return

        obs_utc = metar.observation_time.astimezone(ZoneInfo("UTC"))
        obs_local = metar.observation_time.astimezone(ZoneInfo(station.timezone))
        racing = metar.racing_results or {}
        race_line = ", ".join(
            f"{src}={_fmt(temp, 1, 'F')}" for src, temp in sorted(racing.items(), key=lambda x: x[0])
        ) if racing else "n/a"

        lines = [
            f"NOAA METAR actual - {station_id} ({station.name})",
            f"Obs UTC:   {obs_utc.isoformat()}",
            f"Obs local: {obs_local.isoformat()}",
            "",
            f"Temp:      {_fmt(metar.temp_f, 1, 'F')} ({_fmt(metar.temp_c, 2, 'C')})",
            f"Dewpoint:  {_fmt(metar.dewpoint_c, 2, 'C')}",
            f"Humidity:  {_fmt(metar.humidity_pct, 1, '%')}",
            f"Wind dir:  {metar.wind_dir_degrees if metar.wind_dir_degrees is not None else 'n/a'}",
            f"Wind kt:   {metar.wind_speed_kt if metar.wind_speed_kt is not None else 'n/a'}",
            f"Sky:       {metar.sky_condition or 'n/a'}",
            "",
            f"Race: {race_line}",
        ]
        await self._send_text(chat_id, "\n".join(lines))

    async def _cmd_pws(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones activas disponibles.")
            return

        await self._send_text(chat_id, f"Actualizando PWS para {station_id}...")

        metar = await fetch_metar_race(station_id)
        official_temp_c = metar.temp_c if metar else None
        official_obs_time_utc = metar.observation_time if metar else None
        consensus = await fetch_and_publish_pws(
            station_id,
            official_temp_c=official_temp_c,
            official_obs_time_utc=official_obs_time_utc,
        )

        snapshot = get_world().get_snapshot()
        details = ((snapshot.get("pws_details") or {}).get(station_id)) or []
        metrics = ((snapshot.get("pws_metrics") or {}).get(station_id)) or {}

        if not details and not consensus:
            await self._send_text(chat_id, f"No hay datos PWS disponibles para {station_id}.")
            return

        all_mean_f = _mean([row.get("temp_f") for row in details if isinstance(row, dict)])
        valid_mean_f = _mean(
            [row.get("temp_f") for row in details if isinstance(row, dict) and bool(row.get("valid"))]
        )

        source_groups: Dict[str, List[float]] = {}
        for row in details:
            if not isinstance(row, dict):
                continue
            src = str(row.get("source") or "UNKNOWN")
            t = _safe_float(row.get("temp_f"))
            if t is None:
                continue
            source_groups.setdefault(src, []).append(t)

        source_mean_line = ", ".join(
            f"{src}:{_fmt(_mean(vals), 1, 'F')}" for src, vals in sorted(source_groups.items(), key=lambda x: x[0])
        ) if source_groups else "n/a"

        lines = [
            f"PWS completo - {station_id} ({STATIONS[station_id].name})",
            f"Rows (world.pws_details): {len(details)}",
            f"Media all:   {_fmt(all_mean_f, 2, 'F')}",
            f"Media valid: {_fmt(valid_mean_f, 2, 'F')}",
            f"Media por fuente: {source_mean_line}",
        ]

        if consensus:
            lines.extend(
                [
                    "",
                    f"Consensus source: {consensus.data_source}",
                    f"Consensus median: {_fmt(consensus.median_temp_f, 1, 'F')} ({_fmt(consensus.median_temp_c, 2, 'C')})",
                    f"MAD: {_fmt(consensus.mad, 3, 'C')}",
                    f"Support: {consensus.support}/{consensus.total_queried}",
                    f"Weighted support: {_fmt(consensus.weighted_support, 3)}",
                    f"Drift vs NOAA: {_fmt(consensus.drift_c, 2, 'C')}",
                ]
            )

        if isinstance(metrics, dict) and metrics:
            metric_keys = ", ".join(sorted(metrics.keys()))
            lines.append(f"Metric keys: {metric_keys}")

        await self._send_text(chat_id, "\n".join(lines))

        consensus_payload = None
        if consensus:
            consensus_payload = {
                "station_id": consensus.station_id,
                "median_temp_c": consensus.median_temp_c,
                "median_temp_f": consensus.median_temp_f,
                "mad": consensus.mad,
                "support": consensus.support,
                "total_queried": consensus.total_queried,
                "obs_time_utc": consensus.obs_time_utc.isoformat() if consensus.obs_time_utc else None,
                "drift_c": consensus.drift_c,
                "data_source": consensus.data_source,
                "weighted_support": consensus.weighted_support,
                "station_details": consensus.station_details,
                "source_metrics": consensus.source_metrics,
                "learning_weights": consensus.learning_weights,
                "learning_profiles": consensus.learning_profiles,
            }

        full_payload = {
            "station_id": station_id,
            "timestamp_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
            "consensus": consensus_payload,
            "pws_metrics": metrics,
            "pws_details": details,
        }
        payload_text = json.dumps(full_payload, ensure_ascii=True, indent=2)
        await self._send_text(chat_id, f"PWS payload completo:\n{payload_text}")

    async def _cmd_resumen(self, chat_id: str, args: Sequence[str]) -> None:
        station_id = self._resolve_station_from_args(chat_id, args)
        if not station_id:
            await self._send_text(chat_id, "No hay estaciones activas disponibles.")
            return

        station = STATIONS[station_id]
        now_local = datetime.now(ZoneInfo(station.timezone))
        target_date = now_local.date().isoformat()
        pred = (
            get_latest_prediction_for_date(station_id, target_date)
            or get_latest_prediction(station_id)
        )
        metar = await fetch_metar_race(station_id)
        pws_snapshot = get_world().get_snapshot()
        pws_details = ((pws_snapshot.get("pws_details") or {}).get(station_id)) or []
        pws_metrics = ((pws_snapshot.get("pws_metrics") or {}).get(station_id)) or {}

        lines = [f"Resumen - {station_id} ({station.name})"]
        if pred:
            lines.append(f"HELIOS: {_fmt(pred.get('final_prediction_f'), 1, 'F')} (ts={pred.get('timestamp', 'n/a')})")
        else:
            lines.append("HELIOS: n/a")

        if metar:
            lines.append(f"NOAA METAR: {_fmt(metar.temp_f, 1, 'F')} (obs={metar.observation_time.isoformat()})")
        else:
            lines.append("NOAA METAR: n/a")

        lines.append(f"PWS rows: {len(pws_details)}")
        lines.append(f"PWS metric keys: {', '.join(sorted(pws_metrics.keys())) if pws_metrics else 'n/a'}")
        lines.append("Para detalle completo usa /pws")

        await self._send_text(chat_id, "\n".join(lines))

    def _resolve_station_from_args(self, chat_id: str, args: Sequence[str]) -> Optional[str]:
        active = list(get_active_stations().keys())
        if not active:
            return None

        if args:
            wanted = str(args[0]).upper()
            if wanted in active:
                self._chat_station[chat_id] = wanted
                return wanted
            return self._current_station_for_chat(chat_id, active)
        return self._current_station_for_chat(chat_id, active)

    def _current_station_for_chat(self, chat_id: str, active_ids: Sequence[str]) -> str:
        saved = self._chat_station.get(chat_id)
        if saved in active_ids:
            return saved
        default_station = active_ids[0]
        self._chat_station[chat_id] = default_station
        return default_station

    def _station_keyboard(self, active_ids: Sequence[str]) -> Dict[str, Any]:
        ids = list(active_ids)
        rows = [ids[i : i + 2] for i in range(0, len(ids), 2)]
        return {
            "keyboard": rows,
            "resize_keyboard": True,
            "one_time_keyboard": False,
        }

    async def _send_text(
        self,
        chat_id: str,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> None:
        chunks = self._chunk_text(text or "")
        for idx, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {"chat_id": chat_id, "text": chunk}
            if idx == 0 and reply_markup:
                payload["reply_markup"] = reply_markup
            await self._api_post("sendMessage", payload)

    def _chunk_text(self, text: str) -> List[str]:
        limit = self.settings.max_message_chars
        if len(text) <= limit:
            return [text]

        chunks: List[str] = []
        current = ""
        for raw_line in text.splitlines(keepends=True):
            line = raw_line
            while len(line) > limit:
                piece = line[:limit]
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(piece)
                line = line[limit:]

            if len(current) + len(line) > limit:
                if current:
                    chunks.append(current)
                current = line
            else:
                current += line

        if current:
            chunks.append(current)
        return chunks


def build_telegram_bot_from_env() -> Optional[HeliosTelegramBot]:
    settings = TelegramBotSettings.from_env()
    if not settings:
        return None
    return HeliosTelegramBot(settings)

