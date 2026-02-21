import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collector.metar.temperature_parser import decode_temperature_from_raw


def test_decode_temperature_prefers_t_group_precision() -> None:
    raw = "METAR KLGA 210051Z 03013G21KT 7SM -DZ OVC005 02/01 A2965 RMK AO2 SLP040 P0000 T00220006 $"
    parsed = decode_temperature_from_raw(raw)

    assert parsed["has_t_group"] is True
    assert parsed["temp_c"] == 2.2
    assert parsed["dewp_c"] == 0.6
    assert parsed["temp_c_low"] == 2.2
    assert parsed["temp_c_high"] == 2.2
    assert parsed["settlement_f_low"] == 36
    assert parsed["settlement_f_high"] == 36


def test_decode_temperature_without_t_group_exposes_range() -> None:
    raw = "SPECI KXYZ 121455Z 06005KT 10SM FEW040 02/01 A3000 RMK AO2"
    parsed = decode_temperature_from_raw(raw)

    assert parsed["has_t_group"] is False
    assert parsed["temp_c"] == 2.0
    assert parsed["temp_c_low"] == 1.5
    assert parsed["temp_c_high"] == 2.4
    assert parsed["settlement_f_low"] == 35
    assert parsed["settlement_f_high"] == 36


def test_decode_temperature_without_t_group_can_span_two_f_values() -> None:
    raw = "METAR EGLC 210050Z AUTO 24008KT 9999 NCD 09/07 Q1017"
    parsed = decode_temperature_from_raw(raw)

    assert parsed["has_t_group"] is False
    assert parsed["temp_c"] == 9.0
    assert parsed["settlement_f_low"] == 47
    assert parsed["settlement_f_high"] == 49


def test_decode_temperature_fallback_keeps_exact_value() -> None:
    parsed = decode_temperature_from_raw("", fallback_temp_c=1.7, fallback_dewp_c=-0.6)

    assert parsed["used_fallback"] is True
    assert parsed["temp_c"] == 1.7
    assert parsed["dewp_c"] == -0.6
    assert parsed["temp_c_low"] == 1.7
    assert parsed["temp_c_high"] == 1.7
    assert parsed["settlement_f_low"] == 35
    assert parsed["settlement_f_high"] == 35
