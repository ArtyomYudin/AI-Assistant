# core/tools/weather_tool.py
import logging

import httpx

logger = logging.getLogger(__name__)


WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Возвращает текущую погоду в указанном городе: температуру, "
                "ощущаемую температуру, влажность, скорость ветра. "
                "Используй, когда пользователь спрашивает про погоду, "
                "температуру на улице, осадки, ветер в конкретном городе."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "Город, например 'СПб', 'Санкт-Петербург', 'Москва'. "
                            "Используй как написал пользователь, не переводи."
                        ),
                    }
                },
                "required": ["city"],
            },
        },
    },
]


async def get_weather(city: str) -> dict:
    """
    Возвращает текущую погоду. При ошибке — {'error': '...'}.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "ru"},
            )
            geo_data = geo.json()
            if not geo_data.get("results"):
                return {"error": f"Город '{city}' не найден"}

            loc = geo_data["results"][0]
            weather_resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "current": (
                        "temperature_2m,relative_humidity_2m,"
                        "apparent_temperature,weather_code,wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )
            w = weather_resp.json().get("current", {})

        return {
            "city": loc.get("name"),
            "country": loc.get("country"),
            "temperature_c": w.get("temperature_2m"),
            "feels_like_c": w.get("apparent_temperature"),
            "humidity_pct": w.get("relative_humidity_2m"),
            "wind_speed_kmh": w.get("wind_speed_10m"),
            "weather_code": w.get("weather_code"),
        }
    except Exception as e:
        logger.exception("get_weather failed: %s", e)
        return {"error": str(e)}