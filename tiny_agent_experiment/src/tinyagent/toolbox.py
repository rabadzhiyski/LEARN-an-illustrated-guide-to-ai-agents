import datetime
import random


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    temperature = random.randint(10, 30)
    status = random.choice(["Sunny", "Cloudy", "Rainy"])
    return f"Weather in {location}: {status}, {temperature}°C"


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return float(a) + float(b)


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return float(a) * float(b)


def today() -> str:
    """Return today's date (YYYY-MM-DD)."""
    return datetime.date.today().isoformat()


def days_between(start: str, end: str) -> int:
    """Number of days between two ISO dates (YYYY-MM-DD)."""
    return (
        datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)
    ).days
