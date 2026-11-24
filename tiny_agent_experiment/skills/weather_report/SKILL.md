---
name: weather_report
description: Write a short travel weather report for one or more cities
---

# Weather Report Skill

1. Call `get_weather` once for every city the user mentions.
2. Reply with one line per city: `<city>: <status>, <temperature>°C — <advice>`.
3. Advice: Rainy → "bring an umbrella", below 15°C → "bring a jacket", otherwise "enjoy".
4. End with one sentence naming the best city to visit today.
