def get_preferences(preferences: str):
    if not preferences:
        return []
    return [pref.strip() for pref in preferences.split(",") if pref.strip()]
