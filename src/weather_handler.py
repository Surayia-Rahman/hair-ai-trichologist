import requests
import json

def fetch_local_weather(lat=40.7128, lon=-74.0060):
    """
    Fetches real-time relative humidity and temperature using the free Open-Meteo API.
    Defaults to New York City coordinates if none are provided.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise exception for bad status codes
        data = response.json()
        
        current_stats = data.get("current", {})
        temperature = current_stats.get("temperature_2m")
        humidity = current_stats.get("relative_humidity_2m")
        
        return {
            "status": "success",
            "temperature_c": temperature,
            "relative_humidity_pct": humidity
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def calculate_hair_environmental_risk(humidity_pct):
    """
    Translates raw humidity percentages into actionable clinical hair stress metrics.
    """
    if humidity_pct is None:
        return "Unknown", "Unable to calculate risk factors due to missing weather metrics."
        
    if humidity_pct >= 75:
        risk_level = "High Frizz / High Hydration Loss Risk"
        advice = "Humid air will break hydrogen bonds. Highly porous hair types (Curly/Kinky) require anti-humectants, heavy sealants, or silicones to lock out atmospheric moisture."
    elif 40 <= humidity_pct < 75:
        risk_level = "Optimal Optimal Balance"
        advice = "Atmospheric moisture is stable. Standard balanced routine of humectants (glycerin) and light emollients will perform perfectly."
    else:
        risk_level = "Severe Dryness / Dehydration Risk"
        advice = "Low humidity will strip moisture out of the hair shaft. Focus heavily on deep conditioning, leave-in creams, and water-soluble humectants."
        
    return risk_level, advice

if __name__ == "__main__":
    # Test the module with default coordinates
    print("--- Testing Weather Handler Module ---")
    weather_data = fetch_local_weather()
    print(f"Raw API Response Snippet: {weather_data}")
    
    if weather_data["status"] == "success":
        rh = weather_data["relative_humidity_pct"]
        risk, tips = calculate_hair_environmental_risk(rh)
        print(f"\nCurrent Humidity: {rh}%")
        print(f"Calculated Risk Level: {risk}")
        print(f"Trichologist Advice: {tips}\n")
