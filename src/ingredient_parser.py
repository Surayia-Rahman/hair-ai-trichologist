import re
import json

def parse_ingredients(ingredient_string):
    """
    Cleans a raw ingredient text string, tokenizes it by commas, 
    and checks each ingredient against strict chemical definitions.
    """
    if not ingredient_string or not isinstance(ingredient_string, str):
        return {"error": "Invalid text input"}

    # Normalize text: convert to lowercase and remove trailing periods or brackets
    clean_str = ingredient_string.lower().strip()
    # Split by comma 
    raw_ingredients = [ing.strip() for ing in re.split(r',\s*', clean_str) if ing.strip()]
    
    # Define clinical trichology classifications using regex patterns
    patterns = {
        "sulfates": [r".*sulfate.*", r".*sulfosuccinate.*"],
        "heavy_silicones": [r".*dimethicone$", r".*amodimethicone.*", r".*cyclomethicone.*"],
        "water_soluble_silicones": [r".*copolyol.*", r"^peg-.*dimethicone.*"],
        "humectants": [r".*glycerin.*", r".*glycol.*", r".*panthenol.*", r".*hyaluronic.*"]
    }
    
    # Initialize our analysis report structural mapping
    analysis_report = {
        "total_ingredients_detected": len(raw_ingredients),
        "detected_groups": {
            "sulfates": [],
            "heavy_silicones": [],
            "water_soluble_silicones": [],
            "humectants": []
        },
        "flags": []
    }
    
    # Match ingredients against our criteria
    for ing in raw_ingredients:
        for group, regex_list in patterns.items():
            for pattern in regex_list:
                if re.match(pattern, ing):
                    # Avoid adding duplicates if multiple regexes catch the same item
                    if ing not in analysis_report["detected_groups"][group]:
                        analysis_report["detected_groups"][group].append(ing)

    # Generate engineering triggers based on safety profiles
    if analysis_report["detected_groups"]["sulfates"]:
        analysis_report["flags"].append("STRIPPING_RISK: Contains harsh surfactants that degrade fragile hair lipids.")
        
    if analysis_report["detected_groups"]["heavy_silicones"] and not analysis_report["detected_groups"]["sulfates"]:
        analysis_report["flags"].append("BUILDUP_RISK: Contains heavy silicones without a matching sulfate cleanser to strip them away.")
        
    return analysis_report

if __name__ == "__main__":
    print("--- Testing Ingredient Parser Module ---")
    
    # Test sample simulating a typical supermarket moisturizing conditioner
    sample_bottle = "Water, Cetearyl Alcohol, Dimethicone, Glycerin, Behentrimonium Chloride, Sodium Lauryl Sulfate, Propylene Glycol, Fragrance"
    
    results = parse_ingredients(sample_bottle)
    print(json.dumps(results, indent=4))
