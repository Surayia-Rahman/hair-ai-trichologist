import os
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

# Import our production helper modules
from weather_handler import fetch_local_weather, calculate_hair_environmental_risk
from ingredient_parser import parse_ingredients
from vision_classifier import get_data_transformers, build_model

# Global configurations
MODEL_WEIGHTS_PATH = "hair-ai-trichologist/data/processed/efficientnet_hair_weights.pth"
CLASSES_PATH = "hair-ai-trichologist/src/classes.txt"

# Load the class labels dynamically
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r") as f:
        CLASSES = [line.strip() for line in f.readlines()]
else:
    CLASSES = ['Straight', 'Wavy', 'curly', 'dreadlocks', 'kinky'] # Fallback default

# Instantiate and prepare the real model globally so it stays warm in memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None

if os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"-> Loading trained model weights into app interface memory ({DEVICE.type.upper()})...")
    MODEL = build_model(num_classes=len(CLASSES))
    MODEL.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
else:
    print("-> [WARNING] No trained model weights found. App will run in simulation mode.")

def UI_orchestrator(input_image, zipcode_coords, ingredient_text):
    """
    The core bridge connecting visual UI inputs to our real PyTorch model weights
    and secondary textual APIs.
    """
    if input_image is None:
        return "⚠️ Error: Please upload or take a photo of your hair/scalp first!", "", "", ""
        
    # Parse coordinates from user input (Default to Toronto if blank/malformed)
    lat, lon = 43.6532, -79.3832
    if zipcode_coords and "," in zipcode_coords:
        try:
            lat = float(zipcode_coords.split(",")[0].strip())
            lon = float(zipcode_coords.split(",")[1].strip())
        except:
            pass

    # --- 1. Weather Logic ---
    weather = fetch_local_weather(lat, lon)
    if weather["status"] == "success":
        humidity = weather["relative_humidity_pct"]
        temp = weather["temperature_c"]
        env_risk, env_advice = calculate_hair_environmental_risk(humidity)
        weather_output = f"🌡️ Temperature: {temp}°C\n💧 Relative Humidity: {humidity}%\n⚠️ Climate Risk: {env_risk}"
    else:
        env_advice = "Weather service data unavailable."
        weather_output = "❌ Weather API Fetch Failed."

    # --- 2. Ingredient Logic ---
    ingredient_results = parse_ingredients(ingredient_text)
    flags_list = "\n".join([f"• {f}" for f in ingredient_results["flags"]])
    if not flags_list:
        flags_list = "• No immediate formulation conflicts detected."
    
    ingredient_output = (
        f"📋 Total Compounds Detected: {ingredient_results['total_ingredients_detected']}\n"
        f"🚨 Formulation Flags:\n{flags_list}"
    )

    # --- 3. Real Vision Inference Logic ---
    if MODEL is not None:
        try:
            # Open image file and run our standardized transformations
            img = Image.open(input_image).convert('RGB')
            transform = get_data_transformers()
            img_tensor = transform(img).unsqueeze(0).to(DEVICE) # Add batch dimension and push to GPU
            
            with torch.no_grad():
                outputs = MODEL(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, class_idx = torch.max(probabilities, 0)
                
            predicted_class = CLASSES[class_idx.item()]
            confidence_score = confidence.item() * 100
            vision_output = f"🧬 Diagnosed Texture: {predicted_class}\n🎯 AI Confidence: {confidence_score:.2f}%"
        except Exception as e:
            predicted_class = "Unknown"
            vision_output = f"❌ Vision Engine Failure: {str(e)}"
            confidence_score = 0.0
    else:
        # Fallback if model files aren't found
        predicted_class = "Straight"
        vision_output = "ℹ️ Running in Simulation Mode (No weights detected)."

    # --- 4. Generate Unified Clinical Decision ---
    verdict = (
        f"==================================================\n"
        f"               TRICHOLOGIST CLINICAL VERDICT       \n"
        f"==================================================\n"
        f"Diagnosed Profile: {predicted_class} Hair Texture Type\n\n"
        f"Environmental Advice:\n{env_advice}\n\n"
    )
    
    if predicted_class.lower() in ['curly', 'kinky', 'dreadlocks'] and ingredient_results["detected_groups"]["sulfates"]:
        verdict += (
            "🚨 CRITICAL ROUTINE ADJUSTMENT REQUIRED:\n"
            "Because your hair type is structurally prone to severe moisture loss, using a product containing harsh "
            "cleansing sulfates in this specific climate will strip your protective lipid layers. "
            "Switch to a gentle, sulfate-free cream co-wash immediately to protect your pattern."
        )
    else:
        verdict += "✅ ROUTINE VERDICT:\nYour product formulation is structurally compatible with your diagnosed hair shape under current atmospheric conditions."

    return vision_output, weather_output, ingredient_output, verdict

# --- GRADIO INTERFACE DESIGN ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💇‍♂️ AI Trichologist & Ingredient Formulator")
    gr.Markdown("Identify your real hair profile, evaluate chemical ingredients, and analyze real-time climate impacts instantly.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📥 Step 1: User Inputs")
            input_img = gr.Image(type="filepath", label="Upload Hair/Scalp Photo or Use Camera")
            geo_input = gr.Textbox(value="43.6532, -79.3832", label="Location Coordinates (Lat, Lon)", placeholder="e.g., 40.7128, -74.0060")
            ing_input = gr.Textbox(
                value="Water, Sodium Lauryl Sulfate, Dimethicone, Glycerin", 
                label="Product Ingredients (Paste text list separated by commas)", 
                lines=3
            )
            submit_btn = gr.Button("Analyze My Hair Profile", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 📤 Step 2: System Diagnostic Metrics")
            with gr.Row():
                vis_out = gr.Textbox(label="Computer Vision Analysis", lines=2)
                wea_out = gr.Textbox(label="Local Weather Telemetry", lines=3)
            ing_out = gr.Textbox(label="Ingredient Analysis Summary", lines=3)
            
    gr.Markdown("### 📋 Step 3: Comprehensive Recommendation")
    final_verdict = gr.Textbox(label="AI Trichologist Output Report", lines=8)

    submit_btn.click(
        fn=UI_orchestrator, 
        inputs=[input_img, geo_input, ing_input], 
        outputs=[vis_out, wea_out, ing_out, final_verdict]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
