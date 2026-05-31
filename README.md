
# AI Trichologist & Hair Care Ingredient Formulator

An intelligent, multi-modal data science solution that analyzes hair textures using computer vision, evaluates product ingredient safety profiles via automated keyword parsing, and integrates real-time microclimate environmental data to generate hyper-personalized, clinical hair care routines.

---

## 🚀 Key Features

* **Computer Vision Diagnostic Layer:** Processes consumer hair/scalp imagery using an optimized, pre-trained **EfficientNet-B0** backbone in PyTorch, mapping textures across 5 structural classes (*Straight, Wavy, Curly, Dreadlocks, Kinky*) with over 84% validation accuracy.
* **Context-Aware NLP Ingredient Parser:** Cleans and cross-references product ingredient lists to instantly flag high-strip surfactants (sulfates) and non-water-soluble barrier coatings (heavy silicones).
* **Live Environmental Telemetry Engine:** Leverages the open-access **Open-Meteo API** to pull live relative humidity data, dynamically altering trichology recommendations based on local frizz or atmospheric dehydration risks.
* **Streamlined Interactive Web UI:** Uses a responsive **Gradio** frontend dashboard to let users capture live photos via webcams, input coordinates, and receive instant diagnostic reports.

---

## 📁 Repository Blueprint

```text
hair-ai-trichologist/
│
├── data/
│   └── processed/           # Trained model weight checkpoints (.pth)
│
├── reports/
│   └── figures/             # Auto-generated visual diagnostics
│       ├── real_hair_distribution.png
│       ├── training_metrics_curves.png
│       ├── gradio_input.png
│       └── gradio_output.png
│
├── src/                     # Production Backend Sub-Modules
│   ├── __init__.py
│   ├── data_check.py        # Dataset validation checks
│   ├── vision_classifier.py # PyTorch neural network configuration
│   ├── ingredient_parser.py # Rule-based chemical string parser
│   └── weather_handler.py   # Free API environment connector
│
├── ai_hair_trichologist.py  # Master Orchestrator & Live App Launcher
├── requirements.txt         # Project software dependencies
└── README.md                # Systems documentation

```

---

## 📊 Analytics & Interactive UI

### 1. Model Convergence Performance

The model freezes pre-trained extraction layers and fits custom dense layers on a dataset of ~2,000 multi-class images, achieving solid accuracy in a highly constrained training layout.

### 2. Live Application Interface

Our live web app streams the backend pipeline seamlessly to the end-user. Below are the live snapshots of the interface capturing inputs and resolving a clinical verdict:

| User Submissions Dashboard | Deep Diagnostic Output Report |
| --- | --- |
|  |  |

---

## 🛠️ Local Deployment Guide

To run this interactive ecosystem locally on your machine, follow these steps:

### 1. Clone the Architecture

```bash
git clone [https://github.com/Surayia-Rahman/hair-ai-trichologist.git](https://github.com/Surayia-Rahman/hair-ai-trichologist.git)
cd hair-ai-trichologist

```

### 2. Install Project Requirements

```bash
pip install -r requirements.txt

```

### 3. Launch the App Link

```bash
python ai_hair_trichologist.py

```

This will spin up a local server address and export a secure, public `.gradio.live` link allowing deployment streaming to external test hardware or mobile browsers!


### ⚠️ Current Limitations

* **Geographic Dependencies & Granularity:** Using a global weather API relies on static coordinates rather than automated IP triangulation. This can create edge cases where regional microclimates (like indoor artificial heating or intense seasonal humidity shifts in specific micro-zones) aren't fully accounted for by baseline atmospheric telemetry.
* **Static Rule-Based NLP Constraints:** The ingredient parser relies on deterministic keyword string matching. It cannot dynamically evaluate ingredient concentrations, synergistic chemical formulations, or identify misspelled/unlisted chemical variants that fall outside its static dictionary definitions.

### 🔮 Future Work

* **End-to-End Multi-Task Learning Architecture:** Future iterations will replace the decoupled vision and NLP scripts with a unified, multimodal neural network. This architecture will process hair imagery, user questionnaires, and real-time environmental vectors simultaneously to calculate a single, highly continuous health optimization score.
* **On-Device Edge Deployment:** Transitioning the PyTorch model from an active cloud backend server down to a compressed **ONNX Runtime** or **PyTorch Mobile** format. This will enable real-time, zero-latency inference directly inside native mobile browsers or iOS/Android devices without relying on external cloud hosting infrastructure.
