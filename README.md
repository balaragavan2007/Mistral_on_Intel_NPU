# Mistral 7B on Intel NPU with OpenVINO™ GenAI

This project demonstrates running **Mistral 7B** on an **Intel Ultra Series 1, 2 & 3 NPU** using **OpenVINO™ GenAI**.  
The model is optimized with **INT4 quantization**, making it faster ⚡ and smaller 🚀.

---

## ✨ Features
- Runs **Mistral LLM** locally on Intel NPU
- Uses **OpenVINO™ GenAI**
- **INT4 quantization** for efficient inference
- Works **offline** (no GPU required)
- Built with **Hugging Face Transformers + Optimum-Intel**

---

## 📦 Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/balaragavan2007/Mistral_on_Intel_NPU.git
cd Mistral_on_Intel_NPU
python -m venv llm
llm\Scripts\activate
pip install -r requirements.txt
```


---


## ▶️ Usage

- Run inference with:

```bash
python run.py
```


---


## Performance

Running the Mistral 7B model on my [ASUS Vivobook S16 OLED with Intel Core Ultra 5 125H], I achieved the following performance:

| Device | Performance      |
|--------|------------------|
| CPU    | [7.8] tok/s |
| GPU    | [15.91] tok/s |
| NPU    | [7.7] tok/s |


---


## 📸 Demo
<img width="2113" height="1413" alt="image" src="https://github.com/user-attachments/assets/b92f6cbf-8cc0-40fa-8ddf-41af25d946e5" />


---
