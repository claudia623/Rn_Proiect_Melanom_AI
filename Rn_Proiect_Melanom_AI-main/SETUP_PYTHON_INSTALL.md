# 🔧 SETUP INSTRUCȚIUNI - Python & Aplikație

## ⚠️ PROBLEMĂ: Python nu este instalat corect

Sistemul dă eroare: `Python was not found; run without arguments to install from the Microsoft Store`

Aceasta înseamnă că **trebuie să instalezi Python manual** de pe internet.

---

## 📥 INSTALARE PYTHON 3.11+ (NECESAR)

### Pas 1: Download Python

1. **Deschide browser și mergi la:** https://www.python.org/downloads/
2. **Click pe butonul mare "Download Python 3.12.x"** (sau versiune mai nouă)
3. **Descarcă fișierul `.exe`** pentru Windows

### Pas 2: Instalare Python

1. **Deschide fișierul descărcat** (ex: `python-3.12.0-amd64.exe`)
2. **ÎN FOAIA DE SETUP, BIFEAZĂ:** 
   - ✅ `Add Python 3.12 to PATH` (IMPORTANT!)
   - ✅ `Install pip` (package manager)
3. **Click `Install Now`**
4. **Așteaptă instalare (2-5 min)**
5. **Click `Close` după finalizare**

### Pas 3: Verific Instalare

Deschide **PowerShell** și rulează:
```powershell
python --version
pip --version
```

**Output așteptat:**
```
Python 3.12.x (sau versiune mai nouă)
pip 23.x (sau versiune mai nouă)
```

---

## 📦 INSTALARE DEPENDENȚE

După ce Python e instalat, deschide **PowerShell** și mergi în folder proiect:

```powershell
cd "c:\Users\Claudia Dumitru\Rn_Proiect_Melanom_AI\Rn_Proiect_Melanom_AI-main"

# Instalează pachete necesare
pip install -r requirements.txt
```

**SAU instalează manual:**
```powershell
pip install streamlit tensorflow numpy pandas pillow opencv-python matplotlib scikit-learn
```

**Durată:** 10-30 minute (prima dată, TensorFlow e mare)

---

## 🚀 RULARE APLICAȚIE

Apoi rulează:
```powershell
cd "c:\Users\Claudia Dumitru\Rn_Proiect_Melanom_AI\Rn_Proiect_Melanom_AI-main"
streamlit run src/app/streamlit_ui.py
```

**Așteptări:**
1. ⏳ **Streamlit se inițializează** (5-10 sec)
2. 🌐 **Browser deschide automat** la `http://localhost:8501`
3. ✅ **Aplicație gata de utilizare!**

---

## 🎯 ALTERNATIVĂ RAPIDĂ (Dacă vrei)

Dacă nu vrei să instalezi Python manual, poți folosi:

### **Anaconda Distribution** (Alternativă mai ușoară)
1. Download de la https://www.anaconda.com/download
2. Instalare (include Python + pip + pachete)
3. Rulează din Anaconda Prompt aceeași comandă

---

## ✅ CHECKLIST

- [ ] Download Python 3.11+ de pe python.org
- [ ] Instalare Python (cu "Add to PATH")
- [ ] Verificare: `python --version` funcționează
- [ ] Instalare pachete: `pip install -r requirements.txt`
- [ ] Rulare app: `streamlit run src/app/streamlit_ui.py`
- [ ] Browser deschide http://localhost:8501

---

## 📞 DACĂ RĂMÂI BLOCAT

1. **Verific că pip funcționează:**
   ```powershell
   pip --version
   ```

2. **Verific că streamlit e instalat:**
   ```powershell
   pip install streamlit
   ```

3. **Verific că tensorflow e instalat:**
   ```powershell
   pip install tensorflow
   ```

4. **Rulează cu verbose pentru debug:**
   ```powershell
   streamlit run src/app/streamlit_ui.py --logger.level=debug
   ```

---

**Status:** ⏳ ASTEAPTA INSTALARE PYTHON  
**Timp estimat:** 20-30 min (download + instalare)
