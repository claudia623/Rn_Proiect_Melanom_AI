# ⚡ QUICK START GUIDE - ETAPA 4

**Time to setup:** ~10 minutes  
**Time to first classification:** ~5 minutes

---

## 📋 BEFORE YOU START

Asigură-te că ai:
- ✅ Python 3.10+ instalat
- ✅ Folder proiect `Rn_Proiect_Melanom_AI-main/` descărcat
- ✅ Terminal/Command Prompt deschis
- ✅ Internet pentru download dependencies

---

## 🚀 SETUP STEPS

### Step 1: Navigate to Project Directory
```bash
cd path/to/Rn_Proiect_Melanom_AI-main/
```

### Step 2: Install Dependencies (ONE TIME ONLY)
```bash
pip install -r requirements.txt
```

**Duration:** 5-10 minutes (depends on internet speed)  
**What it installs:**
- TensorFlow 2.15+
- Streamlit 1.28+
- OpenCV, NumPy, Pandas
- Albumentations for augmentation

### Step 3: Organize Images (IF NEEDED)
If ISIC images are in root directory, organize them:
```bash
python organize_images.py
```

**Output:**
- `data/raw/benign/` → 10 benign images
- `data/raw/malignant/` → 10 malignant images

---

## 🔧 RUN THE SYSTEM

### OPTION A: Quick Test (Recommended First Run)

#### Step 1: Generate Synthetic Data
```bash
python src/data_acquisition/generate_synthetic_data.py
```

**Output:** 
```
✅ Loaded 20 images from data/raw/
✅ Generated 40 synthetic images
✅ Metadata saved to data/generated/original/metadata.csv
✅ Data generation completed successfully!
```

**Duration:** ~1-2 minutes

#### Step 2: Create Neural Network Model
```bash
python src/neural_network/similarity_model.py
```

**Output:**
```
✅ Model created successfully!
✅ Model saved to models/similarity_model_untrained.h5
✅ MODUL 2 TEST COMPLETED SUCCESSFULLY
```

**Duration:** ~1 minute

#### Step 3: Launch Web UI
```bash
streamlit run src/app/streamlit_ui.py
```

**Output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

**Duration:** Immediate

---

### OPTION B: Full Pipeline Test

```bash
# All in one command (on same terminal, sequentially)
python src/data_acquisition/generate_synthetic_data.py && \
python src/neural_network/similarity_model.py && \
streamlit run src/app/streamlit_ui.py
```

---

## 🎨 USING THE WEB UI

### Step 1: Open Browser
Navigate to: **http://localhost:8501**

You should see:
```
🏥 Melanom AI - Similarity-Based Classification System
Automatic skin lesion classification: Benign vs Malignant

[Sidebar: System Info]        [Main: Upload Area]
```

### Step 2: Upload Test Image

1. Click **"Upload dermatoscopic image"** area
2. Select image from:
   - `data/raw/benign/` (for benign test)
   - `data/raw/malignant/` (for malignant test)
3. Image preview appears

### Step 3: Analyze

1. Click **"🎯 Analyze Image"** button
2. Wait for spinner (100-400ms)
3. See results:
   ```
   📋 Classification Result
   ✅ BENIGN  (or ⚠️ MALIGNANT)
   
   📊 Confidence
   82.3% (HIGH)
   
   📈 Similarity Scores
   Benign Match: 75.2%  | σ=8.3%
   Malignant: 30.1%    | σ=12.1%
   
   🖼️ Top Similar Reference Images
   [Image 1] [Image 2] [Image 3]
   ```

### Step 4: Check Logs

Open file: `logs/predictions.csv`

You'll see:
```csv
timestamp,filename,classification,benign_score,benign_std,...
2025-12-09T10:30:45,ISIC_0000000.jpg,BENIGN,0.75,0.08,...
```

### Step 5: Try Another Image

- Click "Clear" or upload new image
- System returns to IDLE
- Repeat Step 2-4

---

## ❌ COMMON ISSUES & FIXES

### Issue 1: "ModuleNotFoundError: No module named 'streamlit'"
```bash
# FIX:
pip install streamlit>=1.28.0
```

### Issue 2: "Model file not found"
```bash
# FIX: Run Modul 2 first
python src/neural_network/similarity_model.py
```

### Issue 3: "Reference images not loaded"
```bash
# FIX: Run Modul 1 first
python src/data_acquisition/generate_synthetic_data.py
```

### Issue 4: "Port 8501 already in use"
```bash
# FIX: Use different port
streamlit run src/app/streamlit_ui.py --server.port 8502
```

### Issue 5: "Image too blurry" error
```
This is NORMAL - system is validating image quality.
Simply upload a clearer image.
```

### Issue 6: Python not found
```bash
# Windows: try with 'py' instead of 'python'
py src/data_acquisition/generate_synthetic_data.py

# If still not working:
# 1. Install Python from python.org
# 2. Check: python --version (should be 3.10+)
# 3. Add to PATH if needed
```

---

## 📊 FOLDER STRUCTURE AFTER SETUP

After running all 3 modules, you should have:

```
Rn_Proiect_Melanom_AI-main/
│
├─ data/
│  ├─ raw/
│  │  ├─ benign/          [10 images]
│  │  └─ malignant/       [10 images]
│  ├─ generated/
│  │  └─ original/        [40 generated images]
│  │     ├─ benign/
│  │     ├─ malignant/
│  │     ├─ metadata.csv
│  │     ├─ augmentation_log.json
│  │     └─ generation_statistics.csv
│  └─ ...
│
├─ models/
│  └─ similarity_model_untrained.h5    [~20MB]
│
├─ logs/
│  └─ predictions.csv                  [predictions log]
│
├─ src/
│  ├─ data_acquisition/
│  ├─ neural_network/
│  ├─ app/
│  └─ ...
│
└─ [OTHER FILES]
```

---

## ✅ VERIFICATION CHECKLIST

After setup, verify:

- [ ] `data/generated/original/` has 40+ images
- [ ] `models/similarity_model_untrained.h5` exists (~20MB)
- [ ] Streamlit UI loads on http://localhost:8501
- [ ] Can upload image without errors
- [ ] Classification returns BENIGN/MALIGNANT
- [ ] logs/predictions.csv created and updated

---

## 📖 NEXT STEPS

### To Learn More:
1. Read `README_Etapa4_Arhitectura_SIA.md` (full specification)
2. Read `docs/STATE_MACHINE_DESCRIPTION.md` (architecture)
3. Check module README files:
   - `src/data_acquisition/README_Module1.md`
   - `src/neural_network/README_Module2.md`
   - `src/app/README_Module3.md`

### To Test More:
1. Upload different images (see how confidence changes)
2. Edit `src/app/streamlit_ui.py` and restart
3. Modify augmentation in `src/data_acquisition/generate_synthetic_data.py`
4. Check different reference images in `data/generated/original/`

### To Debug:
1. Check logs: `logs/predictions.csv`
2. Check generation: `data/generated/original/`
3. Check model: `models/similarity_model_untrained.h5`
4. Add `print()` statements in code

---

## ⏱️ TYPICAL TIMING

| Step | Time |
|------|------|
| Install dependencies | 5-10 min |
| Generate data (Modul 1) | 1-2 min |
| Create model (Modul 2) | 1 min |
| Start UI (Modul 3) | 10 sec |
| Upload + Analyze | 500ms |
| **TOTAL FIRST RUN** | **~10 min** |
| **Subsequent Runs** | **30 sec** |

---

## 🎯 SUCCESS INDICATORS

You'll know it's working when:

✅ `generate_synthetic_data.py` outputs "Data generation completed successfully!"  
✅ `similarity_model.py` outputs "MODUL 2 TEST COMPLETED SUCCESSFULLY"  
✅ Streamlit shows "Local URL: http://localhost:8501"  
✅ Upload area appears in browser  
✅ Can upload image without validation error  
✅ Click "Analyze" returns classification result  
✅ Predictions appear in `logs/predictions.csv`  

---

## 🆘 NEED HELP?

### Check These Files (in order):
1. **SUBMISSION_SUMMARY.md** - System overview
2. **TESTING_GUIDE_ETAPA4.md** - Detailed testing
3. **README_Etapa4_Arhitectura_SIA.md** - Full specification
4. **Module README files** - Specific module help

### Common Questions:

**Q: How long does analysis take?**  
A: 300-400ms (varies by CPU)

**Q: Can I use my own images?**  
A: Yes! Upload any JPG/PNG. System validates size (100-2048px) and blur.

**Q: Where are results saved?**  
A: logs/predictions.csv (local file, not uploaded anywhere)

**Q: Is my data private?**  
A: Yes! Everything runs locally. No cloud upload.

**Q: Can I run on GPU?**  
A: Yes! Install tensorflow-gpu. System auto-detects.

**Q: Why is accuracy ~50%?**  
A: Model is UNTRAINED (Etapa 4). Weights are ImageNet pretrained but not fine-tuned for melanoma. Etapa 5 will train it.

---

## 🚀 YOU'RE READY!

Proceed with steps above and you'll have a fully functional Melanoma AI system in ~10 minutes.

**Good luck! 🎉**

---

**Document:** QUICK_START_GUIDE.md  
**Version:** 0.4-architecture  
**Date:** 09.12.2025
