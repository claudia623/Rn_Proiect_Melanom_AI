# State Machine Diagram - Melanom AI Classification System
# Format: ASCII + descriere detaliată
# Versiune: Etapa 4
# Data: 09.12.2025

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    MELANOM AI - STATE MACHINE ARCHITECTURE                         │
│                      Similarity-Based Classification System                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │    START     │
                                    └──────┬───────┘
                                           │
                                           ↓
                        ┌──────────────────────────────────────┐
                        │           IDLE STATE                 │
                        │  Server ready, awaiting user input   │
                        │  • Streamlit server running          │
                        │  • Model loaded in memory            │
                        │  • Reference images cached           │
                        └──────────────────┬───────────────────┘
                                           │
                ┌──────────────────────────┼──────────────────────────┐
                │ [User uploads image]     │                          │
                ↓                          │                          ↓
     ┌────────────────────────┐            │        ┌────────────────────────┐
     │   VALIDATE_INPUT       │            │        │  ERROR_NO_FILE         │
     │ Check format, size,    │            │        │  (rare, timeout)       │
     │ blur quality           │            │        │  → back to IDLE        │
     └────────────┬───────────┘            │        └────────────────────────┘
                  │                        │
     ┌────────────┴──────────┬─────────────┘
     │ [Valid] ✓             │ [Invalid] ✗
     ↓                       ↓
┌────────────┐        ┌──────────────────────┐
│ PREPROCESS │        │  ERROR_INVALID_IMAGE │
│ • Resize   │        │  • Blur too high     │
│ • Normalize│        │  • Wrong dimensions  │
│ • RGB      │        │  • Unsupported format│
│   convert  │        │  → Display message   │
└─────┬──────┘        │  → back to IDLE      │
      │               └──────────────────────┘
      │
      ↓
┌─────────────────────────────────┐
│  FEATURE_EXTRACTION             │
│  • Load model: EfficientNetB0   │
│  • Input: 224x224 RGB image     │
│  • Output: 256D feature vector  │
│  • Duration: ~100-150ms         │
└────────┬────────────────────────┘
         │
         ↓
┌──────────────────────────────────────┐
│  LOAD_REFERENCE_DATABASE             │
│  • Load benign refs (15 images)      │
│  • Load malignant refs (15 images)   │
│  • Already cached from startup       │
└────────┬─────────────────────────────┘
         │
         ↓
┌──────────────────────────────────────┐
│  COMPUTE_SIMILARITY                  │
│  For each reference image:           │
│  • Extract 256D features             │
│  • Cosine similarity computation     │
│  • Similarity_test vs Similarity_ref │
│  • Aggregate per class               │
│  • Duration: ~50-100ms               │
└────────┬─────────────────────────────┘
         │
         ↓
    ┌────────────────────────────────────┐
    │   score_benign = mean(sim_benign)  │
    │   score_malignant = mean(sim_mal)  │
    │   confidence = |score_benign -     │
    │                score_malignant|    │
    └────────┬─────────────────────────┬─┘
             │                         │
    [score_benign > score_malignant]   │
             │                         │
             ↓                         ↓
    ┌─────────────────────┐  ┌────────────────────┐
    │ CLASSIFY_BENIGN     │  │ CLASSIFY_MALIGNANT │
    │ Label: "BENIGN"     │  │ Label: "MALIGNANT" │
    │ Color: Green ✅     │  │ Color: Red ⚠️      │
    └────────┬────────────┘  └────────┬───────────┘
             │                        │
             └────────────┬───────────┘
                          │
                          ↓
                ┌─────────────────────┐
                │  DISPLAY_RESULT     │
                │ • Show badge        │
                │ • Similarity %      │
                │ • Confidence score  │
                │ • Top 3 refs/class  │
                │ • Expandable stats  │
                └──────────┬──────────┘
                           │
                           ↓
                ┌─────────────────────┐
                │   LOG_RESULT        │
                │ • Save to CSV       │
                │ • Timestamp         │
                │ • Filename          │
                │ • Classification    │
                │ • Scores/Confidence │
                │ → logs/predictions  │
                └──────────┬──────────┘
                           │
                ┌──────────┴────────────┐
                │ [User click "Clear"]  │
                │ or [Auto timeout 30s] │
                ↓                       │
         ┌─────────────────────┐        │
         │  RETURN_TO_IDLE     │        │
         │ • Reset UI          │        │
         │ • Clear file buffer │        │
         │ • Clear results     │        │
         │ • Cleanup memory    │        │
         └─────────┬───────────┘        │
                   │                    │
                   └──────────┬─────────┘
                              │
                              ↓
                    ┌────────────────────┐
                    │    IDLE (Again)    │
                    │ Ready for next     │
                    │ classification     │
                    └────────┬───────────┘
                             │
                    [Loop back to User upload]
                             │
                ┌────────────┴─────────────┐
                │ [Server shutdown]        │
                │ OR [Emergency Stop]      │
                ↓                          ↓
    ┌──────────────────────┐   ┌──────────────────────┐
    │  SAFE_SHUTDOWN       │   │  ERROR_HANDLER       │
    │  • Close streams     │   │  • Log error         │
    │  • Save final logs   │   │  • Cleanup resources │
    │  • Free GPU memory   │   │  • Graceful exit     │
    │  • Exit 0            │   │  • Exit 1            │
    └──────────────────────┘   └──────────────────────┘
             │                          │
             └──────────────┬───────────┘
                            ↓
                        ┌────────┐
                        │  END   │
                        └────────┘
```

## STAĂRI ȘI TRANZIȚII DETALIATE

### State 1: IDLE
**Descriere:** Server Streamlit pornit și așteptă input de la utilizator.

**Intrări:**
- Model RN încărcat în memorie
- Reference images cached
- Streamlit session initialized

**Condiții de ieșire:**
- [User upload file] → VALIDATE_INPUT
- [Server shutdown signal] → SAFE_SHUTDOWN
- [Emergency stop] → ERROR_HANDLER

**Timp de ședere:** Nelimitat (server-side)

---

### State 2: VALIDATE_INPUT
**Descriere:** Verificare integritate imagine (format, dimensiuni, blur).

**Verificări:**
```python
1. File format: JPEG | PNG
2. File size: ≤ 10 MB
3. Image dimensions: 100x100 ≤ size ≤ 2048x2048
4. Blur detection: Laplacian variance ≥ 100
5. Image readability: cv2.imread() succeeds
```

**Condiții de ieșire:**
- [All checks pass] → PREPROCESS
- [Any check fails] → ERROR_INVALID_IMAGE

**Timp de ședere:** ~20-50ms

---

### State 3: PREPROCESS
**Descriere:** Standardizare imagine pentru RN.

**Operații:**
```python
1. Resize: original → 224x224 (bilinear interpolation)
2. Color conversion: BGR → RGB
3. Normalizare: [0-255] → [0-1]
4. Data type: uint8 → float32
```

**Output:** Imagine preprocessată ready pentru RN

**Condiții de ieșire:**
- [Preprocess success] → FEATURE_EXTRACTION

**Timp de ședere:** ~10-20ms

---

### State 4: FEATURE_EXTRACTION
**Descriere:** Extragere vector features cu RN.

**Arhitectură:**
```
Input (224x224x3)
  ↓
EfficientNetB0 frozen backbone
  ↓
GlobalAveragePooling (→ 1280D)
  ↓
Dense(256, ReLU)
  ↓
L2 Normalization
  ↓
Output: 256D features (norm ≈ 1.0)
```

**Computație:**
- Forward pass: ~80-150ms (CPU)
- Batch size: 1

**Condiții de ieșire:**
- [Feature extraction success] → LOAD_REFERENCE_DATABASE
- [Error: OOM memory] → ERROR_HANDLER

**Timp de ședere:** ~100-150ms

---

### State 5: LOAD_REFERENCE_DATABASE
**Descriere:** Încarcă imagini referință din cache.

**Date:**
- 15 imagini benigne
- 15 imagini maligne
- Total: 30 imagini referință

**Cache:**
- Pre-loaded la startup (@st.cache_data)
- Nu se reîncarcă la fiecare clasificare

**Condiții de ieșire:**
- [References loaded] → COMPUTE_SIMILARITY
- [References missing] → ERROR_NO_REFERENCES

**Timp de ședere:** ~5ms (cached)

---

### State 6: COMPUTE_SIMILARITY
**Descriere:** Calculează similaritate cu toate imaginile referință.

**Algoritm:**
```python
for each reference image R_i:
    features_R = model(R_i)  # 256D
    similarity_i = cosine(features_test, features_R)
    # Output: [0, 1] (1 = identical, 0 = orthogonal)

similarities_benign = [sim_b1, sim_b2, ..., sim_b15]
similarities_malignant = [sim_m1, sim_m2, ..., sim_m15]
```

**Metrici computate:**
- Mean, Std, Min, Max per clasă

**Computație:**
- 30 forward passes: ~100-150ms
- 30 cosine computations: ~50ms

**Condiții de ieșire:**
- [Similarity computation success] → CLASSIFY_BENIGN/MALIGNANT
- [Error in computation] → ERROR_HANDLER

**Timp de ședere:** ~150-200ms

---

### State 7a: CLASSIFY_BENIGN
**Descriere:** Imagen classified ca BENIGN.

**Condiție:**
```
score_benign > score_malignant
```

**Confidence:**
```
confidence = score_benign - score_malignant
Range: [0, 1]
Interpretation:
  > 0.7: HIGH confidence
  0.3-0.7: MEDIUM confidence
  < 0.3: LOW confidence (ambiguous)
```

**UI Display:**
- Badge: "✅ BENIGN" (verde)
- Metrics: benign% | malignant%
- Recommendation: Routine follow-up (if confidence high)

**Condiții de ieșire:**
- [Classification confirmed] → DISPLAY_RESULT

---

### State 7b: CLASSIFY_MALIGNANT
**Descriere:** Imagine classified ca MALIGNANT.

**Condiție:**
```
score_malignant ≥ score_benign
```

**Confidence:** Same as 7a

**UI Display:**
- Badge: "⚠️ MALIGNANT" (roșu)
- Warning: "Consult dermatologist"
- Reference: Most similar malignant images

**Condiții de ieșire:**
- [Classification confirmed] → DISPLAY_RESULT

---

### State 8: DISPLAY_RESULT
**Descriere:** Afișare rezultate în UI.

**Elemente UI:**
1. **Classification Badge:** Text + culoare (verde/roșu)
2. **Confidence Metric:** Percentage + textual ("HIGH/MEDIUM/LOW")
3. **Similarity Scores:** Benign % | Malignant %
4. **Statistics:** Mean, Std, Min, Max (expandable)
5. **Reference Grid:** Top 3 benign + Top 3 malignant
6. **Expandable Details:** Full statistics table

**Timp de afișare:** Real-time (< 100ms)

**Condiții de ieșire:**
- [Automated timeout 30s] → LOG_RESULT → RETURN_TO_IDLE
- [User clicks "Clear"] → LOG_RESULT → RETURN_TO_IDLE

**Timp de ședere:** 0-30 secunde (user-dependent)

---

### State 9: LOG_RESULT
**Descriere:** Salvare predicție în CSV pentru audit.

**CSV Fields:**
```csv
timestamp,filename,classification,benign_score,benign_std,
malignant_score,malignant_std,confidence
2025-12-09T10:30:45,patient_001.jpg,MALIGNANT,0.35,0.05,
0.65,0.08,0.30
```

**Fișier:** `logs/predictions.csv`

**Scopul:** Audit clinic (doctor review, future model improvement)

**Condiții de ieșire:**
- [Log saved successfully] → RETURN_TO_IDLE
- [Log write error] → Log warning, still → RETURN_TO_IDLE (graceful)

**Timp de ședere:** ~5-10ms

---

### State 10: RETURN_TO_IDLE
**Descriere:** Reset UI pentru următoarea clasificare.

**Operații:**
```python
1. Clear uploaded file from buffer
2. Clear displayed results
3. Reset session state variables
4. Free GPU memory (if available)
5. Refresh UI components
```

**Condiții de ieșire:**
- [Cleanup complete] → IDLE

**Timp de ședere:** ~10-20ms

---

### Error State: ERROR_INVALID_IMAGE
**Descriere:** Imagine nu trece validare.

**Mesaje Posibile:**
- "❌ Image too blurry (score: X). Please retake photo."
- "❌ Image too small: 80x80. Min 100x100 required."
- "❌ Image too large: 3000x2000. Max 2048x2048 allowed."
- "❌ File size too large: 12.5MB > 10MB."
- "❌ Invalid image format. Use JPG or PNG."

**Acțiune:** Display error message + button "Try Again"

**Condiții de ieșire:**
- [User re-uploads] → VALIDATE_INPUT
- [User cancels] → IDLE

---

### Error State: ERROR_HANDLER
**Descriere:** Erori neprevăzute (OOM, network, etc.).

**Loggare:**
```python
logger.error(f"Critical error: {exception_details}")
```

**UI Message:** "⚠️ System error occurred. Please try again or contact support."

**Graceful Recovery:**
- Close open resources
- Log stack trace
- Return to IDLE (not crash)

**Condiții de ieșire:**
- [Cleanup complete] → IDLE
- [Critical failure] → SAFE_SHUTDOWN

---

### Exit State: SAFE_SHUTDOWN
**Descriere:** Server shutdown controlat.

**Operații:**
```python
1. Close Streamlit session
2. Unload model from memory
3. Save final logs
4. Close file handles
```

**Exit Code:** 0 (success)

---

## TRANZIȚII CRITICE (Safety-Critical)

### Tranziție 1: VALIDATE_INPUT → ERROR_INVALID_IMAGE
**Când:** Blur detection fail
**Importanță:** Imagini medicale de calitate proastă → diagnostic eronat
**Implementare:** Laplacian variance < 100 → Reject
**Retry:** User poate reface poza

### Tranziție 2: COMPUTE_SIMILARITY → CLASSIFY_BENIGN/MALIGNANT
**Când:** Score aggregation
**Importanță:** Decizia clasificării → Tratament pacient
**Implementare:** mean(similarities) per clasă
**Threshold:** Nicio threshold hard (confidence relativ)

### Tranziție 3: Any State → ERROR_HANDLER
**Când:** Exception neprevăzută
**Importanță:** Stabilitatea sistemului
**Implementare:** Try-except blocks, graceful recovery
**Recovery:** Log error, return to IDLE

### Tranziție 4: DISPLAY_RESULT → LOG_RESULT
**Când:** Predicție confirmată
**Importanță:** Audit trail (HIPAA compliance)
**Implementare:** Save to CSV cu timestamp
**Backup:** Log error message if write fails

---

## FLUX DE CONTROL - CASOS DE FOLOSINȚĂ

### Caz 1: Happy Path (Imagine validă, clasificare rapidă)
```
IDLE → VALIDATE_INPUT [pass] → PREPROCESS → FEATURE_EXTRACTION →
LOAD_REFERENCE_DATABASE → COMPUTE_SIMILARITY → CLASSIFY_BENIGN →
DISPLAY_RESULT → LOG_RESULT → RETURN_TO_IDLE → IDLE
Timp total: ~300-400ms
```

### Caz 2: Imagine Blurry (Rejecare)
```
IDLE → VALIDATE_INPUT [blur fail] → ERROR_INVALID_IMAGE →
Display "Image too blurry" → [User retakes photo] →
VALIDATE_INPUT [pass] → PREPROCESS → ... → IDLE
Timp total: ~50ms + user time
```

### Caz 3: Low Confidence Classification
```
... → COMPUTE_SIMILARITY → confidence < 0.3 →
CLASSIFY_[BENIGN/MALIGNANT] but confidence LOW →
DISPLAY_RESULT [warning "Low confidence - manual review recommended"] →
LOG_RESULT → IDLE
```

### Caz 4: System Error (OOM)
```
... → FEATURE_EXTRACTION [OOM] → ERROR_HANDLER →
Log error, cleanup → SAFE_SHUTDOWN
(Server restart needed)
```

---

## IMPLEMENTARE STATE MACHINE

Cod Pseudocode:

```python
class MelanomaClassifier:
    def __init__(self):
        self.state = "IDLE"
        self.model = load_model()
        self.references = load_references()
    
    def transition(self, event):
        if self.state == "IDLE" and event == "USER_UPLOAD":
            self.state = "VALIDATE_INPUT"
        elif self.state == "VALIDATE_INPUT" and event == "VALID":
            self.state = "PREPROCESS"
        elif self.state == "VALIDATE_INPUT" and event == "INVALID":
            self.state = "ERROR_INVALID_IMAGE"
        # ... many more transitions
        elif any_state and event == "ERROR":
            self.state = "ERROR_HANDLER"
        
        return self.state
    
    def execute_state(self):
        if self.state == "IDLE":
            self.idle()
        elif self.state == "VALIDATE_INPUT":
            self.validate_input()
        # ... execute methods per state
```

---

## METRICI MONITORING

| Metric | Target | Unit |
|--------|--------|------|
| Total Latency | < 500ms | ms |
| Validation Time | < 50ms | ms |
| Feature Extraction | < 150ms | ms |
| Similarity Compute | < 150ms | ms |
| Classification Accuracy (Etapa 5) | > 85% | % |
| System Uptime | > 99% | % |
| Error Recovery Time | < 2s | s |

---

**Versiune:** 0.4-architecture  
**Data:** 09.12.2025  
**Autor:** Dumitru Claudia-Stefania  
**Status:** Implementat (Etapa 4)
