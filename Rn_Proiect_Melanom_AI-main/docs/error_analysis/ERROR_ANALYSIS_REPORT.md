
# RAPORT ANALIZA ERORI - ETAPA 5

## Rezumat Evaluare Test Set

- **Total imagini test:** 30
- **Imagini corecte:** 21 (70.0%)
- **Imagini gresite:** 9 (30.0%)

## Descompunere Erori

### False Positives (Benign clasificate ca Malignant)
- Count: 7
- Rata: 46.7% din imagini Benign
- **Implicatie:** Recomandare falsa de tratament / over-alarm

### False Negatives (Malignant clasificate ca Benign)
- Count: 2
- Rata: 13.3% din imagini Malignant
- **Implicatie:** CRITICA - cazuri maligne netestate (miss medical)

## Analiză Detaliat Confuzii

```
Confusion Matrix:
             Predicted Benign    Predicted Malignant
True Benign          8                     7
True Malignant       2                    13
```

## Metrici Derivate

- **Sensitivity (Recall pentru Malignant):** 86.7%
  - Abilitate de a detecta cazuri maligne reale
  
- **Specificity:** 53.3%
  - Abilitate de a identifica corect cazurile benigne
  
- **Precision (Positive Predictive Value):** 65.0%
  - Proporție predicții malignant care sunt corecte

## Cauze Probabili ale Erorilor

1. **Variații Iluminare:** Imagini cu iluminare neomogenă, reflexii
   - Soluție: Data augmentation cu jitter iluminare
   
2. **Artefacte Imagistică:** Glint (reflexii), markeri de linie
   - Soluție: Filtrare preprocessing specifică
   
3. **Similitudine Vizuală:** Leziuni benigne care imita melanom (ex: nevus)
   - Soluție: Colectare imagini specifice pentru cazuri atipice
   
4. **Distribuție Dataset Hibrid:** Mix imagini sintetice + reale
   - Soluție: Validare separata pe imagini reale vs sintetice

## Recomandări Îmbunătățire

### Prioritate 1 (Critic) - Reducere False Negatives
1. Ajustare prag (threshold) de 0.5 → 0.35
   - Favorizează recall pentru Malignant
   - Cost: creștere False Positives acceptabilă în context medical
   
2. Reantrenare cu `class_weights`
   - Weight malignant mai mare (ex: 3:1)
   
3. Augmentări specifice pentru zone atipice
   - Histograma egalizare pentru contrast variabil
   - Crop aleator din diferite regiuni

### Prioritate 2 - Date Suplimentare
1. Colectare ≥50 imagini noi pentru cazuri atipice
2. Validare separata pe ImageNet-Only pretrained vs Altă bază

### Prioritate 3 - Optimizare Model
1. Explorare alte arhitecturi (ResNet, DenseNet)
2. Ensemble de modele cu thresholds diferite
3. Feature matching + similarity-based (fallback)

## Imagini Problematice

Vezi dosarul `docs/error_analysis/` pentru:
- Top 5 False Positive examples
- Top 5 False Negative examples
- Annotări cu confidence scores și predicții

---

**Generat:** 12.01.2026  
**Script:** generate_etapa5_visualizations.py
