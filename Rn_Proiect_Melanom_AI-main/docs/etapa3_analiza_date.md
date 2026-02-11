# Documentatie Dataset - Melanom AI

Acest director contine documentatia detaliata a setului de date utilizat in proiect.

## Structura Folderului

docs/datasets/
  README.md                     - Prezent
  data_dictionary.md            - Descriere atribute
  collection_process.md         - Cum au fost colectate datele
  preprocessing_steps.md        - Pasi preprocesare
  quality_metrics.md            - Metrici calitate

## Referinta Rapida

Dataset Actual:
- Imagini procesate: 206 imagini
- Benign: 101 imagini (49%)
- Malignant: 101 imagini (51%)
- Split: 70% train, 15% validation, 15% test
- Dimensiune: 224x224 pixeli
- Format: JPEG

Sursa de Date:
- ISIC Archive (https://www.isic-archive.com/)
- Dataset public pentru cercetare si educatie
- Validare: Clinica si histopatologica

## Documente Disponibile

1. Data Dictionary
Descriere detaliata a fiecarui atribut al imaginilor:
- Pixeli RGB
- Dimensiuni
- Contrast
- Textura
- Forma
- Culoare

2. Collection Process
Cum au fost colectate si achizitie datele:
- Sursa ISIC Archive
- Perioada colectarii
- Etichetare si validare

3. Preprocessing Steps
Pasi de preprocesare aplicati:
- Curatare si validare
- Transformari (resize, normalize)
- Operatii morfologice
- Augmentare date

4. Quality Metrics
Metrici de calitate dataset:
- Distributie clase
- Balance train/val/test
- Artefacte si probleme

## Link-uri Importante

- Data Folder: ../../data/
- Raw Data: ../../data/raw/
- Processed Data: ../../data/processed/
- Train/Val/Test: ../../data/train/, ../../data/validation/, ../../data/test/

Dataset Echilibrat: 50/50 benign/malignant
Preprocessat Complet: Eliminare par, CLAHE, augmentare
Bine Documentat: Fiecare pas este explicat
 **Ușor de Utilizat:** Split perfect train/val/test  

---

**Actualizat:** 20 ianuarie 2026
