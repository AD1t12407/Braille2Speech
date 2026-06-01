# 🔵 Braille-to-Speech System
### Camera-Based English Braille Recognition with Gemini AI + gTTS

> **Convert a photo of a Braille document into spoken audio** — using classical computer vision for dot detection, Google Gemini for OCR error correction, and gTTS for natural speech output.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Pipeline Architecture](#-pipeline-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Step-by-Step Breakdown](#-step-by-step-breakdown)
- [File Structure](#-file-structure)
- [Tech Stack](#-tech-stack)
- [Known Limitations](#-known-limitations)
- [Evaluation Metrics](#-evaluation-metrics)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

##  Overview

This Jupyter Notebook implements a full end-to-end **Braille-to-Speech** pipeline that:

1. Accepts a **photo of an embossed Braille document** (JPG or PNG)
2. Preprocesses the image to isolate Braille dots
3. Detects and segments individual Braille cells using OpenCV
4. Decodes the Braille dot patterns into raw English text
5. Corrects OCR errors using **Google Gemini AI**
6. Converts the final text into an **MP3 audio file** using Google Text-to-Speech (gTTS)

The system is designed to run entirely inside **Google Colab** with no local setup required, making it accessible for rapid prototyping and experimentation.

---

## Pipeline Architecture

The notebook is organized into **12 sequential steps**, each building on the previous:

| Step | Name | Description |
|------|------|-------------|
| 0 | Install Dependencies | Installs all required Python packages |
| 1 | Configuration | Sets API key, output path, language, and model |
| 2 | Imports & Braille Dictionary | Loads Grade-1 Braille → character mappings |
| 3 | Image Input & Upload | Accepts uploaded image or generates synthetic test |
| 4 | Image Preprocessing | Grayscale, deskew, CLAHE, Gaussian blur, adaptive threshold |
| 5 | Braille Dot Detection (v1) | Initial blob detection with neighbor validation |
| 6 | Refined Dot Detection & Cell Segmentation | Circularity + NMS filtering, 3×2 cell grouping |
| 7 | Pattern Recognition & Decoding | Converts dot patterns to text via `BRAILLE_MAP` |
| 8 | AI Error Correction | Gemini-powered OCR correction |
| 9 | Text-to-Speech | gTTS MP3 generation and inline playback |
| 10 | Results Dashboard | Full visual summary of all pipeline stages |
| 11 | Evaluation Metrics | CER and WER against ground truth |
| 12 | One-Shot Pipeline Runner | `run_full_pipeline()` convenience wrapper |



## 🔬 Step-by-Step Breakdown

###  Braille Dictionary (`BRAILLE_MAP`)

The dictionary encodes Grade-1 English Braille using **6-bit binary strings**. Each bit corresponds to one of the 6 dot positions in a Braille cell:

```
Dot Layout:
  1 • • 4
  2 • • 5
  3 • • 6

Pattern string = d1 d2 d3 d4 d5 d6
  '1' = raised dot
  '0' = flat (no dot)

Examples:
  'a' → '100000'  (only dot 1 raised)
  'b' → '110000'  (dots 1 and 2 raised)
  'l' → '111000'  (dots 1, 2, 3, 4 raised)
```

> ⚠️ **Known Issue:** The current `BRAILLE_MAP` contains **duplicate keys** (same 6-bit pattern mapped to multiple characters). In Python dictionaries, later entries silently overwrite earlier ones. This is a known source of decoding ambiguity and should be refactored in future versions.

---

### Image Preprocessing

The `preprocess_braille_image()` function applies a 6-stage pipeline:

| Stage | Method | Purpose |
|-------|--------|---------|
| Grayscale | `cv2.cvtColor` | Removes color noise |
| Resize | `cv2.resize` | Caps max dimension at 1600px |
| Deskew | Hough line detection + rotation | Corrects camera tilt (up to ±30°) |
| CLAHE | `cv2.createCLAHE(clipLimit=3.0)` | Enhances local contrast on dot shadows |
| Gaussian Blur | `cv2.GaussianBlur(3×3)` | Smooths noise before thresholding |
| Adaptive Threshold | `cv2.ADAPTIVE_THRESH_GAUSSIAN_C` | Binarizes dots from background |

---

###  Dot Detection

The refined `detect_braille_dots()` function uses a three-stage approach:

**1. SimpleBlobDetector**
```python
params.filterByArea       = True   # 30–1000 px²
params.filterByCircularity = True  # minCircularity = 0.7
```
Detects circular blobs (raised Braille dots) and rejects irregular shapes like smudges or paper texture.

**2. Non-Maximum Suppression (NMS)**
Removes duplicate detections caused by shadow artifacts:
```python
# Dots within 1.5× radius of a larger dot are suppressed
candidates = [c for c in candidates
              if np.hypot(c[0]-curr[0], c[1]-curr[1]) > curr[2] * 1.5]
```

**3. Neighbor Validation**
Keeps only dots that have at least 2 neighbors within 80px — ensuring isolated noise blobs are discarded while real Braille clusters are preserved.

---

### Cell Segmentation

`segment_braille_cells()` groups detected dots into 3-row × 2-column Braille cells:

1. **Estimate dot spacing** using nearest-neighbor distances between dots
2. **Cluster dots into rows** using y-coordinate proximity (tolerance = 60% of dot pitch)
3. **Group rows into triplets** — consecutive rows within 2.5× dot pitch belong to the same Braille cell row
4. **Pair left/right columns** within each triplet to form complete cells
5. **Assign dot positions 1–6** based on row (1–3 = left column, 4–6 = right column)

---

### Decoding

`decode_braille_cells()` converts cell dot patterns to text:

- **Empty cell** (`'000000'`) → space character
- **Capital indicator** (`'000001'`) → capitalizes the next decoded character
- **Number indicator** (`'010111'`) → activates number mode for subsequent characters
- **Everything else** → looked up in `BRAILLE_MAP`; unknown patterns return `'?'`

---

###  Gemini AI Correction

The Gemini prompt instructs the model to:
- Fix spelling errors from misidentified dot patterns
- Restore missing characters from undetected dots
- Rejoin words broken by segmentation errors
- Correct punctuation
- **Preserve original meaning** — no hallucination of new content

If no API key is provided, this step is silently skipped and raw OCR text is used.

---

###  Text-to-Speech

Uses **gTTS (Google Text-to-Speech)** to convert the final text to MP3:

```python
tts = gTTS(text=corrected_text, lang='en', slow=False)
tts.save("braille_output.mp3")
```

The audio plays inline in the notebook via `IPython.display.Audio`. Requires active internet connection.

---

## 📁 File Structure

```
braille_to_speech/
│
├── Copy_of_braille_to_speech__1_.ipynb   # Main notebook
├── braille_output.mp3                     # Generated audio (after running)
├── braille_dashboard.png                  # Pipeline dashboard (after running)
└── README.md                              # This file
```

---

## 🛠️ Tech Stack

| Component | Library / Service | Version |
|-----------|-------------------|---------|
| Image processing | OpenCV (`cv2`) | 4.x |
| Numerical computation | NumPy | 1.x |
| Image analysis | scikit-image | 0.x |
| Blob detection | `cv2.SimpleBlobDetector` | — |
| Spatial indexing | `scipy.spatial.KDTree` | — |
| Visualization | Matplotlib | 3.x |
| AI OCR correction | Google Gemini (`gemini-1.5-flash`) | API |
| Text-to-Speech | gTTS | 2.x |
| Notebook UI widgets | ipywidgets | 7.x / 8.x |
| Runtime | Google Colab / Jupyter | — |

---

## ⚠️ Known Limitations

### 1. Duplicate Keys in `BRAILLE_MAP`
The dictionary has many conflicting mappings where the same 6-bit pattern is assigned to multiple characters (e.g., `'100000'` maps to both `'a'` and `'1'` and several Grade 2 contractions). Python silently keeps only the last definition, making some characters undecodable. **Fix:** Separate Grade-1 letters, numbers, and Grade-2 contractions into distinct dictionaries with context-aware selection.

### 2. No True Grade-2 Braille Support
Grade-2 Braille uses 180+ contractions (e.g., `the`, `and`, `ing`). The current implementation includes some Grade-2 patterns in the flat dictionary but they conflict with Grade-1 patterns. A proper Grade-2 decoder requires a context-aware state machine.

### 3. Lighting Sensitivity
Dot detection depends heavily on shadows cast by raised dots. Poor or flat lighting causes missed detections, directly impacting decoding accuracy.

### 4. Fixed Dot Spacing Assumptions
The cell segmentation uses median nearest-neighbor distances to estimate dot pitch. This works for standard Braille but may fail on:
- Non-standard Braille embossers
- Heavily worn or low-relief documents
- Images taken at an angle

### 5. Single Row of Braille Only
The pipeline is primarily validated on single-line Braille. Multi-line documents may see segmentation errors at row boundaries due to the triplet-grouping heuristic.

### 6. gTTS Requires Internet
gTTS relies on Google's servers and will fail offline. For offline TTS, see [Future Improvements](#-future-improvements).

---

## 📊 Evaluation Metrics

The notebook computes two standard speech/OCR metrics in **Step 11**:

### Character Error Rate (CER)
```
CER = edit_distance(hypothesis, reference) / len(reference)
```
Measures character-level accuracy. Lower is better. A CER of 0.0 = perfect match.

### Word Error Rate (WER)
```
WER = edit_distance(hyp_words, ref_words) / len(ref_words)
```
Measures word-level accuracy. Lower is better.

To use these metrics meaningfully, update `GROUND_TRUTH` in Step 11:
```python
GROUND_TRUTH = "hello world"   # ← set this to the known text of your image
```

The notebook prints a comparison table showing raw OCR vs. AI-corrected performance, along with the CER improvement from Gemini correction.

---

## 🔮 Future Improvements

| Area | Proposed Improvement |
|------|----------------------|
| **Dot detection** | Train a lightweight YOLOv8-nano model on a labeled Braille dot dataset for robust detection across lighting conditions |
| **Cell segmentation** | Use Hough Transform for baseline detection to handle rotated pages and multi-column layouts |
| **Grade-2 Braille** | Implement a context-aware state machine to handle all 180+ Grade-2 contractions correctly |
| **Real-time** | Process webcam frames with OpenCV + temporal averaging for stable live decoding |
| **Mobile deployment** | Convert to TFLite / Core ML for on-device processing via Flutter or Swift |
| **TTS quality** | Swap gTTS for [Coqui TTS](https://github.com/coqui-ai/TTS) (offline) or ElevenLabs (high quality) |
| **Web UI** | Build a Gradio or Streamlit app with drag-and-drop image upload and real-time audio playback |
| **Multilingual Braille** | Extend to Arabic, French, Hindi, and Unified English Braille (UEB) |
| **Image quality guide** | Use focus/blur metrics to prompt the user to retake a photo if quality is too low |
| **Gemini Vision bypass** | Send the preprocessed image directly to Gemini Vision API to skip classical CV entirely |
| **BRAILLE_MAP fix** | Separate letter/number/punctuation/Grade-2 maps with context-aware dispatch |

---

## 🤝 Contributing

Contributions are welcome! Some high-priority areas:

- **Fix the `BRAILLE_MAP` duplicate key issue** — the most impactful single fix
- **Add multi-line Braille support** — improve the triplet-grouping heuristic in `segment_braille_cells()`
- **Improve deskew** — the Hough-based method sometimes over-rotates on text-heavy images
- **Add unit tests** — especially for `cell_to_pattern()` and `decode_braille_cells()`

To contribute:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/fix-braille-map`
3. Commit your changes
4. Open a pull request with a clear description

---

## 📄 License

This project is open-source. See the repository for license details.

---

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/) — computer vision backbone
- [Google Gemini](https://deepmind.google/technologies/gemini/) — AI OCR correction
- [gTTS](https://github.com/pndurette/gTTS) — text-to-speech
- [scikit-image](https://scikit-image.org/) — image analysis utilities
- The open Braille standardization community for Grade-1 and Grade-2 encoding references

---

*Built with OpenCV · NumPy · scikit-image · Google Gemini · gTTS*
