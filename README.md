# P7 – Handwritten Sudoku Reader and Solver

A pipeline that reads a handwritten Sudoku grid and solves it:

1. **CNN (PyTorch)** recognises the digit in each of the 81 cells (trained on MNIST).
2. **Conflict check (AC-3)** detects duplicate digits in rows, columns and boxes.
3. **Search (UCS / Greedy / A\*)** finds the most likely correction of misread cells,
   using the CNN's probabilities as costs.
4. **CSP solver (Backtracking / Forward Checking / AC-3)** solves the corrected puzzle.

Test puzzles are mosaics built from real MNIST **test** digits, so the CNN never sees them during training.

---

## Project structure

```
P7/
├── app.py               # Streamlit demo
├── config.py            # paths and default settings
├── requirements.txt     # pinned dependencies
├── csp/                 # Sudoku CSP: BT, FC, AC-3, MRV/LCV
├── data/                # MNIST preprocessing, puzzle generator, sample puzzles
├── eval/                # evaluation and plots
├── integration/         # recognition → correction search → CSP pipeline
└── nn/                  # CNN model, training, prediction
```

---

## Environment setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/P7-Handwritten-Sudoku.git
cd P7-Handwritten-Sudoku

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 3. Install the pinned dependencies
python -m pip install -r requirements.txt
```

---

## Run commands

Run all commands from the project root (the folder that contains `app.py`).

### 1. Prepare the data and sample puzzles
Downloads/preprocesses MNIST and generates the test puzzles in `data/puzzles/`.
```bash
python -m data.make_data
```

### 2. Train the CNN
<!-- EDIT: replace with your real training command -->
```bash
python -m nn.train
```
Skip this step if the trained model file is already included in the repository.

### 3. Run the evaluation
<!-- EDIT: replace with your real evaluation command, and list the output files/plots -->
```bash
python -m eval.run_eval
```

### 4. Launch the demo
```bash
python -m streamlit run app.py
```
The app opens at **http://localhost:8501**.

---

## Using the demo

**Input** (sidebar)
- **Generated test puzzle**: one of the saved puzzles from `data/puzzles/`, with ground truth.
- **New random puzzle**: generates a fresh unique puzzle from a seed and number of givens.
- **Upload image**: a clean, cropped Sudoku image (PNG/JPG). No ground truth is available in this mode.

**Settings**
- **Inject misclassifications** (0–5): simulates CNN errors to show the correction search working.
- **Search algorithm**: UCS, Greedy or A\*.
- **CSP algorithm**: BT+FC+AC-3, BT+FC or BT.
- **Use NN probabilities**: ON uses −log(probability) costs; OFF uses a uniform cost of 1 per correction.
- **Max corrections / Confidence threshold / Unique solution**: limits for the correction search.

Press **Run correction & solve** to see the detected conflicts, the proposed corrections and their costs,
the solved grid, search and CSP statistics, and a comparison of the three CSP algorithms.

---

## Author

**Shahd Yasser**: sole author; designed and implemented all components (CNN, correction search, CSP solver, integration, evaluation and Streamlit demo).
