# Comparing TabPFN, TabICL and KNN

This project investigates and compares three approaches to tabular classification:

- **TabPFN** – Prior-Data Fitted Network
- **TabICL** – In-Context Learning for tabular data
- **KNN** – K-Nearest Neighbors baseline

The project evaluates model performance on multiple datasets and investigates how prediction performance changes with different context sizes and experimental settings.

## 1. Prerequisites

- Python 3.11 (recommended, subject to package compatibility)
- Git
- pip, the Python package installer



## 2. Clone the repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/sofialindhh/DeepMachineLearning.git
cd DeepMachineLearning
```


## 3. Create a virtual environment

A virtual environment keeps the project's Python dependencies separate from other Python projects.

On macOS:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

After activation, your terminal should show `(.venv)`.

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Whenever you return to the project in a new terminal session, activate the environment again:

```bash
source .venv/bin/activate
```

To deactivate it:

```bash
deactivate
```

## 4. Install dependencies

With the virtual environment activated, install the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

**Note:** TabPFN and TabICL may have specific Python, PyTorch, hardware, or package-version requirements. If installation fails, check the respective package documentation and ensure the selected versions are compatible.

## 5. Datasets

The datasets used in the experiments are stored in the `data/` directory in this repository.

```text
DeepMachineLearning/
├── data/
│   ├── ...
├── main.py
├── requirements.txt
└── README.md
```

After cloning the repository, check that the `data/` directory is present and contains the required dataset files.

No separate dataset download is required, provided that the files are included in the cloned repository.

The project uses a data-loading function in `main.py` to load the datasets. Make sure that the paths in the data-loading code point to the correct files in the `data/` directory.

If a dataset cannot be found, check the file name, file extension, and relative path used by the data loader.

