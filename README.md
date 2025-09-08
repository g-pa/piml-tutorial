# Physics Informed Machine Learning tutorial
This repository contains the code for a tutorial on **Physics Informed Machine Learning** (PIML) and **Neural Networks**.\
This is strongly inspired by famous [Ben Moseley post](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/) and demonstrates how to incorporate physical laws into machine learning models to improve predictions and interpretability.

---

## Repository Structure
```bash
.
├── .env.template
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── uv.lock
├── notebooks/
│   └── 0 Inertial Oscillator.ipynb
└── src/
    ├── models.py
    ├── utils.py
    └── physics/
        └── inertial_oscillator.py
```

- ```notebooks/``` – Jupyter notebooks demonstrating tutorial examples.
- ```src/models.py``` – Neural network models.
- ```src/utils.py``` – Utility functions.
- ```src/physics/``` – Physics-informed functions for modeling systems (e.g., inertial oscillator).
- ```.env.template``` – Template for environment variables.
- ```pyproject.toml``` – Project dependencies and configurations.
- ```uv.lock``` – Locked dependencies file for reproducibility.

---

## Environment Setup
### 1. Clone the repository
```bash
git clone https://github.com/g-pa/piml-tutorial.git
cd piml-tutorial
```

### 2. Setup Python environment
This project uses Python with ```uv``` as the dependency manager. If you don’t have ```uv``` installed, you can install it following the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

Then you can install all the dependencies running the following command within the cloned repository:
```bash
uv sync
```
This command will:
- Read the project's dependency specifications
- Create a virtual environment (if one doesn't exist)
- Install all required packages with their correct versions

### 3. Configure environment variables
1. Copy the template ```.env.template``` to ```.env```
2. Modify ```.env``` accordingly to include all the custom paths needed

---

## Running the tutorial
### 1. Activate the Python environment:
#### macOS and Linux
```bash
source .venv/bin/activate
```
#### Windows
```bash
.venv\Scripts\activate
```

### 2. Run the Jupyter notebooks
Run the notebooks in ```notebooks``` directory to explore Physics Informed Machine Learning examples.

---

## References
- [Ben Moseley – So what is a physics-informed neural network?](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)
- [Damped Harmonic Oscillator](https://beltoforion.de/en/harmonic_oscillator/)