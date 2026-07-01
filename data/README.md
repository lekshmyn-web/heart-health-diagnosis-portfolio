# Data Setup

Raw dataset files are not included in this public repository.

To run the project locally, create this folder structure:

```text
data/
├── heart_disease_cp2.csv
└── cp2_shuffled.csv
```

## Required Files

* `heart_disease_cp2.csv`
  The source cardiovascular dataset used for preparation and analysis.

* `cp2_shuffled.csv`
  The shuffled dataset used by the modelling scripts.

## Setup Steps

1. Obtain the original Framingham Heart Study–based dataset from the source used for this project.

2. Save it as `heart_disease_cp2.csv` inside the `data` folder.

3. Run:

   ```bash
   python python/prepare_dataset.py
   ```

4. This creates or refreshes `cp2_shuffled.csv`.

5. Run any model script from the `python` folder.

## Note

The dataset is excluded from this repository to keep the public portfolio version lightweight and to respect external data-source terms.
