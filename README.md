# Impact of External Disruptions on Passenger Traffic at Schiphol Airport

**TIL6022 – Python Programming | Group 11 | TU Delft (2026)**

## 1. Project Overview

This project investigates how different external disruptions affect passenger traffic at Amsterdam Airport Schiphol and what these changes reveal about its resilience.

Three major disruptions are compared:
- September 11 attacks (2001)
- Global financial crisis (2008–2009)
- COVID-19 pandemic (2020)

Using historical monthly passenger traffic data, we analyse the magnitude of each disruption, changes relative to baseline conditions, and subsequent recovery patterns. Additional datasets are used to contextualise and validate the findings.

## 2. Authors

- Britt de Heer
- Lotte van den Bos
- Mathijs Kristen
- Koen Meijer
- Meike van Sliedregt

Individual contributions are described in Section 10.2 of the main report.

## 3. Repository Structure

The main project report is located in `Reports/Main_report_group11.ipynb`.

| File / Folder | Description |
|---|---|
| `Reports/` | Main report and intermediate results |
| `results 911.ipynb` | Analysis of the 9/11 attacks |
| `results_covid.ipynb` | Analysis of the COVID-19 pandemic |
| `Financial_crisis_dataresults.ipynb` | Analysis of the financial crisis |
| `results_comparison.ipynb` | Comparison of the three disruptions |
| `covid_data/` | COVID-19 datasets |
| `financial_data/` | Financial crisis and AEX datasets |
| `US_transport_data/` | US transportation data for 9/11 validation |
| `Figures 911/` | Figures related to 9/11 |
| `Figures Covid/` | Figures related to COVID-19 |
| `Figures financial crisis/` | Figures related to the financial crisis |
| `Comparison figures/` | Comparative visualisations |
| `Code/` | Supporting analysis code |
| `app.py` | Project application |
| `requirements.txt` | Required Python packages |

## 4. Requirements and Installation

The project is developed in Python using Jupyter notebooks.

To install the required dependencies:

1. Clone or download the repository.
2. Open the project folder in VS Code or Jupyter.
3. Open a terminal in the repository's root directory.
4. Install the required packages:


python -m pip install -r requirements.txt


5. Select the appropriate Python environment as the Jupyter kernel.

## 5. How to Run the Analysis

To reproduce the analyses:

1. **Keep the original folder structure unchanged**, as the notebooks use relative file paths to access datasets and figures.
2. Open the relevant Jupyter notebook in VS Code or Jupyter.
3. Run the cells sequentially from top to bottom.
4. For individual event analyses, use:
   - `results 911.ipynb` for the 9/11 analysis.
   - `Financial_crisis_dataresults.ipynb` for the financial crisis analysis.
   - `results_covid.ipynb` for the COVID-19 analysis.
5. Run `results_comparison.ipynb` for the comparative analysis.
6. Open `Reports/Main_report_group11.ipynb` for the final report, including methodology, main visualisations, results, and conclusions.

Additional intermediate code is available in `Reports/Intermediate_results_code.ipynb`.


## 6. Data Sources

The primary dataset consists of monthly passenger traffic statistics from Statistics Netherlands (CBS).

Additional datasets are used to validate and contextualise the findings, including:
- US transportation statistics for the 9/11 analysis.
- AEX stock market data for the financial crisis analysis.
- Additional aviation and COVID-19 statistics.

Further details and references are provided in the main report.

## 7. Final Report

The final report is available in:

`Reports/Main_report_group11.ipynb`

It includes the research background, methodology, data analysis, visualisations, results, discussion, and conclusions.