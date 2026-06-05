"""Reduced workflow check for the supplementary code archive.

The full simulation is computationally intensive. This demonstration script
does not rerun the primary simulation; it verifies that the archive structure is
present and writes small expected-output templates used to check the workflow.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "outputs_expected"
EXPECTED.mkdir(exist_ok=True)


def write_template(name, columns):
    path = EXPECTED / name
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    return path


def main():
    required = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "Supplementary_Code_1_main_simulation.py",
        ROOT / "Supplementary_Code_2_additional_sensitivity.py",
    ]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing archive file(s): " + ", ".join(missing))

    written = [
        write_template(
            "primary_results_template.csv",
            ["Scenario", "Method", "Bias", "RMSE", "Cov", "CI_Width", "VR", "N_valid"],
        ),
        write_template(
            "sensitivity_results_template.csv",
            ["scenario_group", "fid", "Method", "Bias", "RMSE", "Cov", "CI_Width", "VR", "N_valid"],
        ),
        write_template(
            "figure_outputs_template.csv",
            ["figure_file", "scenario", "analysis_group"],
        ),
    ]

    print("Supplementary code archive structure verified.")
    print("Expected-output templates written:")
    for path in written:
        print(f"- {path.relative_to(ROOT)}")
    print("Run the two Supplementary_Code_*.py scripts for the full simulation workflow.")


if __name__ == "__main__":
    main()
