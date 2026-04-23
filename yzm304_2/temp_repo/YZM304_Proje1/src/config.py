from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

CSV_PATH = DATA_DIR / "BankNote_Authentication.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20