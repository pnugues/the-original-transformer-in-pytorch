from datasets import load_dataset
from pathlib import Path

data_dir = Path('wmt14')
data_dir.mkdir(exist_ok=True)

# Téléchargement du jeu de données WMT14 (allemand-anglais)
dataset = load_dataset("wmt/wmt14", "de-en")

# Extraction et sauvegarde en fichiers texte
dataset['train'].to_json(data_dir / "train.json",
                         orient="records")
dataset['validation'].to_json(
    data_dir / "validation.json", orient="records")
dataset['test'].to_json(data_dir / "test.json",
                        orient="records")
