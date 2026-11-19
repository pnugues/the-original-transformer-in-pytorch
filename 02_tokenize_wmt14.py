"""
prepare_wmt14_sentencepiece.py
Préparation WMT14 EN-DE avec SentencePiece de Kudo (meilleur que subword-nmt)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("wmt14")          # dossier des données
SRC_LANG = "en"
TGT_LANG = "de"
VOCAB_SIZE = 37000
SPLITS = ("train", "validation", "test")


def extract_plain_files(splits=SPLITS):
    """
    Extraire les fichiers .en et .de à partir des JSON pour chaque split.
    Crée notamment train.en, train.de, validation.en, validation.de,
    test.en, test.de.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        json_path = DATA_DIR / f"{split}.json"
        if not json_path.exists():
            print(f"⚠ {json_path} introuvable, skip")
            continue

        src_path = DATA_DIR / f"{split}.{SRC_LANG}"
        tgt_path = DATA_DIR / f"{split}.{TGT_LANG}"

        print(f"Extracting {split}.{SRC_LANG} / {split}.{TGT_LANG}...")
        records = pd.read_json(json_path, lines=True).to_dict(orient="records")

        with open(src_path, "w", encoding="utf-8") as f_src, \
                open(tgt_path, "w", encoding="utf-8") as f_tgt:
            for row in tqdm(records, desc=split):
                src = row["translation"][SRC_LANG].strip()
                tgt = row["translation"][TGT_LANG].strip()
                f_src.write(src + "\n")
                f_tgt.write(tgt + "\n")

        print(f"✓ {src_path}, {tgt_path}")


def combine_train_files():
    """
    Combiner train.en et train.de pour l'entraînement SentencePiece
    (vocabulaire partagé EN-DE).
    """
    combined_file = DATA_DIR / "train.combined"
    train_en = DATA_DIR / f"train.{SRC_LANG}"
    train_de = DATA_DIR / f"train.{TGT_LANG}"

    # S'assurer que les fichiers source existent
    if not train_en.exists() or not train_de.exists():
        extract_plain_files(splits=("train",))

    if not combined_file.exists():
        print("Creating combined training file...")
        with open(combined_file, "w", encoding="utf-8") as fout, \
                open(train_en, encoding="utf-8") as fen, \
                open(train_de, encoding="utf-8") as fde:
            for line in fen:
                fout.write(line)
            for line in fde:
                fout.write(line)
        print(f"✓ {combined_file}")
    else:
        print("✓ Combined training file already exists")


def train_sentencepiece():
    """
    Entraîner SentencePiece sur les données WMT14.
    Vocabulaire partagé EN-DE comme dans le papier original.
    """
    print("\nTraining SentencePiece model...")

    combined_file = DATA_DIR / "train.combined"
    if not combined_file.exists():
        combine_train_files()

    model_prefix = str(DATA_DIR / "spm")

    spm.SentencePieceTrainer.train(
        input=str(combined_file),
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,  # 0.9995, # allemand + caractères spéciaux
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=[],
        num_threads=16,
        train_extremely_large_corpus=False,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1000000,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
    )
    print(
        f"✓ SentencePiece model trained: {model_prefix}.model, {model_prefix}.vocab")


def apply_sentencepiece():
    """Appliquer SentencePiece à tous les splits."""
    print("\nApplying SentencePiece encoding...")

    model_path = DATA_DIR / "spm.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    for split in SPLITS:
        json_path = DATA_DIR / f"{split}.json"
        if not json_path.exists():
            print(f"⚠ {json_path} introuvable, skip")
            continue

        src_out = DATA_DIR / f"{split}.spm.{SRC_LANG}"
        tgt_out = DATA_DIR / f"{split}.spm.{TGT_LANG}"

        print(f"Processing {split}...")
        records = pd.read_json(json_path, lines=True).to_dict(orient="records")

        with open(src_out, "w", encoding="utf-8") as f_src, \
                open(tgt_out, "w", encoding="utf-8") as f_tgt:
            for row in tqdm(records, desc=split):
                src_line = row["translation"][SRC_LANG].strip()
                tgt_line = row["translation"][TGT_LANG].strip()
                src_tok = sp.encode(src_line, out_type=str)
                tgt_tok = sp.encode(tgt_line, out_type=str)
                f_src.write(" ".join(src_tok) + "\n")
                f_tgt.write(" ".join(tgt_tok) + "\n")

    print("✓ SentencePiece encoding complete")


def print_stats():
    sp = spm.SentencePieceProcessor()
    sp.load(str(DATA_DIR / "spm.model"))

    print("\n" + "=" * 60)
    print(f"Vocabulaire : {sp.vocab_size()}")
    for i in range(4):
        print(f"  id {i}: {sp.id_to_piece(i)}")

    train_spm_en = DATA_DIR / f"train.spm.{SRC_LANG}"
    lengths = []
    with open(train_spm_en, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            lengths.append(len(line.split()))

    if not lengths:
        sys.exit(f"❌ {train_spm_en} est vide")

    a = np.array(lengths)
    print(
        f"Longueur (10K premières phrases train) : "
        f"moyenne {a.mean():.1f}, médiane {np.median(a):.0f}, "
        f"p95 {np.percentile(a, 95):.0f}, max {a.max()}"
    )

    example_en = (
        "The Transformer architecture has revolutionized "
        "natural language processing."
    )
    example_de = (
        "Die Transformer-Architektur hat die Verarbeitung "
        "natürlicher Sprache revolutioniert."
    )

    print("\nExample encoding:")
    print(f"EN: {example_en}")
    print(f"    → {sp.encode(example_en, out_type=str)}")
    print(f"DE: {example_de}")
    print(f"    → {sp.encode(example_de, out_type=str)}")
    print("=" * 60)


if __name__ == "__main__":
    # 1. Extraire les fichiers texte bruts (train/validation/test .en/.de)
    extract_plain_files()

    # 2. Combiner train pour SentencePiece
    combine_train_files()

    # 3. Entraîner SentencePiece
    train_sentencepiece()

    # 4. Encoder tous les splits
    apply_sentencepiece()

    # 5. Stats
    print_stats()

    print("\n✓ Data preparation complete!")
    print("\nGenerated files:")
    print(f"  - {DATA_DIR}/spm.model")
    print(f"  - {DATA_DIR}/spm.vocab")
    print(f"  - {DATA_DIR}/train.{SRC_LANG}, train.{TGT_LANG}")
    print(f"  - {DATA_DIR}/validation.{SRC_LANG}, validation.{TGT_LANG}")
    print(f"  - {DATA_DIR}/test.{SRC_LANG}, test.{TGT_LANG}")
    print(f"  - {DATA_DIR}/*.spm.{SRC_LANG}")
    print(f"  - {DATA_DIR}/*.spm.{TGT_LANG}")
