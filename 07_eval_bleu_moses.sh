#!/bin/bash

# ============================================================
# Script d'évaluation BLEU avec Moses + sacrebleu
# Usage : ./07_eval_bleu_moses.sh [-s indices.txt] <fichier_predictions> <fichier_references>
#   exemple : ./07_eval_bleu_moses.sh hyps_test_moyenne5.txt wmt14/test.de
#   (les PRÉDICTIONS d'abord, la RÉFÉRENCE ensuite)
#
# Option -s / --subset FICHIER : n'évaluer que certaines phrases du jeu de test.
#   FICHIER contient un numéro de ligne par ligne (à partir de 1), relatif au
#   jeu COMPLET (celui des deux fichiers donnés en argument). Il se fabrique
#   avec make_subset_indices.py (par exemple pour le jeu réduit de 2737 phrases).
#   exemple : ./07_eval_bleu_moses.sh -s subset_2737.txt hyps_test_moyenne5.txt wmt14/test.de
# ============================================================

usage() {
    echo "Usage: $0 [-s indices.txt] <fichier_predictions> <fichier_references>"
    exit 1
}

PRED=""
REF=""
SUBSET=""
NARGS=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        -s|--subset)
            [ "$#" -ge 2 ] || usage
            SUBSET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            NARGS=$((NARGS + 1))
            if [ "$NARGS" -eq 1 ]; then PRED="$1"
            elif [ "$NARGS" -eq 2 ]; then REF="$1"
            else usage
            fi
            shift
            ;;
    esac
done
[ "$NARGS" -eq 2 ] || usage

TARGET_LANG="de"   # <-- renommé pour ne pas écraser LANG (locales)

if [ ! -f "$PRED" ]; then
    echo "Erreur : '$PRED' n'existe pas."
    exit 1
fi

if [ ! -f "$REF" ]; then
    echo "Erreur : '$REF' n'existe pas."
    exit 1
fi

# Rappel des rôles, pour repérer tout de suite une inversion des arguments
echo "Prédictions (système) : $PRED  ($(wc -l < "$PRED") lignes)"
echo "Références            : $REF  ($(wc -l < "$REF") lignes)"
if [ "$(wc -l < "$PRED")" -ne "$(wc -l < "$REF")" ]; then
    echo "Erreur : nombre de lignes différent entre prédictions et références."
    exit 1
fi

# --- Fichiers temporaires ---
REF_TOK=$(mktemp)
HYP_TOK=$(mktemp)
REF_ATAT=$(mktemp)
HYP_ATAT=$(mktemp)
PRED_SEL=$(mktemp)
REF_SEL=$(mktemp)
IDX_CLEAN=$(mktemp)
trap 'rm -f "$REF_TOK" "$HYP_TOK" "$REF_ATAT" "$HYP_ATAT" "$PRED_SEL" "$REF_SEL" "$IDX_CLEAN"' EXIT

# --- Sous-ensemble de phrases (option -s) ---
if [ -n "$SUBSET" ]; then
    if [ ! -f "$SUBSET" ]; then
        echo "Erreur : '$SUBSET' n'existe pas."
        exit 1
    fi
    tr -d '\r' < "$SUBSET" | grep -E '^[0-9]+$' | sort -n -u > "$IDX_CLEAN"
    NSEL=$(wc -l < "$IDX_CLEAN" | tr -d ' ')
    NTOT=$(wc -l < "$REF" | tr -d ' ')
    MAXIDX=$(tail -1 "$IDX_CLEAN")
    if [ "$NSEL" -eq 0 ] || [ "$MAXIDX" -gt "$NTOT" ] || [ "$(sort -n "$IDX_CLEAN" | head -1)" -lt 1 ]; then
        echo "Erreur : '$SUBSET' doit contenir des numéros de ligne entre 1 et $NTOT."
        exit 1
    fi
    awk 'NR==FNR { k[$1+0]=1; next } (FNR in k)' "$IDX_CLEAN" "$PRED" > "$PRED_SEL"
    awk 'NR==FNR { k[$1+0]=1; next } (FNR in k)' "$IDX_CLEAN" "$REF"  > "$REF_SEL"
    echo "Sous-ensemble : $NSEL phrases évaluées sur $NTOT (indices de $SUBSET)"
    PRED="$PRED_SEL"
    REF="$REF_SEL"
fi

# --- Moses ---
MOSES_DIR="${MOSES_DIR:-$HOME/mosesdecoder}"
if [ ! -d "$MOSES_DIR" ]; then
    echo "Téléchargement de Moses..."
    git clone https://github.com/moses-smt/mosesdecoder.git "$MOSES_DIR"
fi

export NONBREAKING_PREFIXES="$MOSES_DIR/scripts/share/nonbreaking_prefixes"

# --- Tokenisation ---
echo "Tokenisation des références..."
perl "$MOSES_DIR/scripts/tokenizer/replace-unicode-punctuation.perl" -l "$TARGET_LANG" < "$REF" \
    | perl "$MOSES_DIR/scripts/tokenizer/tokenizer.perl" -l "$TARGET_LANG" > "$REF_TOK"

echo "Tokenisation des prédictions..."
perl "$MOSES_DIR/scripts/tokenizer/replace-unicode-punctuation.perl" -l "$TARGET_LANG" < "$PRED" \
    | perl "$MOSES_DIR/scripts/tokenizer/tokenizer.perl" -l "$TARGET_LANG" > "$HYP_TOK"

# --- Fractionnement ---
echo "Fractionnement des tirets..."
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < "$REF_TOK" > "$REF_ATAT"
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < "$HYP_TOK" > "$HYP_ATAT"

# --- BLEU ---
echo "Calcul du score BLEU..."
sacrebleu --tokenize none --force "$REF_ATAT" < "$HYP_ATAT"
