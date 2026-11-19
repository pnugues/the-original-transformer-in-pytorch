"""
evaluate_translation.py
Évaluation d'un Transformer entraîné (EN -> DE, WMT14) : traduit le jeu de test
par beam search (taille 4, pénalité de longueur 0,6, longueur max = source + 50,
comme dans Vaswani et al. 2017) et écrit un fichier d'hypothèses, une par ligne.
Aucun entraînement ici. Le modèle et le décodage sont ceux du notebook.

Exemples
--------
# moyenne des 5 derniers checkpoints (comme dans le papier) :
python evaluate_translation.py --ckpt-dir checkpoints --average 5

# un seul checkpoint (par exemple le dernier, pour comparer) :
python evaluate_translation.py --checkpoint checkpoints/checkpoint_step_100000.pt

# essai rapide sur 300 phrases :
python evaluate_translation.py --ckpt-dir checkpoints --average 5 --limit 300

Données attendues : <data-dir>/spm.model et <data-dir>/<split>.json
(un objet JSON par ligne, avec "translation": {"en": ..., "de": ...}).
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch                      # noqa: E402
import torch.nn as nn             # noqa: E402
import sentencepiece as spm       # noqa: E402

try:
    from tqdm import tqdm
except ImportError:               # tqdm facultatif
    tqdm = None


# ----------------------------------------------------------------------------
# Modèle : identique à celui du notebook (mêmes noms de paramètres, donc les
# checkpoints se chargent tels quels)
# ----------------------------------------------------------------------------
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 4 * EMB_SIZE
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        self.scale = emb_size ** 0.5
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pe = torch.zeros((maxlen, emb_size))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pe.unsqueeze(0))
        self.emb_size = emb_size

    def forward(self, token_embedding):
        return self.dropout(token_embedding * self.scale
                            + self.pos_embedding[:, :token_embedding.size(1), :])


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=EMB_SIZE, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=FFN_HID_DIM, dropout=DROPOUT, batch_first=True)
        self.emb_size = EMB_SIZE
        self.embedding = nn.Embedding(vocab_size, EMB_SIZE, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(EMB_SIZE, dropout=DROPOUT)
        self.generator = nn.Linear(EMB_SIZE, vocab_size, bias=False)
        self.generator.weight = self.embedding.weight      # poids partagés

    def encode(self, src, src_mask=None):
        return self.transformer.encoder(
            self.positional_encoding(self.embedding(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.embedding(tgt)), memory, tgt_mask)


# ----------------------------------------------------------------------------
# Chargement des poids, moyenne de checkpoints
# ----------------------------------------------------------------------------
def load_state_dict_file(path):
    """Lit un checkpoint (sur CPU) et retourne (state_dict, step)."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as e:                       # ancien format : pickle complet
        print(f"  (weights_only=True a échoué pour {path}: {type(e).__name__}; "
              f"relecture sans cette protection, fichier supposé de confiance)")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    # un modèle compilé (torch.compile) ajoute le préfixe "_orig_mod."
    sd = {re.sub(r'^(_orig_mod\.|module\.)+', '', k): v for k, v in sd.items()}
    return sd, ckpt.get('step')


def step_of(path):
    m = re.search(r'checkpoint_step_(\d+)\.pt$', path)
    return int(m.group(1)) if m else -1


def average_state_dicts(paths):
    avg, n = None, 0
    for p in paths:
        sd, step = load_state_dict_file(p)
        print(f"  + {p} (step {step})")
        if avg is None:
            avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            if sd.keys() != avg.keys():
                sys.exit(f"Les paramètres de {p} ne correspondent pas aux précédents.")
            for k in avg:
                avg[k] += sd[k].float()
        n += 1
    return {k: (v / n).float() for k, v in avg.items()}, n


# ----------------------------------------------------------------------------
# Décodage : beam search, copié du notebook
# ----------------------------------------------------------------------------
@torch.no_grad()
def beam_search_decode(model, src, bos_idx, eos_idx, beam_size=4, alpha=0.6,
                       extra_len=50):
    device = src.device
    max_len = src.size(1) + extra_len
    memory = model.encode(src, None)

    ys = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
    beam = [(ys, 0.0, 1, False)]            # (séquence, log-prob cumulée, longueur, terminée)

    for _ in range(max_len - 1):
        candidates = []
        for ys_seq, score, length, done in beam:
            if done:
                candidates.append((ys_seq, score, length, True))
                continue
            tgt_len = ys_seq.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(model.transformer.nhead, -1, -1)
            out = model.decode(ys_seq, memory, tgt_mask)
            logprobs = torch.log_softmax(model.generator(out[:, -1, :]), dim=-1)
            top_lp, top_ix = torch.topk(logprobs, beam_size, dim=1)
            for i in range(beam_size):
                tok = top_ix[0, i].item()
                new_seq = torch.cat([ys_seq, torch.tensor([[tok]], device=device)], dim=1)
                candidates.append((new_seq, score + top_lp[0, i].item(),
                                   length + 1, tok == eos_idx))
        # pénalité de longueur de Wu et al. / Vaswani : ((5 + longueur) / 6) ** alpha
        beam = sorted(candidates,
                      key=lambda x: x[1] * ((5 + 1) / (5 + x[2])) ** alpha,
                      reverse=True)[:beam_size]
        if all(done for _, _, _, done in beam):
            break

    best = beam[0][0].squeeze(0)
    eos_pos = (best == eos_idx).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        best = best[:eos_pos[0] + 1]
    return best


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='wmt14')
    ap.add_argument('--split', default='test', choices=['test', 'validation', 'train'])
    ap.add_argument('--checkpoint', help="un seul checkpoint .pt")
    ap.add_argument('--ckpt-dir', default='checkpoints',
                    help="dossier des checkpoint_step_*.pt (pour la moyenne)")
    ap.add_argument('--average', type=int, default=5,
                    help="moyenne des N derniers checkpoints (ignoré avec --checkpoint)")
    ap.add_argument('--beam', type=int, default=4)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--extra-len', type=int, default=50, help="longueur max = source + extra-len")
    ap.add_argument('--limit', type=int, default=None, help="ne traduire que les N premières phrases")
    ap.add_argument('--out', default=None, help="fichier d'hypothèses")
    ap.add_argument('--device', default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"PyTorch {torch.__version__} | device : {device}")

    # --- vocabulaire --------------------------------------------------------
    sp = spm.SentencePieceProcessor()
    sp.load(str(data_dir / 'spm.model'))
    vocab_size = len(sp)
    bos, eos, pad = sp.bos_id(), sp.eos_id(), sp.pad_id()
    print(f"Vocabulaire : {vocab_size} | BOS={bos} EOS={eos} PAD={pad}")

    # --- poids ----------------------------------------------------------------
    if args.checkpoint:
        print(f"Checkpoint unique : {args.checkpoint}")
        state_dict, step = load_state_dict_file(args.checkpoint)
        tag = f"step{step}"
    else:
        files = sorted(glob.glob(os.path.join(args.ckpt_dir, 'checkpoint_step_*.pt')),
                       key=step_of, reverse=True)[:args.average]
        if len(files) < 1:
            sys.exit(f"Aucun checkpoint_step_*.pt dans {args.ckpt_dir}")
        print(f"Moyenne des {len(files)} derniers checkpoints de {args.ckpt_dir} :")
        state_dict, n = average_state_dicts(files)
        tag = f"moyenne{n}"

    model = Seq2SeqTransformer(vocab_size, pad)
    model.load_state_dict(state_dict, strict=True)   # erreur si un nom de paramètre diffère
    model.to(device)
    model.transformer.encoder.enable_nested_tensor = False
    model.eval()
    assert model.generator.weight.data_ptr() == model.embedding.weight.data_ptr(), \
        "les poids du générateur et des embeddings ne sont plus partagés"
    fingerprint = model.embedding.weight.detach().cpu().double().abs().sum().item()  # MPS n'a pas de float64
    print(f"Poids chargés (strict). Empreinte des embeddings : {fingerprint:.4f}")
    print("  -> deux chargements différents (dernier checkpoint / moyenne) doivent "
          "donner deux empreintes différentes.")

    # --- données --------------------------------------------------------------
    path = data_dir / f'{args.split}.json'
    srcs, refs = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            tr = json.loads(line)['translation']
            srcs.append(tr['en'].strip())
            refs.append(tr['de'].strip())
    if args.limit:
        srcs, refs = srcs[:args.limit], refs[:args.limit]
    print(f"{args.split} : {len(srcs)} phrases")

    out_path = args.out or f"hyps_{args.split}_{tag}.txt"

    # --- traduction ---------------------------------------------------------
    t0 = time.time()
    hyps, n_long = [], 0
    it = enumerate(srcs)
    if tqdm:
        it = tqdm(it, total=len(srcs))
    with torch.no_grad():
        for idx, s in it:
            ids = sp.encode(s, out_type=int, add_bos=True, add_eos=True)  # pas de troncature
            if len(ids) > 256:
                n_long += 1
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            best = beam_search_decode(model, src, bos, eos, args.beam, args.alpha,
                                      args.extra_len)
            hyp = sp.decode(best.tolist()).replace('<s>', '').replace('</s>', '')
            hyp = hyp.replace('\r', ' ').replace('\n', ' ').strip()   # une hypothèse = une ligne
            hyps.append(hyp)
            if not tqdm and (idx + 1) % 200 == 0:
                print(f"  {idx + 1}/{len(srcs)}  ({time.time() - t0:.0f} s)", flush=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for h in hyps:
            f.write(h + '\n')
    print(f"{len(hyps)} hypothèses écrites dans {out_path} ({time.time() - t0:.0f} s)")
    if n_long:
        print(f"Attention : {n_long} phrases sources de plus de 256 tokens (non tronquées).")
    n_empty = sum(1 for h in hyps if not h)
    if n_empty:
        print(f"Attention : {n_empty} hypothèses vides.")

    # --- référence et BLEU ----------------------------------------------------
    ref_path = f"ref_{args.split}.txt"
    with open(ref_path, 'w', encoding='utf-8') as f:
        for r in refs:
            f.write(r.replace('\r', ' ').replace('\n', ' ') + '\n')
    print(f"Références dans {ref_path}")
    try:
        import sacrebleu
        for tok in ('intl', '13a'):
            sc = sacrebleu.BLEU(tokenize=tok).corpus_score(hyps, [refs])
            print(f"BLEU (sacrebleu, tokenize={tok}) : {sc.score:.2f}   {sc}")
    except ImportError:
        print("sacrebleu n'est pas installé : pip install sacrebleu")


if __name__ == '__main__':
    main()
