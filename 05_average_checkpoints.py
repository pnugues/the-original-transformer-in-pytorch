# Moyennage de modèles : on charge plusieurs modèles entraînés, on fait la moyenne de leurs poids, et on sauvegarde le modèle moyen.
import glob
import os
import torch
from collections import OrderedDict


def average_checkpoints(checkpoint_paths, device='cpu'):
    """
    Charge plusieurs checkpoints et retourne un state_dict dont les poids sont la moyenne.
    - checkpoint_paths : liste des chemins vers les .pt
    - device : 'cpu' ou 'cuda' ou 'mps'
    Retourne : state_dict moyenné (OrderedDict)
    """
    avg_state_dict = None
    count = 0

    for path in checkpoint_paths:
        print(f"Chargement de {path}")
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        if avg_state_dict is None:
            avg_state_dict = OrderedDict()
            for key, param in state_dict.items():
                # on copie en float
                avg_state_dict[key] = param.clone().float()
        else:
            for key in avg_state_dict.keys():
                avg_state_dict[key] += state_dict[key].float()

        count += 1

    # Division par le nombre de checkpoints
    for key in avg_state_dict.keys():
        # on remet en float32
        avg_state_dict[key] = (avg_state_dict[key] /
                               count).to(dtype=torch.float32)

    print(f"Moyenne effectuée sur {count} modèles.")
    return avg_state_dict


def step_from_filename(f):
    # ex: checkpoints/checkpoint_step_1000.pt
    return int(f.split('_')[-1].split('.')[0])


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_dir = "checkpoints/"
checkpoint_files = glob.glob(os.path.join(
    checkpoint_dir, "checkpoint_step_*.pt"))
# du plus récent au plus ancien
checkpoint_files.sort(key=step_from_filename, reverse=True)
last_five = checkpoint_files[:5]  # les 5 derniers

if len(last_five) < 2:
    print("Pas assez de checkpoints pour faire une moyenne (il en faut au moins 2).")
else:
    avg_state_dict = average_checkpoints(last_five, device=DEVICE)

torch.save({
    'step': 'average',
    'model_state_dict': avg_state_dict,
    'loss': None  # ou mettez une valeur
}, "checkpoints/averaged_model.pt")
