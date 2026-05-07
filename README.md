# Synchro Jump

Prototype Python pour construire un modele plan d'un saut vertical avec une plateforme mobile,
formuler un OCP `bioptim`, et explorer les parametres depuis une GUI avec sliders et figures.

## Developpement

`bioptim` n'est pas installee via `pip` sur ce projet. La voie recommandee est Conda :

```bash
conda env create -f environment.yml
conda activate synchro-jump
```

Le projet cible desormais `bioptim 3.4.0` et conserve une compatibilite avec
`3.2.1` pour la construction de l'OCP explicite.

Si l'environnement existe deja :

```bash
conda env update -f environment.yml --prune
conda activate synchro-jump
```

Le package du projet est ensuite disponible en mode editable via le fichier d'environnement.

Si tu veux installer manuellement dans un environnement deja cree :

```bash
conda install -c conda-forge bioptim=3.4.0
python -m pip install -e .[test,opt,gui]
./scripts/run_checks.sh
```

## Lancer la GUI

```bash
PYTHONPATH=src python -m synchro_jump
```

## Prototype Avatar 3D

Le prototype 3D est volontairement isole du reste du projet pour ne pas casser
la GUI actuelle en Tk. Il vit dans :

- [src/synchro_jump/avatar_viewer](/Users/mickaelbegon/Documents/Synchro_jump/src/synchro_jump/avatar_viewer)
- [examples/run_avatar_gui.py](/Users/mickaelbegon/Documents/Synchro_jump/examples/run_avatar_gui.py)

Dependances optionnelles :

```bash
python -m pip install -e .[avatar3d]
```

Lancer l'inspection du rig uniquement :

```bash
PYTHONPATH=src python examples/run_avatar_gui.py --inspect-only
```

Lancer le prototype Qt + Panda3D :

```bash
PYTHONPATH=src python examples/run_avatar_gui.py
```

Limitations actuelles :

- le viewer 3D est optionnel et non connecte a la GUI Tk principale pour l'instant
- le mapping pilote surtout la racine et le dos ; les bras et les jambes restent fixes au pose de repos
- les corrections d'axes par os sont prevues dans l'architecture, mais demandent encore une validation visuelle sur le rig cible
