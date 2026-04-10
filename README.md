# Synchro Jump

Prototype Python pour construire un modele plan d'un saut vertical avec une plateforme mobile,
formuler un OCP `bioptim`, et explorer les parametres depuis une GUI avec sliders et figures.

## Developpement

```bash
python -m pip install -e .[test,opt,gui]
./scripts/run_checks.sh
```

## Environnement Conda

Creer un environnement Conda reproductible depuis la racine du projet :

```bash
conda env create -f environment.yml
conda activate synchro-jump
```

Si l'environnement existe deja et que tu veux juste le mettre a jour :

```bash
conda env update -f environment.yml --prune
conda activate synchro-jump
```

## Lancer la GUI

```bash
PYTHONPATH=src python -m synchro_jump
```
