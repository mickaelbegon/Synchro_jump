# Synchro Jump

Prototype Python pour construire un modele plan d'un saut vertical avec une plateforme mobile,
formuler un OCP `bioptim`, et explorer les parametres depuis une GUI avec sliders et figures.

## Developpement

`bioptim` n'est pas installee via `pip` sur ce projet. La voie recommandee est Conda :

```bash
conda env create -f environment.yml
conda activate synchro-jump
```

Si l'environnement existe deja :

```bash
conda env update -f environment.yml --prune
conda activate synchro-jump
```

Le package du projet est ensuite disponible en mode editable via le fichier d'environnement.

Si tu veux installer manuellement dans un environnement deja cree :

```bash
conda install -c conda-forge bioptim
python -m pip install -e .[test,opt,gui]
./scripts/run_checks.sh
```

## Lancer la GUI

```bash
PYTHONPATH=src python -m synchro_jump
```
