# Shared SNN checkpoints

Each subfolder is one training run (pigeons **A** and **B** together):

```
n10000_main/
  weights.pt
  classifier_a.joblib
  classifier_b.joblib
  meta.json
latest.txt          # name of the default run for Playground.ipynb
```

After `Models.ipynb` Step 10 (`save_run`), commit this folder so teammates can
`load_run("n10000_main")` without retraining.

**Size note:** a 10 000-neuron run is hundreds of MB. This repo tracks those
files with **Git LFS** (see `.gitattributes`). Run once:

```bash
git lfs install
```
