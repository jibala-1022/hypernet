import os

from python_research.experiments.hsi_attention.arguments import Arguments
from python_research.experiments.hsi_attention.train_attention import main

DATA_DIR = "data"
RESULTS_DIR = "results"

arguments = Arguments(
    dataset_path=os.path.join(DATA_DIR, "PaviaU.npy"),
    labels_path=os.path.join(DATA_DIR, "PaviaU_gt.npy"),
    selected_bands=None,
    validation=0.1,
    test=0.1,
    epochs=9999,
    modules=3,
    patience=5,
    output_dir=RESULTS_DIR,
    batch_size=64,
    attn="y",
    run_idx=str(1),
    cont=0.1
)

main(args=arguments)
