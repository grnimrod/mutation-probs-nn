from gwf import Workflow


gwf = Workflow()

INPUT_FILE = "./../data/DNM_15mer_v1.tsv"
SPLIT_PREFIX = "./../data/splits/chunk_"
NUM_TASKS = 10

gwf.target("split_tsv",
           inputs=[INPUT_FILE],
           outputs=[f"{SPLIT_PREFIX}{i}" for i in range(NUM_TASKS)], cores=4, memory="16g") << f"""
           split -d -l $(( $(wc -l < {INPUT_FILE}) / {NUM_TASKS} )) {INPUT_FILE} {SPLIT_PREFIX}
           """

for i in range(NUM_TASKS):
    gwf.target(f"process_chunk_{i}",
               inputs=[f"{SPLIT_PREFIX}{i}"],
               outputs=[f"{SPLIT_PREFIX}{i}_encoded"], cores=4, memory="16g") << f"""
               ~/MutationAnalysis/Nimrod/.venv/bin/python preprocessing.py {SPLIT_PREFIX}{i}
               """

gwf.target("merge_results",
           inputs=[f"{SPLIT_PREFIX}{i}_encoded" for i in range(NUM_TASKS)],
           outputs=["./../data/DNM_15mer_v1_encoded"], cores=4, memory="16g") << f"""
           cat {" ".join(f"{SPLIT_PREFIX}{i}_encoded" for i in range(NUM_TASKS))} > ./../data/DNM_15mer_v1_encoded
           """