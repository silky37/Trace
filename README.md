# Trace
This is an official repository of Trace.

## Run evaluation
After installing packages in requirement.txt, run the following codes.
```
# Evaluate Open-source models
bash ./src/run_open.sh

# Evaluate Closed-source models
bash ./src/run_closed.sh
```

## Citation
If you use this dataset or code, please cite our paper:

```bibtex
@inproceedings{jeon2025trace,
  title     = {"Going to a trap house" conveys more fear than "Going to a mall": Benchmarking Emotion Context Sensitivity for LLMs},
  author    = {Jeon, Eojin and Lee, Mingyu and Kim, Sangyun and Kim, Junho and Cho, Wanzee and Kam, Tae-Eui and Lee, Sang Keun},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025},
  pages     = {14848--14869},
  url       = {[https://aclanthology.org/2025.findings-emnlp.802/](https://aclanthology.org/2025.findings-emnlp.802/)}
}
