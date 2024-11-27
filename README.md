## DRS: Deep Question Reformulation With Structured Output

This is the repository for our paper [DRS: Deep Question Reformulation With Structured Output]()

In this paper, our proposed DRS method leverages large language models and the DFS-based algorithm to iteratively search for possible entity combinations and constrain the output with certain entities, effectively improving the capabilities of large language models in question reformulation.

<img src="./Figs/algorithm.png" alt="algorithm" style="zoom:80%;" />

The results are shown in the Table below:
<img src="./Figs/score.png" alt="score" style="zoom:100%;" />

### Code Instruction

- Upload ``config.yaml`` file.

```bas
openai:
  api_key: "YOUR_API_KEY"
  organization: "YOUR_ORGANIZATION"
```

- Export ``huggingface api key`` for open-source models.

- Change the ``subset_name`` or ``run_model`` in ``parameters.yaml``.

- Run baselines with OpenAI models.

```bas
python baseline_openai.py
```

- Run baselines with HuggingFace models.

```bas
python baseline_huggingface.py
```

- Run **DRS** with OpenAI models.

```bash
python drs_openai.py
```

- Run **DRS** with HuggingFace models.

```bas
python drs_huggingface.py
```

