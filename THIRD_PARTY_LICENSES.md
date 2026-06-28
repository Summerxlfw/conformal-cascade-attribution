# Third-Party Licenses And Sources

This repository does not redistribute raw datasets or model weights. Users must obtain them from the official sources and comply with the original licenses.

| Asset | Source | License / Terms | Use in this study |
|---|---|---|---|
| SLURP corpus | https://github.com/pswietojanski/slurp | CC BY 4.0 according to the project/license metadata; verify against the official repository before reuse. | Speech-to-intent corpus. |
| Whisper models/code | https://github.com/openai/whisper | MIT | ASR stage for small.en and medium.en experiments. |
| faster-whisper | https://github.com/SYSTRAN/faster-whisper | MIT | Word-level log-probability / confidence signals. |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | Sentence embedding front end for the intent classifier. |
| NumPy, pandas, SciPy, scikit-learn, Matplotlib, pyarrow | PyPI / upstream projects | Respective open-source licenses | Analysis and plotting. |

Please cite the corresponding papers and software repositories when reusing these assets.
