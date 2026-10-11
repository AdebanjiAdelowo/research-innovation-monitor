# Historical run records

Run records kept as evidence after they were superseded. They are not the source of
any number in the top-level README; the current records are in `results/<experiment>/`.

| Run | Why it was superseded |
|---|---|
| `rag_scifact_local-20261007T011007Z-bf542c8-4161e98d` | Its commit was renamed by the history rewrite ([docs/history-rewrite.md](../../docs/history-rewrite.md)) |
| `retrieval_dapfam-20261007T011024Z-bf542c8-3121204c` | Same |
| `routing-20261007T000821Z-331b9c3-a01e5003` | Recorded `dirty: true` without the list of uncommitted paths |

Each was rerun from a clean tree and reproduced; the comparison is in
[docs/history-rewrite.md](../../docs/history-rewrite.md#reruns). The files here are
byte-for-byte copies; nothing in them was edited. `answers.jsonl` is
listed with its checksum in the RAG record but is not redistributed, as in the
current runs.
