# Results folder

Generated files are intentionally not committed in this cleaned version because old outputs were stale and contradictory.

Regenerate them with:

```bash
python example_usage.py --sample-size 5 --num-docs 100
```

The regenerated outputs include answer metrics plus operational traces:

- EM / F1 / retrieval hit rate
- retrieval rate
- LLM calls, vector queries, and sparse queries
- weak retrieval rate
- suspicious-context rate
- answer-support rate and grounding score
- duplicate-context rate
- query complexity and retrieval strategy per question
- `run_config.json` with the exact model/retrieval/chunking settings used for that run
