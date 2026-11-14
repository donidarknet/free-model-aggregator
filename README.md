# Free Model Aggregator

A single long-running server that serves the OpenAI Chat Completions API and
the Anthropic Messages API by routing every request to the best free model
that can serve it, across ten simulated free-tier providers.

Entry for the make.fun competition `free-model-aggregator-01` (agent/v1).

## Run

```sh
python3 main.py            # reads $MAKEFUN_PROMPT_FILE, registers, serves, exits 0 when done
python3 main.py --list     # print the catalog (brief-only fallback when the env is unreachable)
```

Python 3.13, standard library only. No dependencies.

## How it works

- **Catalog probing.** On startup it probes each provider's catalog route on
  all three wires (OpenAI, Anthropic, Gemini) and registers exactly the free
  models it finds, with the provider directory's `base_url` and `key_env`.
- **Rating-first routing.** Candidates are ordered by the sealed quality
  rating (highest first), filtered by what the request needs: tools, vision,
  JSON mode, minimum context. The paid model `anthropic/claude-opus-5.5`
  (rating 1520) is never picked.
- **Cross-wire translation.** Requests and responses (JSON and SSE) are
  translated between the OpenAI, Anthropic and Gemini wires without touching
  client fields: system text, user text, tool definitions, `max_tokens`,
  stops, images (base64), and usage are passed through untouched.
- **One success only.** Exactly one successful (2xx) provider call may
  answer a request; a success is never discarded.
- **Failover.** A 402/429/5xx refusal is retried once on the same candidate
  (it is often one scheduled failure, not a spent budget), then the next
  candidate is tried. 400/401 are never retried. A provider that refuses
  twice in a row is treated as spent.
- **Deterministic replay.** No wall clock, no unseeded randomness: the same
  trial replays byte-for-byte.

## Local practice results

Kit harness (`run_trial.sh`), three practice trials:

| trial | score | quality | served | wire | failover | catalog | valid |
|-------|-------|---------|--------|------|----------|---------|-------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 |

Quality 99 is the trial ceiling: the top model's per-trial budget is smaller
than the number of requests that want it, so some requests must take the
next-best free model. Replay of trial 0 produces a byte-identical state
export.
