# Waggle Comparative Evaluation

- Scenarios: 27
- Queries: 69
- Task families: adversarial_paraphrase, cross_scenario_synthesis, decision_delta, factual_recall, multi_session_change, temporal_latest, temporal_original

| System | Hit@k | Exact support | Mean tokens | Median tokens | p95 tokens |
|--------|-------|---------------|-------------|---------------|------------|
| waggle | 88% | 77% | 58.5 | 42.0 | 111.6 |
| rag_naive | 100% | 97% | 150.9 | 151.0 | 165.0 |

## Waggle Query Policy

- `flat`: max_depth=0; families=factual_recall, temporal_latest, temporal_original
- `graph`: max_depth=1,2; families=adversarial_paraphrase, cross_scenario_synthesis, decision_delta, multi_session_change

## Waggle Mode Breakdown

| Mode | Cases | Max depth | Hit@k | Exact support |
|------|-------|-----------|-------|---------------|
| flat | 39 | 0 | 85% | 85% |
| graph | 30 | 1, 2 | 93% | 67% |

## Failure Protocol

- If Waggle token reduction is under 15 percent, inspect whether graph serialization or context assembly is offsetting compression gains.
- If the tuned baseline matches Waggle on retrieval quality, frame the result as efficiency and structure first rather than retrieval superiority.
- If temporal queries do not separate systems, audit whether the corpus actually requires temporal reasoning before expanding claims.
- If multi-session change queries are inconclusive, expand that slice before broadening the whole pilot corpus.

## Query Stress Eval

- Cases: 40
- Families: adversarial_paraphrase, temporal_latest

| System | Hit@k | Exact support | Mean tokens |
|--------|-------|---------------|-------------|
| graph_raw | 98% | 98% | 37.0 |
| graph_hybrid | 98% | 98% | 63.3 |
