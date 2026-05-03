# Injection Hardening RL Environment

I built an RL environment in which an LLM agent iteratively hardens a base language model against **indirect prompt injection attacks** — malicious instructions embedded in external content such as retrieved documents, emails, or tool outputs — while preserving the model's helpfulness on legitimate requests.

---

## Architecture

```
injection-hardening-rl/
│
├── agent/                        ← the agent being trained
│   ├── __init__.py
│   ├── base.py                   ← BaseAgent, SYSTEM_PROMPT
│   ├── api_agent.py              ← APIAgent
│   └── local_agent.py            ← LocalModelAgent
│
├── environment/                  ← everything the agent operates inside
│   ├── __init__.py
│   ├── env.py                    ← HardeningEnv
│   ├── judge.py                  ← Judge
│   ├── datasets.py               ← loaders + synthetic fallbacks
│   └── scripts/                  ← starting files the agent can read and modify
│       ├── train.py              ← baseline training script
│       └── eval.py               ← baseline evaluation script
│
├── training/                     ← outer loop that trains the agent
│   ├── __init__.py
│   ├── rl_train.py               ← RLTrainer, RLConfig
│   ├── runner.py                 ← EpisodeRunner (moved from root)
│   ├── rollout.py                ← RolloutCollector
│   ├── buffer.py                 ← TrajectoryBuffer
│   └── checkpoint.py             ← CheckpointManager
│
├── utils/
│   ├── __init__.py
│   └── logging.py                ← MetricsLogger
│
├── submission/
│   └── model_template.py
│
├── tests/
│   └── test_smoke.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Reward Function

```
reward = defense_success_rate × compliance_rate × quality_score
```

| Metric | Description |
|---|---|
| **defense_success_rate** | Fraction of injection attacks the model resists |
| **compliance_rate** | Fraction of benign requests answered without blanket refusal |
| **quality_score** | Average response quality on benign requests (heuristic or LLM-as-judge) |

The multiplicative formula ensures the model cannot collapse on any one axis
(e.g. refuse everything → `compliance_rate = 0 → reward = 0`).
