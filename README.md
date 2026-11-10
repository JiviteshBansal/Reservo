# RL Restaurant Booking Chatbot

Reinforcement learning for dialogue policy in restaurant booking, using the
**MultiWOZ 2.1** dataset (restaurant domain only).

Dataset Link - https://www.kaggle.com/datasets/taejinwoo/multiwoz-22?resource=download

The agent learns to manage a conversation — asking for preferences, querying a
database, offering venues, handling bookings — entirely through trial and error
via a DQN trained against an agenda-based user simulator.

## Results

Evaluated on 1,000 held-out user goals (greedy, 3 seeds averaged):

| Policy | Success | Avg Reward | Avg Turns |
|--------|:-------:|:----------:|:---------:|
| Random baseline | 13.8% | −30.4 | 8.2 |
| Behaviour cloning (MLP) | 2.7% | −41.7 | 14.5 |
| Rule-based (hand-coded) | 100.0% | 33.3 | 6.7 |
| DQN (from scratch) | 100.0% | 36.4 | 3.2 |
| DQN (warm-started from BC) | 100.0% | 36.8 | 3.1 |

The learned policy matches the hand-coded one on success rate while finishing
dialogues in about half the turns.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+, CPU is fine. See `requirements.txt` for packages (PyTorch,
Gymnasium, NumPy, Pandas, Matplotlib, etc.)

## How to run

Each script is meant to be run from the project root. They build on each
other, so run them in order the first time:

```bash
python3 scripts/prepare_data.py        # download MultiWOZ, extract DB & goals
python3 scripts/test_simulator.py      # sanity-check the user simulator
python3 scripts/test_environment.py    # validate the gym environment
python3 scripts/evaluate_baselines.py  # rule-based & random baselines
python3 scripts/train_supervised.py    # behaviour cloning (for comparison)
python3 scripts/train_dqn.py           # DQN training — cold & warm start
python3 scripts/run_experiments.py     # ablation studies
```

The DQN training (`train_dqn.py`) takes ~25 min on a MacBook. Everything else
is under a minute. The experiment script reuses saved models where possible, so
re-running is fast.

## Project layout

```
├── config/config.yaml           # hyperparameters, paths, reward values
├── data/
│   ├── raw/                     # MultiWOZ 2.1 (auto-downloaded)
│   └── processed/               # extracted DB, goals, state-action pairs
├── src/
│   ├── utils.py                 # constants, seeding, config loader
│   ├── data_loader.py           # data extraction from MultiWOZ
│   ├── user_simulator.py        # agenda-based user simulator
│   ├── environment.py           # gymnasium env wrapping the simulator
│   └── policies/
│       ├── rule_based.py        # hand-coded + random policies
│       ├── supervised.py        # MLP for behaviour cloning
│       └── dqn.py               # DQN agent, replay buffer, training loop
├── scripts/                     # one script per stage (see above)
├── results/
│   ├── models/                  # saved .pt checkpoints
│   └── plots/                   # all generated figures
├── requirements.txt
└── README.md
```

## Domain details

Working with the restaurant domain from MultiWOZ 2.1:

- **Database**: 110 Cambridge restaurants
- **Informable slots**: food type, area, price range
- **Requestable slots**: address, phone, postcode, booking reference
- **Booking slots**: number of people, day, time
- **Action space**: 15 system dialogue acts (request_food, confirm_area, offer_venue, inform_phone, book, bye, etc.)
- **State vector**: 28 floats (filled slots, DB match bins, last user act, pending requests, status flags, turn count)

## Ablations

The experiment script (`run_experiments.py`) covers four studies:

**Reward shaping** — Removing the penalties for redundant/invalid actions
drops success from 100% to 59%. The shaped rewards give a much denser learning
signal that's critical for convergence.

**Training noise** — Training the simulator with 10% slot-value noise gives
the best robustness across noise levels. A model trained at 0% noise drops to
~67% success when tested at 20% noise; the 10%-trained model holds at ~73%.

**Cold vs warm start** — Both reach the same ceiling (100%), but warm-starting
from the supervised policy converges about 2× faster and has lower variance
across seeds.

**Goal complexity** — The DQN policy handles simple (≤1 constraint), medium
(2), and complex (3) goals equally well. Its biggest advantage over the
rule-based policy is on simple goals, where the hand-coded confirmation
protocol wastes the most turns.

## Reproducibility

- All hyperparameters in `config/config.yaml`
- Seeds: 42, 123, 456 (used throughout)
- 1,000 fixed eval goals saved in `data/processed/eval_goals_1000.json`
- Deterministic seeding via `src/utils.set_seed()`

## Contributors
Jivitesh Bansal
Sanchit Luthra
Jasman Kaur
