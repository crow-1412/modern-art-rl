# Data Schema

## `EpisodeLog` JSONL
One row per seat per round.

Required fields:
- `episode_id` (string)
- `t` (int): round index
- `seat_id` (string): `seat_0..seat_3`
- `state_hash` (string): hash of public state
- `prompt` (string): model input prompt
- `raw_output` (string): model output text
- `parsed_action` (object): `play_card_index`, `price`, `bid`, `insert_card_index`
- `legal` (bool)
- `reward_delta` (float)
- `bankroll_after` (float)
- `done` (bool)

## `PreferencePair` JSONL
One row per chosen/rejected pair from the same state.

Required fields:
- `pair_id` (string)
- `state_hash` (string)
- `prompt` (string)
- `chosen` (string)
- `rejected` (string)
- `chosen_score` (float)
- `rejected_score` (float)
- `margin` (float)
- `source_episode_ids` (string)

## Scoring default
`score = reward_delta + 0.01 * bankroll_after`

Rows are paired by `(episode_id, t, state_hash)`, selecting max score as chosen and min score as rejected.
Pairs with `margin < threshold` are dropped.
