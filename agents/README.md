# Ideas

## Brainstorming
- Minimax
- MCST
- Strip
- BFS
- Heuristics
- Knowledge base
- Search optimization
- Inference and domain reduction
- Could try beam search rather than testing all moves

## Reminders
- Do not run test locally
- No external library
- Nbr of turns cap
- 2 sec

## Testing our agent:
```bash
# Simple game
python simulator.py --player_1 random_agent --player_2 steve_agent --display 

# Multiple
python simulator.py --player_1 steve_agent --player_2 random_agent --autoplay --autoplay_runs 100

# Board selection
python simulator.py --player_1 steve_agent --player_2 greedy_corners_agent --autoplay --autoplay_runs 200 --board_path boards/empty_7x7.csv
```
