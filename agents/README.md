# Ideas

## Todos
- [ ] Optimize the evaluation function (finetune heuristics)
- [ ] MCTS param optimization and transposition table
- [ ] Determine switch trigger
- [ ] New heurisitcs (aka agressive for the walls map or smth)
- [ ] Separate moves by types (duplicate vs move) + (exploration vs consolidation of neighbours)

## Brainstorming
- Minimax
- MCST
- BFS
- Heuristics
- Knowledge base, transposition table
- Search optimization
- Inference and domain reduction

## Reminders
- Do not run test locally
- No external library

## Testing our agent:
```bash
# Custom simulator wrapper (test over all maps with turn swap)
python '.\run_experiments.py'

# Simple game
python simulator.py --player_1 greedy_corners_agent --player_2 steve_agent --display 

# Multiple
python simulator.py --player_1 steve_agent --player_2 random_agent --autoplay --autoplay_runs 100
python simulator.py --player_1 avram_agent --player_2 greedy_corners_agent --autoplay --autoplay_runs 100

# Board selection
python simulator.py --player_1 steve_agent --player_2 greedy_corners_agent --autoplay --autoplay_runs 200 --board_path boards/empty_7x7.csv
```
