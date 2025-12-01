import os
import logging
from types import SimpleNamespace
from simulator import Simulator # Imports your provided class

# TODO: finetuning loop for the mcts parameters + depths + ratio to swtich to greedy

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)

def run_specific_maps():
    # --- CONFIGURATION ---
    map_folder = "boards"  # The folder where your 5 maps are
    player_1 = "greedy_corners_agent"  # Change to your agent name
    player_2 = "student_agent"
    # ---------------------

    # 2. Get list of all map files
    if not os.path.exists(map_folder):
        print(f"Error: Folder '{map_folder}' does not exist.")
        return

    # Get all .board or .csv files and sort them so order is consistent
    map_files = [
        os.path.join(map_folder, f) 
        for f in os.listdir(map_folder) 
        if f.endswith(".board") or f.endswith(".csv")
    ]
    
    # Take exactly the first 5 (or however many are there)

    print(f"Found {len(map_files)} maps. Starting sequential run...")

    # 3. Create Mock Arguments
    # We create a fake arguments object to satisfy the Simulator class
    args = SimpleNamespace(
        player_1=player_1,
        player_2=player_2,
        board_path=None, 
        board_roster_dir=map_folder, # Not strictly needed for this logic, but good practice
        display=False,               # Set True if you want to watch them
        display_delay=0,
        display_save=False,
        display_save_path="plots/",
        autoplay=False,              # We are handling the looping manually
        autoplay_runs=0
    )

    # 4. Initialize Simulator
    sim = Simulator(args)

    # 5. Loop through the maps and run one game per map
    results = []
    
    for i, map_path in enumerate(map_files):
        print(f"\n--- Game {i+1}/{len(map_files)} ---")
        print(f"Map: {map_path}")

        for s in range(2):
            # Swap players every other game to ensure fairness
            swap = (s % 2 != 0)
            
            # Manually run the simulator on this specific map
            p0_score, p1_score, p0_time, p1_time = sim.run(swap_players=swap, board_fpath=map_path)
            
            # Store who won
            # Note: If swap was True, p0_score is actually Player 2's score
            real_p1_score = p1_score if swap else p0_score
            real_p2_score = p0_score if swap else p1_score
            
            winner = player_1 if real_p1_score > real_p2_score else player_2
            if real_p1_score == real_p2_score: winner = "Tie"
            
            results.append({
                "map": os.path.basename(map_path),
                "winner": winner,
                "score": f"{real_p1_score}-{real_p2_score}"
            })

    # 6. Print Final Report
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    for res in results:
        print(f"Map: {res['map']:<15} | Winner: {res['winner']:<15} | Score: {res['score']}")

if __name__ == "__main__":
    # Import your custom agent here if needed
    try:
        from agents.steve_agent import SteveAgent
    except ImportError:
        pass # Ignore if using built-in agents

    run_specific_maps()