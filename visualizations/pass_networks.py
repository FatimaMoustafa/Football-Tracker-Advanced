import mplsoccer
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


root = Path(__file__).resolve().parent.parent
json_path = root / 'match_data.json'


def _valid_position_count(player):
    return len([pos for pos in player["positions"] if pos["x"] is not None])


def _get_top_players(players, team, min_appearances=30, max_players=11):
    """Filter ghost tracks and cap at max_players by appearance count."""
    team_players = [p for p in players if p["team"] == team]
    team_players = [p for p in team_players if _valid_position_count(p) >= min_appearances]
    team_players = sorted(team_players, key=_valid_position_count, reverse=True)
    return team_players[:max_players]


def _get_avg_positions(team_players):
    """
    Compute each player's average position and normalize to 0-100
    using the team's overall coordinate range.
    """
    all_positions = []
    for player in team_players:
        all_positions.extend(player["positions"])

    valid = [pos for pos in all_positions if pos["x"] is not None and pos["y"] is not None]
    if not valid:
        return {}

    x_all = [pos["x"] for pos in valid]
    y_all = [pos["y"] for pos in valid]
    x_min, x_max = min(x_all), max(x_all)
    y_min, y_max = min(y_all), max(y_all)

    avg_positions = {}
    for player in team_players:
        positions = [pos for pos in player["positions"] if pos["x"] is not None and pos["y"] is not None]
        if not positions:
            continue

        avg_x = np.mean([pos["x"] for pos in positions])
        avg_y = np.mean([pos["y"] for pos in positions])

        norm_x = (avg_x - x_min) / (x_max - x_min) * 100 if x_max != x_min else 50.0
        norm_y = (avg_y - y_min) / (y_max - y_min) * 100 if y_max != y_min else 50.0

        avg_positions[player["player_id"]] = (norm_x, norm_y)

    return avg_positions


def pass_network(players, passes, team, min_appearances=30, max_players=11):
    pitch = mplsoccer.Pitch(
        pitch_type='opta',
        pitch_color='#22312b',
        line_color='#c7d5cc',
        line_zorder=2
    )
    # --- Filter to valid players only ---
    team_players = _get_top_players(players, team, min_appearances, max_players)

    if not team_players:
        print(f"No valid players found for team {team}")
        return

    valid_ids = {p["player_id"] for p in team_players}
    # --- Filter passes: both passer and receiver must be valid players on this team ---
    team_passes = [
        p for p in passes
        if p["team"] == team
        and p["passer_id"] in valid_ids
        and p["receiver_id"] in valid_ids
    ]

    if not team_passes:
        print(f"Not enough passes detected for Team {team} — try a longer video clip.")
        # Draw empty pitch with message
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.set_facecolor('black')
        ax.text(50, 50, "Not enough data\nTry a longer video", 
                color='white', fontsize=16, ha='center', va='center')
        ax.set_title(f"Team {team} Pass Network", color='white', fontsize=14)
        plt.tight_layout()
        plt.savefig(root / f'pass_network_team_{team}.png', facecolor='black')
        plt.show()
        return

    # --- Average positions ---
    avg_positions = _get_avg_positions(team_players)

    # --- Count pass pairs and passes made per player ---
    pair_counts = {}
    passes_made = {}
    for p in team_passes:
        pair = (p["passer_id"], p["receiver_id"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        passes_made[p["passer_id"]] = passes_made.get(p["passer_id"], 0) + 1

    # --- Draw ---
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('black')

    # Draw edges (pass lines)
    max_count = max(pair_counts.values()) if pair_counts else 1
    for (passer_id, receiver_id), count in pair_counts.items():
        if passer_id not in avg_positions or receiver_id not in avg_positions:
            continue

        x_start, y_start = avg_positions[passer_id]
        x_end,   y_end   = avg_positions[receiver_id]

        line_width = (count / max_count) * 8 + 1    # scale 1–9
        alpha      = (count / max_count) * 0.7 + 0.2 # scale 0.2–0.9

        ax.plot(
            [x_start, x_end], [y_start, y_end],
            color='white', linewidth=line_width, alpha=alpha, zorder=3
        )

        mid_x = (x_start + x_end) / 2
        mid_y = (y_start + y_end) / 2
        ax.text(mid_x, mid_y, str(count), color='yellow', fontsize=7,
                ha='center', va='center', zorder=5)

    # Draw nodes (players)
    max_passes = max(passes_made.values()) if passes_made else 1
    for player_id, (x, y) in avg_positions.items():
        size = (passes_made.get(player_id, 0) / max_passes) * 800 + 200  # scale 200–1000

        ax.scatter(x, y, s=size, color='#e74c3c', edgecolors='white',
                   linewidths=2, zorder=4)
        ax.text(x, y, str(player_id), color='white', fontsize=8,
                ha='center', va='center', fontweight='bold', zorder=5)

    ax.set_title(f"Team {team} Pass Network", color='white', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(root / f'pass_network_team_{team}.png', facecolor='black')
    plt.show()


if __name__ == "__main__":
    with open(json_path, 'r') as f:
        match_data = json.load(f)

    players = match_data["player"]
    passes  = match_data["passes"]

    pass_network(players, passes, team=1)
    pass_network(players, passes, team=2)