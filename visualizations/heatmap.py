import mplsoccer
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from pathlib import Path


root = Path(__file__).resolve().parent.parent
json_path = root / 'match_data.json'


def normalize_coords(positions):
    x_coords = [pos["x"] for pos in positions if pos["x"] is not None]
    y_coords = [pos["y"] for pos in positions if pos["y"] is not None]

    if not x_coords or not y_coords:
        return [], []

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Guard against division by zero when a player barely moved (single position)
    x_norm = [(x - x_min) / (x_max - x_min) * 100 if x_max != x_min else 50.0 for x in x_coords]
    y_norm = [(y - y_min) / (y_max - y_min) * 100 if y_max != y_min else 50.0 for y in y_coords]

    return x_norm, y_norm


def heatmap_by_team(players, team, min_appearances=30):
    team_players = [p for p in players if p["team"] == team]
    
    team_players = [
        p for p in team_players 
        if len([pos for pos in p["positions"] if pos["x"] is not None]) >= 30
    ]
    
    # Cap at 11 regardless
    team_players = sorted(
        team_players,
        key=lambda p: len([pos for pos in p["positions"] if pos["x"] is not None]),
        reverse=True
    )[:11]

    
    if not team_players:
        print(f"No players found for team {team}")
        return
    
    n_players = len(team_players)
    ncols = 3
    nrows = (n_players + ncols - 1) // ncols
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#000000", "#ff0000"])

    pitch = mplsoccer.Pitch(
        pitch_type='opta',
        pitch_color='#22312b',
        line_color='#c7d5cc',
        line_zorder=2
    )
    fig, axs = pitch.draw(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4))
    fig.set_facecolor('black')

    if n_players == 1:
        axs_flat = [axs]
    else:
        axs_flat = np.array(axs).flatten()

    for idx, player in enumerate(team_players):
        ax = axs_flat[idx]

        x_coords, y_coords = normalize_coords(player["positions"])

        if len(x_coords) > 10 and len(y_coords) > 10:
            pitch.kdeplot(x_coords, y_coords, ax=ax, fill=True, cmap=custom_cmap, levels=50, zorder=1)
        elif x_coords and y_coords:
            ax.scatter(x_coords, y_coords, color='red', s=5, zorder=1)
        else:
            ax.set_visible(False)
            continue

        ax.set_title(f"Player {player['player_id']}", color='white', fontsize=8)

    for idx in range(len(team_players), len(axs_flat)):
        axs_flat[idx].set_visible(False)

    plt.suptitle(f"Team {team} Heatmaps", color='white', fontsize=14)
    plt.tight_layout()
    plt.savefig(root / f'heatmaps_team_{team}.png', facecolor='black')
    plt.show()


def heatmap_for_team(players, team, min_appearances=50):
    team_players = [p for p in players if p["team"] == team]
    team_players = [p for p in team_players if len([pos for pos in p["positions"] if pos["x"] is not None]) >= min_appearances]

    if not team_players:
        print(f"No players found for team {team}")
        return

    all_positions = []
    for player in team_players:
        all_positions.extend(player["positions"])

    x_coords, y_coords = normalize_coords(all_positions)

    if not x_coords or not y_coords:
        print(f"No valid positions for team {team}")
        return

    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#000000", "#ff0000"])
    pitch = mplsoccer.Pitch(
        pitch_type='opta',
        pitch_color='#22312b',
        line_color='#c7d5cc',
        line_zorder=2
    )
    fig, ax = pitch.draw(figsize=(6, 4))
    fig.set_facecolor('black')

    if len(x_coords) > 10 and len(y_coords) > 10:
        pitch.kdeplot(x_coords, y_coords, ax=ax, fill=True, cmap=custom_cmap, levels=50, zorder=1)
    else:
        ax.scatter(x_coords, y_coords, color='red', s=5, zorder=1)

    ax.set_title(f"Team {team} Heatmap", color='white', fontsize=14)
    plt.tight_layout()
    plt.savefig(root / f'heatmap_team_{team}.png', facecolor='black')
    plt.show()


if __name__ == "__main__":
    with open(json_path, 'r') as f:
        match_data = json.load(f)

    players = match_data["player"]

    heatmap_by_team(players, team=1)
    heatmap_by_team(players, team=2)

    heatmap_for_team(players, team=1)
    heatmap_for_team(players, team=2)