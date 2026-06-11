from pathlib import Path

from run_chunked_pipeline import ChunkedPipeline
from data.json_extractor import export_match_data
from visualizations.heatmap import heatmap_by_team, heatmap_for_team
from visualizations.pass_networks import pass_network
from visualizations.json_extractor import (
    assign_ball_touches,
    determine_ball_control,
    detect_passes,
    build_players,
)


def main():
    video_path = 'input_videos/08fd33_4.mp4'
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    pipeline = ChunkedPipeline(
        model_path='models/best.pt',
        chunk_duration=30
    )

    pipeline_result = pipeline.process_video_chunked(
        video_path=video_path,
        output_dir='output_videos',
        use_stub=False,
        return_merged_tracks=True
    )

    tracks = pipeline_result.pop('tracks', None)
    if tracks is None:
        raise RuntimeError("Pipeline did not return merged tracks.")

    counter_path = Path('match_counter.txt')
    match_num = int(counter_path.read_text().strip() or '0') + 1 if counter_path.exists() else 1
    counter_path.write_text(str(match_num))
    match_id = f'match_{match_num:03d}'
    match_dir = f'data/{match_id}'

    export_match_data(tracks, fps=30, match_id=match_id, output_dir=match_dir)

    assign_ball_touches(tracks)
    determine_ball_control(tracks)
    passes, turnovers = detect_passes(tracks)
    players = build_players(tracks, passes, turnovers)

    heatmap_by_team(players, team=1)
    heatmap_by_team(players, team=2)
    heatmap_for_team(players, team=1)
    heatmap_for_team(players, team=2)

    pass_network(players, passes, team=1)
    pass_network(players, passes, team=2)


if __name__ == '__main__':
    main()