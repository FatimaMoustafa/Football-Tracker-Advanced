from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PitchConfig:
    length_m: float = 105.0
    width_m: float = 68.0


@dataclass
class ExporterConfig:
    fps: float = 30.0
    match_id: str = "match_001"
    pitch: PitchConfig = field(default_factory=PitchConfig)
    team_names: dict[int, str] = field(default_factory=lambda: {1: "Team A", 2: "Team B"})
    tackle_distance_m: float = 1.5
    foul_distance_m: float = 1.0
    foul_speed_drop_threshold: float = 2.0
    ball_state_possession_radius: float = 2.0


def _frame_to_time(frame: int, fps: float) -> float:
    return round(frame / fps, 2)


def _safe_pos(track: dict[str, Any]) -> tuple[float, float] | None:
    pos = track.get("position_transformed") or track.get("transformed_position")
    if pos is None:
        return None
    if isinstance(pos, (list, tuple)) and len(pos) == 2:
        x, y = pos
        if x is not None and y is not None:
            return round(float(x), 3), round(float(y), 3)
    return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class MatchIQExporter:
    """Exports pipeline tracks into clean MatchIQ JSON files."""

    def __init__(
        self,
        tracks: dict[str, list[dict[int, dict[str, Any]]]],
        fps: float = 30.0,
        match_id: str = "match_001",
        output_dir: str | None = None,
        config: ExporterConfig | None = None,
    ) -> None:
        self.tracks = tracks
        self.config = config or ExporterConfig(fps=fps, match_id=match_id)
        resolved_dir = output_dir if output_dir is not None else f"data/{match_id}"
        self.output_dir = Path(resolved_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._total_frames = len(tracks.get("players", []))
        self._fps = self.config.fps

    def export_all(self) -> dict[str, Path]:
        paths = {
            "raw_tracking":        self._write("raw_tracking.json", self._build_raw_tracking()),
            "processed_positions": self._write("processed_positions.json", self._build_processed_positions()),
            "events":              self._write("events.json", self._build_events()),
            "stats":               self._write("stats.json", self._build_stats()),
        }
        print(f"[MatchIQExporter] Exported 4 files to '{self.output_dir}/'")
        return paths

    def _build_raw_tracking(self) -> dict[str, Any]:
        cfg = self.config
        duration_s = round(self._total_frames / self._fps, 2)

        player_frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
        ref_frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
        ball_frames: list[dict[str, Any]] = []

        player_tracks = self.tracks.get("players", [])
        ref_tracks = self.tracks.get("referees", [])
        ball_tracks = self.tracks.get("ball", [])

        for frame_num, frame_data in enumerate(player_tracks):
            t = _frame_to_time(frame_num, self._fps)
            for player_id, track in frame_data.items():
                bbox = track.get("bbox")
                if bbox is None:
                    continue
                confidence = round(float(track.get("confidence", 0.9)), 3)
                player_frames[player_id].append({
                    "frame":        frame_num,
                    "time":         t,
                    "bbox":         [round(float(v), 1) for v in bbox],
                    "confidence":   confidence,
                    "interpolated": bool(track.get("interpolated", False)),
                })

        for frame_num, frame_data in enumerate(ref_tracks):
            t = _frame_to_time(frame_num, self._fps)
            for ref_id, track in frame_data.items():
                bbox = track.get("bbox")
                if bbox is None:
                    continue
                ref_frames[ref_id].append({
                    "frame": frame_num,
                    "time":  t,
                    "bbox":  [round(float(v), 1) for v in bbox],
                })

        for frame_num, frame_data in enumerate(ball_tracks):
            t = _frame_to_time(frame_num, self._fps)
            ball_info = frame_data.get(1, {})
            bbox = ball_info.get("bbox")
            if bbox is None:
                continue
            ball_frames.append({
                "frame":        frame_num,
                "time":         t,
                "bbox":         [round(float(v), 1) for v in bbox],
                "interpolated": bool(ball_info.get("interpolated", False)),
            })

        players_raw = [
            {"id": pid, "raw_frames": frames}
            for pid, frames in sorted(player_frames.items())
        ]
        referees_raw = [
            {"id": rid, "raw_frames": frames}
            for rid, frames in sorted(ref_frames.items())
        ]

        return {
            "metadata": {
                "match_id":         cfg.match_id,
                "fps":              self._fps,
                "total_frames":     self._total_frames,
                "duration_seconds": duration_s,
                "pitch": {
                    "length_m": cfg.pitch.length_m,
                    "width_m":  cfg.pitch.width_m,
                },
                "coordinate_system": {
                    "type":        "metric",
                    "origin":      "top_left",
                    "x_direction": "right",
                    "y_direction": "down",
                },
                "homography_applied": True,
            },
            "players":  players_raw,
            "referees": referees_raw,
            "ball": {
                "raw_frames": ball_frames,
            },
        }

    def _build_processed_positions(self) -> dict[str, Any]:
        player_tracks = self.tracks.get("players", [])
        ball_tracks = self.tracks.get("ball", [])

        player_data: dict[int, dict[str, Any]] = {}

        for frame_num, frame_data in enumerate(player_tracks):
            t = _frame_to_time(frame_num, self._fps)
            for player_id, track in frame_data.items():
                pos = _safe_pos(track)
                if pos is None:
                    continue
                x, y = pos
                speed = round(float(track.get("speed", 0.0) or 0.0), 3)
                team_id = int(track.get("team", 0))

                vx = 0.0
                vy = 0.0
                if player_id in player_data and player_data[player_id]["tracking"]:
                    prev = player_data[player_id]["tracking"][-1]
                    dt = t - prev["time"]
                    if dt > 0:
                        vx = round((x - prev["x"]) / dt, 3)
                        vy = round((y - prev["y"]) / dt, 3)

                direction_deg: float | None = None
                if vx != 0 or vy != 0:
                    direction_deg = round(math.degrees(math.atan2(vy, vx)) % 360, 1)

                frame_entry: dict[str, Any] = {
                    "frame":        frame_num,
                    "time":         t,
                    "x":            x,
                    "y":            y,
                    "speed_mps":    speed,
                    "velocity":     {"vx": vx, "vy": vy},
                    "confidence":   round(float(track.get("confidence", 0.9)), 3),
                    "interpolated": bool(track.get("interpolated", False)),
                }
                if direction_deg is not None:
                    frame_entry["direction_deg"] = direction_deg

                if player_id not in player_data:
                    player_data[player_id] = {
                        "id":      player_id,
                        "team_id": team_id,
                        "tracking": [],
                    }

                player_data[player_id]["tracking"].append(frame_entry)
                if team_id:
                    player_data[player_id]["team_id"] = team_id

        possession_map = self._build_possession_map()

        ball_tracking: list[dict[str, Any]] = []
        for frame_num, frame_data in enumerate(ball_tracks):
            t = _frame_to_time(frame_num, self._fps)
            ball_info = frame_data.get(1, {})
            pos = _safe_pos(ball_info)
            if pos is None:
                continue
            x, y = pos
            entry: dict[str, Any] = {
                "frame": frame_num,
                "time":  t,
                "x":     x,
                "y":     y,
                "confidence": round(float(ball_info.get("confidence", 0.85)), 3),
                "state": "in_play",
                "interpolated": bool(ball_info.get("interpolated", False)),
            }
            if frame_num in possession_map:
                entry["possessed_by"] = possession_map[frame_num]
            ball_tracking.append(entry)

        return {
            "players": sorted(player_data.values(), key=lambda p: p["id"]),
            "ball": {
                "tracking": ball_tracking,
            },
        }

    def _build_events(self) -> dict[str, Any]:
        cfg = self.config
        player_tracks = self.tracks.get("players", [])
        possession_map = self._build_possession_map()

        events: list[dict[str, Any]] = []
        event_counter = 0

        def get_player_pos(frame_data: dict[int, dict[str, Any]], pid: int) -> tuple[float, float] | None:
            track = frame_data.get(pid)
            if track is None:
                return None
            return _safe_pos(track)

        def get_player_speed(frame_data: dict[int, dict[str, Any]], pid: int) -> float:
            track = frame_data.get(pid)
            if track is None:
                return 0.0
            return float(track.get("speed", 0.0) or 0.0)

        prev_possessor: int | None = None

        for frame_num, frame_data in enumerate(player_tracks):
            if frame_num == 0:
                prev_possessor = possession_map.get(0, {}).get("player_id")
                continue

            t = _frame_to_time(frame_num, self._fps)
            current_possession = possession_map.get(frame_num)
            current_possessor = current_possession["player_id"] if current_possession else None

            if (
                prev_possessor is not None
                and current_possessor is not None
                and current_possessor != prev_possessor
            ):
                old_team = possession_map.get(frame_num - 1, {}).get("team_id", 0)
                new_team = current_possession.get("team_id", 0) if current_possession else 0
                pos = get_player_pos(frame_data, current_possessor)
                event_type = "turnover" if old_team != new_team else "pass"
                event_counter += 1
                ev: dict[str, Any] = {
                    "id":         f"event_{event_counter:04d}",
                    "type":       event_type,
                    "frame":      frame_num,
                    "time":       t,
                    "players": {
                        "primary":   current_possessor,
                        "secondary": prev_possessor,
                    },
                    "confidence": 0.78,
                }
                if pos:
                    ev["location"] = {"x": pos[0], "y": pos[1]}
                events.append(ev)

            player_ids = list(frame_data.keys())
            for i, pid_a in enumerate(player_ids):
                pos_a = get_player_pos(frame_data, pid_a)
                if pos_a is None:
                    continue
                for pid_b in player_ids[i + 1:]:
                    pos_b = get_player_pos(frame_data, pid_b)
                    if pos_b is None:
                        continue

                    d = _dist(pos_a, pos_b)

                    if d < cfg.tackle_distance_m and (
                        prev_possessor in (pid_a, pid_b)
                        and current_possessor in (pid_a, pid_b)
                        and prev_possessor != current_possessor
                    ):
                        winner = current_possessor
                        loser = prev_possessor
                        mid_x = round((pos_a[0] + pos_b[0]) / 2, 3)
                        mid_y = round((pos_a[1] + pos_b[1]) / 2, 3)
                        event_counter += 1
                        events.append({
                            "id":       f"event_{event_counter:04d}",
                            "type":     "tackle",
                            "frame":    frame_num,
                            "time":     t,
                            "location": {"x": mid_x, "y": mid_y},
                            "players":  {"winner": winner, "loser": loser},
                            "confidence": round(max(0.5, 0.9 - d / cfg.tackle_distance_m * 0.3), 2),
                        })

                    elif d < cfg.foul_distance_m:
                        speed_a = get_player_speed(frame_data, pid_a)
                        speed_b = get_player_speed(frame_data, pid_b)

                        if frame_num > 0:
                            prev_frame_data = player_tracks[frame_num - 1]
                            prev_speed_a = get_player_speed(prev_frame_data, pid_a)
                            prev_speed_b = get_player_speed(prev_frame_data, pid_b)
                            drop_a = prev_speed_a - speed_a
                            drop_b = prev_speed_b - speed_b

                            attacker, defender, drop = None, None, 0.0
                            if drop_b > cfg.foul_speed_drop_threshold:
                                attacker, defender, drop = pid_a, pid_b, drop_b
                            elif drop_a > cfg.foul_speed_drop_threshold:
                                attacker, defender, drop = pid_b, pid_a, drop_a

                            if attacker is not None:
                                mid_x = round((pos_a[0] + pos_b[0]) / 2, 3)
                                mid_y = round((pos_a[1] + pos_b[1]) / 2, 3)
                                event_counter += 1
                                events.append({
                                    "id":       f"event_{event_counter:04d}",
                                    "type":     "possible_foul",
                                    "frame":    frame_num,
                                    "time":     t,
                                    "location": {"x": mid_x, "y": mid_y},
                                    "players":  {"attacker": attacker, "defender": defender},
                                    "confidence": round(
                                        min(0.85, 0.4 + (drop / (cfg.foul_speed_drop_threshold * 3))),
                                        2,
                                    ),
                                })

            prev_possessor = current_possessor

        return {
            "match_id": self.config.match_id,
            "total_events": len(events),
            "events": events,
        }

    def _build_stats(self) -> dict[str, Any]:
        player_tracks = self.tracks.get("players", [])
        possession_map = self._build_possession_map()

        player_speeds: dict[int, list[float]] = defaultdict(list)
        player_distances: dict[int, float] = defaultdict(float)
        player_team: dict[int, int] = {}

        zone_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        pitch_l = self.config.pitch.length_m
        pitch_w = self.config.pitch.width_m

        def zone_id(x: float, y: float) -> int:
            col = min(int(x / pitch_l * 3), 2)
            row = min(int(y / pitch_w * 3), 2)
            return row * 3 + col

        for frame_num, frame_data in enumerate(player_tracks):
            for player_id, track in frame_data.items():
                speed = float(track.get("speed", 0.0) or 0.0)
                dist = float(track.get("distance", 0.0) or 0.0)
                team = int(track.get("team", 0))

                player_speeds[player_id].append(speed)
                player_distances[player_id] = max(player_distances[player_id], dist)
                if team:
                    player_team[player_id] = team

                pos = _safe_pos(track)
                if pos:
                    zone_counts[player_id][zone_id(pos[0], pos[1])] += 1

        team_possession_counts: dict[int, int] = defaultdict(int)
        for poss in possession_map.values():
            team_id = poss.get("team_id", 0)
            if team_id:
                team_possession_counts[team_id] += 1
        total_possession_frames = sum(team_possession_counts.values()) or 1

        team_stats = []
        for tid, name in self.config.team_names.items():
            pct = round(team_possession_counts.get(tid, 0) / total_possession_frames * 100, 1)
            team_stats.append({
                "id":                 tid,
                "name":               name,
                "possession_percent": pct,
            })

        player_stats = []
        for player_id in sorted(set(player_speeds) | set(player_distances)):
            speeds = player_speeds[player_id]
            avg_speed = round(sum(speeds) / len(speeds), 3) if speeds else 0.0
            max_speed = round(max(speeds), 3) if speeds else 0.0
            total_dist = round(player_distances[player_id], 2)
            heatmap = [
                {"zone_id": zid, "touch_count": cnt}
                for zid, cnt in sorted(zone_counts[player_id].items())
            ]
            player_stats.append({
                "id":                player_id,
                "team_id":           player_team.get(player_id, 0),
                "distance_covered_m": total_dist,
                "avg_speed_mps":     avg_speed,
                "max_speed_mps":     max_speed,
                "heatmap_zones":     heatmap,
            })

        return {
            "match_id": self.config.match_id,
            "players":  player_stats,
            "teams":    team_stats,
        }

    def _build_possession_map(self) -> dict[int, dict[str, int]]:
        player_tracks = self.tracks.get("players", [])
        result: dict[int, dict[str, int]] = {}
        last: dict[str, int] | None = None

        for frame_num, frame_data in enumerate(player_tracks):
            found = False
            for player_id, track in frame_data.items():
                if track.get("has_ball"):
                    team_id = int(track.get("team", 0))
                    result[frame_num] = {"player_id": player_id, "team_id": team_id}
                    last = result[frame_num]
                    found = True
                    break
            if not found and last is not None:
                result[frame_num] = last

        return result

    def _write(self, filename: str, data: dict[str, Any]) -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        size_kb = path.stat().st_size // 1024
        print(f"  ✓ {filename:35s} ({size_kb} KB)")
        return path


def export_match_data(
    tracks: dict[str, Any],
    fps: float = 30.0,
    match_id: str = "match_001",
    output_dir: str | None = None,
    team_names: dict[int, str] | None = None,
) -> dict[str, Path]:
    config = ExporterConfig(fps=fps, match_id=match_id)
    if team_names:
        config.team_names = team_names

    exporter = MatchIQExporter(
        tracks=tracks,
        fps=fps,
        match_id=match_id,
        output_dir=output_dir,
        config=config,
    )
    return exporter.export_all()
