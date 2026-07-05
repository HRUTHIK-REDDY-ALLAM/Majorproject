"""Synthetic benchmark scenario generator.

Generates 10-15 investigative scenarios with:
- Simulated camera topology (5-10 cameras with blind spots)
- Synthetic person tracks with known ground truth
- Access log events aligned with tracks
- Planted false leads
- Witness statements (some accurate, some misleading)
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Camera Topologies ─────────────────────────────────────────

BUILDING_TOPOLOGY = {
    "cameras": [
        {"id": "cam_entrance", "location": "Main Entrance", "position": [0, 0], "coverage_area": "lobby"},
        {"id": "cam_lobby", "location": "Lobby", "position": [10, 0], "coverage_area": "lobby"},
        {"id": "cam_elevator", "location": "Elevator Bank", "position": [20, 0], "coverage_area": "elevator"},
        {"id": "cam_hallway_2f", "location": "2nd Floor Hallway", "position": [20, 10], "coverage_area": "hallway"},
        {"id": "cam_server_room", "location": "Server Room", "position": [30, 10], "coverage_area": "server_room"},
        {"id": "cam_parking", "location": "Parking Garage", "position": [-10, 0], "coverage_area": "parking"},
        {"id": "cam_stairwell", "location": "Stairwell", "position": [15, 5], "coverage_area": "stairwell"},
        {"id": "cam_back_exit", "location": "Back Exit", "position": [30, 0], "coverage_area": "exit"},
    ],
    "connections": [
        {"from": "cam_entrance", "to": "cam_lobby", "travel_time": 15, "distance": 20, "via_blind_spot": False},
        {"from": "cam_lobby", "to": "cam_elevator", "travel_time": 30, "distance": 30, "via_blind_spot": False},
        {"from": "cam_lobby", "to": "cam_stairwell", "travel_time": 20, "distance": 15, "via_blind_spot": True},
        {"from": "cam_elevator", "to": "cam_hallway_2f", "travel_time": 45, "distance": 10, "via_blind_spot": True},
        {"from": "cam_stairwell", "to": "cam_hallway_2f", "travel_time": 60, "distance": 20, "via_blind_spot": True},
        {"from": "cam_hallway_2f", "to": "cam_server_room", "travel_time": 20, "distance": 15, "via_blind_spot": False},
        {"from": "cam_parking", "to": "cam_entrance", "travel_time": 60, "distance": 50, "via_blind_spot": True},
        {"from": "cam_lobby", "to": "cam_back_exit", "travel_time": 45, "distance": 40, "via_blind_spot": True},
        {"from": "cam_server_room", "to": "cam_hallway_2f", "travel_time": 20, "distance": 15, "via_blind_spot": False},
        {"from": "cam_hallway_2f", "to": "cam_elevator", "travel_time": 45, "distance": 10, "via_blind_spot": True},
        {"from": "cam_elevator", "to": "cam_lobby", "travel_time": 30, "distance": 30, "via_blind_spot": False},
        {"from": "cam_lobby", "to": "cam_entrance", "travel_time": 15, "distance": 20, "via_blind_spot": False},
    ],
}

# ── People ────────────────────────────────────────────────────

PEOPLE = [
    {"id": "EMP001", "name": "John Smith", "role": "Software Engineer"},
    {"id": "EMP002", "name": "Jane Doe", "role": "System Administrator"},
    {"id": "EMP003", "name": "Robert Johnson", "role": "Security Guard"},
    {"id": "EMP004", "name": "Emily Chen", "role": "Data Analyst"},
    {"id": "EMP005", "name": "Michael Brown", "role": "IT Manager"},
    {"id": "VIS001", "name": "Alex Wilson", "role": "Visitor"},
    {"id": "VIS002", "name": "Sarah Davis", "role": "Contractor"},
]


def generate_scenario(
    scenario_id: int,
    base_time: datetime | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a single synthetic investigative scenario.

    Returns a complete scenario with ground truth, access logs,
    camera sightings, witness statements, and planted false leads.
    """
    if seed is not None:
        random.seed(seed)

    base_time = base_time or datetime(2025, 1, 15, 8, 0, 0)

    # Select scenario type
    scenario_types = [
        "unauthorized_access",
        "data_theft",
        "after_hours_intrusion",
        "tailgating",
        "insider_threat",
    ]
    scenario_type = scenario_types[scenario_id % len(scenario_types)]

    # Select suspect and decoy
    suspect = random.choice(PEOPLE[:5])
    decoy = random.choice([p for p in PEOPLE if p["id"] != suspect["id"]])

    # Generate the actual suspect path (ground truth)
    suspect_path = _generate_suspect_path(suspect, scenario_type, base_time)

    # Generate other people's normal movements
    normal_movements = []
    for person in PEOPLE:
        if person["id"] != suspect["id"]:
            normal_movements.extend(
                _generate_normal_movement(person, base_time)
            )

    # Generate access logs
    access_logs = _generate_access_logs(suspect_path, normal_movements)

    # Generate camera sightings
    camera_sightings = _generate_camera_sightings(suspect_path, normal_movements)

    # Generate witness statements (some accurate, some misleading)
    statements = _generate_witness_statements(
        suspect, decoy, scenario_type, base_time
    )

    # Plant false leads
    false_leads = _generate_false_leads(decoy, base_time)

    scenario = {
        "id": f"scenario_{scenario_id:02d}",
        "title": _scenario_title(scenario_type, scenario_id),
        "description": _scenario_description(scenario_type),
        "scenario_type": scenario_type,
        "topology": BUILDING_TOPOLOGY,
        "ground_truth": {
            "suspect": suspect,
            "suspect_path": suspect_path,
            "crime_location": "cam_server_room",
            "crime_time": (base_time + timedelta(hours=1, minutes=30)).isoformat(),
            "motive": f"Unauthorized {scenario_type.replace('_', ' ')}",
        },
        "evidence": {
            "access_logs": access_logs,
            "camera_sightings": camera_sightings,
            "witness_statements": statements,
        },
        "false_leads": false_leads,
        "metadata": {
            "num_cameras": len(BUILDING_TOPOLOGY["cameras"]),
            "num_people": len(PEOPLE),
            "num_blind_spots": sum(
                1 for c in BUILDING_TOPOLOGY["connections"]
                if c.get("via_blind_spot")
            ),
            "num_false_leads": len(false_leads),
            "difficulty": "medium" if scenario_id < 5 else "hard",
        },
    }

    return scenario


def _generate_suspect_path(
    suspect: dict, scenario_type: str, base_time: datetime
) -> list[dict]:
    """Generate the actual path the suspect took (ground truth)."""
    path = []
    t = base_time

    # Entry
    path.append({
        "camera_id": "cam_parking",
        "timestamp": t.isoformat(),
        "person": suspect,
        "action": "arrives at parking",
    })

    t += timedelta(minutes=random.randint(3, 8))
    path.append({
        "camera_id": "cam_entrance",
        "timestamp": t.isoformat(),
        "person": suspect,
        "action": "enters building",
    })

    t += timedelta(minutes=random.randint(1, 3))
    path.append({
        "camera_id": "cam_lobby",
        "timestamp": t.isoformat(),
        "person": suspect,
        "action": "crosses lobby",
    })

    # Normal activity for a while
    t += timedelta(minutes=random.randint(30, 60))

    # Suspicious activity — takes stairwell (blind spot) to avoid elevator camera
    if scenario_type in ("unauthorized_access", "data_theft", "insider_threat"):
        path.append({
            "camera_id": "cam_stairwell",
            "timestamp": t.isoformat(),
            "person": suspect,
            "action": "enters stairwell (avoiding elevator camera)",
            "is_suspicious": True,
        })

        t += timedelta(minutes=random.randint(2, 5))
        # Blind spot gap here — not seen until hallway

        t += timedelta(minutes=random.randint(3, 8))
        path.append({
            "camera_id": "cam_hallway_2f",
            "timestamp": t.isoformat(),
            "person": suspect,
            "action": "appears on 2nd floor hallway",
        })
    else:
        path.append({
            "camera_id": "cam_elevator",
            "timestamp": t.isoformat(),
            "person": suspect,
            "action": "takes elevator",
        })
        t += timedelta(minutes=1)
        path.append({
            "camera_id": "cam_hallway_2f",
            "timestamp": t.isoformat(),
            "person": suspect,
            "action": "exits elevator on 2nd floor",
        })

    # Crime scene
    t += timedelta(minutes=random.randint(1, 3))
    path.append({
        "camera_id": "cam_server_room",
        "timestamp": t.isoformat(),
        "person": suspect,
        "action": "enters server room",
        "is_crime_scene": True,
    })

    # Spends time in server room
    t += timedelta(minutes=random.randint(5, 20))
    path.append({
        "camera_id": "cam_server_room",
        "timestamp": t.isoformat(),
        "person": suspect,
        "action": "exits server room",
    })

    # Escape path
    t += timedelta(minutes=random.randint(2, 5))
    path.append({
        "camera_id": "cam_hallway_2f",
        "timestamp": t.isoformat(),
        "person": suspect,
        "action": "returns to hallway",
    })

    # Takes back exit (blind spot)
    t += timedelta(minutes=random.randint(5, 10))
    path.append({
        "camera_id": "cam_back_exit",
        "timestamp": t.isoformat(),
        "person": suspect,
        "action": "exits via back door",
        "is_suspicious": True,
    })

    return path


def _generate_normal_movement(
    person: dict, base_time: datetime
) -> list[dict]:
    """Generate normal (non-suspicious) movement for a person."""
    movements = []
    t = base_time + timedelta(minutes=random.randint(-15, 30))

    # Normal entry
    movements.append({
        "camera_id": "cam_entrance",
        "timestamp": t.isoformat(),
        "person": person,
        "action": "enters building",
    })

    t += timedelta(minutes=random.randint(1, 3))
    movements.append({
        "camera_id": "cam_lobby",
        "timestamp": t.isoformat(),
        "person": person,
        "action": "in lobby",
    })

    # Some people go upstairs
    if random.random() > 0.4:
        t += timedelta(minutes=random.randint(2, 10))
        movements.append({
            "camera_id": "cam_elevator",
            "timestamp": t.isoformat(),
            "person": person,
            "action": "takes elevator",
        })

    return movements


def _generate_access_logs(
    suspect_path: list[dict], normal_movements: list[dict]
) -> list[dict]:
    """Generate badge access log entries."""
    logs = []

    # Suspect logs
    for point in suspect_path:
        if point["camera_id"] in ("cam_entrance", "cam_server_room"):
            logs.append({
                "person_id": point["person"]["id"],
                "person_name": point["person"]["name"],
                "location": point["camera_id"].replace("cam_", "").replace("_", " ").title(),
                "timestamp": point["timestamp"],
                "action": "entry",
            })

    # Normal people logs
    for mov in normal_movements:
        if mov["camera_id"] == "cam_entrance":
            logs.append({
                "person_id": mov["person"]["id"],
                "person_name": mov["person"]["name"],
                "location": "Main Entrance",
                "timestamp": mov["timestamp"],
                "action": "entry",
            })

    logs.sort(key=lambda x: x["timestamp"])
    return logs


def _generate_camera_sightings(
    suspect_path: list[dict], normal_movements: list[dict]
) -> list[dict]:
    """Generate camera detection records."""
    sightings = []

    for point in suspect_path:
        sightings.append({
            "camera_id": point["camera_id"],
            "timestamp": point["timestamp"],
            "person_id": point["person"]["id"],
            "detection_confidence": round(random.uniform(0.7, 0.95), 2),
            "description": point["action"],
        })

    for mov in normal_movements:
        sightings.append({
            "camera_id": mov["camera_id"],
            "timestamp": mov["timestamp"],
            "person_id": mov["person"]["id"],
            "detection_confidence": round(random.uniform(0.6, 0.9), 2),
            "description": mov["action"],
        })

    sightings.sort(key=lambda x: x["timestamp"])
    return sightings


def _generate_witness_statements(
    suspect: dict, decoy: dict, scenario_type: str, base_time: datetime
) -> list[dict]:
    """Generate witness statements — some accurate, some misleading."""
    statements = []

    # Accurate statement
    statements.append({
        "source": "Security Guard Mike",
        "text": (
            f"I noticed {suspect['name']} acting unusual around "
            f"{(base_time + timedelta(hours=1)).strftime('%I:%M %p')}. "
            f"They were looking around nervously near the stairwell entrance. "
            f"Usually they take the elevator."
        ),
        "timestamp": (base_time + timedelta(hours=3)).isoformat(),
        "event_time": (base_time + timedelta(hours=1)).isoformat(),
        "reliability_score": 0.85,
        "is_accurate": True,
    })

    # Partially accurate statement (correct person, wrong time)
    statements.append({
        "source": "Receptionist Lisa",
        "text": (
            f"I saw {suspect['name']} arrive around "
            f"{(base_time + timedelta(minutes=30)).strftime('%I:%M %p')}. "
            f"They seemed to be in a hurry."
        ),
        "timestamp": (base_time + timedelta(hours=3)).isoformat(),
        "event_time": (base_time + timedelta(minutes=30)).isoformat(),
        "reliability_score": 0.7,
        "is_accurate": True,
    })

    # Misleading statement (FALSE LEAD — implicates decoy)
    statements.append({
        "source": "Janitor Dave",
        "text": (
            f"I think I saw {decoy['name']} near the server room around "
            f"{(base_time + timedelta(hours=1, minutes=15)).strftime('%I:%M %p')}. "
            f"I'm not 100% sure though, the lighting was poor."
        ),
        "timestamp": (base_time + timedelta(hours=4)).isoformat(),
        "event_time": (base_time + timedelta(hours=1, minutes=15)).isoformat(),
        "reliability_score": 0.4,
        "is_accurate": False,
        "is_false_lead": True,
    })

    return statements


def _generate_false_leads(decoy: dict, base_time: datetime) -> list[dict]:
    """Generate deliberately planted false leads."""
    return [
        {
            "type": "misleading_witness",
            "description": f"Witness incorrectly identifies {decoy['name']} near crime scene",
            "person_implicated": decoy,
            "should_be_flagged_by_critic": True,
        },
        {
            "type": "coincidental_access",
            "description": f"{decoy['name']} legitimately accessed a nearby area around the same time",
            "person_implicated": decoy,
            "should_be_flagged_by_critic": True,
        },
    ]


def _scenario_title(scenario_type: str, idx: int) -> str:
    titles = {
        "unauthorized_access": f"Unauthorized Server Room Access — Case {idx + 1}",
        "data_theft": f"Suspected Data Exfiltration — Case {idx + 1}",
        "after_hours_intrusion": f"After-Hours Building Intrusion — Case {idx + 1}",
        "tailgating": f"Tailgating Incident at Secure Door — Case {idx + 1}",
        "insider_threat": f"Insider Threat Investigation — Case {idx + 1}",
    }
    return titles.get(scenario_type, f"Investigation Case {idx + 1}")


def _scenario_description(scenario_type: str) -> str:
    descriptions = {
        "unauthorized_access": "Security alarm triggered in the server room. Investigate who accessed the room and how they avoided detection.",
        "data_theft": "Anomalous data transfer detected from a server. Determine who was responsible and what data was accessed.",
        "after_hours_intrusion": "Motion sensors triggered after business hours. Identify the intruder and their entry/exit path.",
        "tailgating": "Access control system logged an entry without a corresponding badge swipe. Investigate the tailgating incident.",
        "insider_threat": "Internal audit flagged suspicious system access patterns. Investigate potential insider threat.",
    }
    return descriptions.get(scenario_type, "Investigate the incident.")


def generate_all_scenarios(
    count: int = 10,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Generate a full benchmark suite of scenarios.

    Args:
        count: Number of scenarios to generate.
        output_dir: Directory to save scenario JSON files.

    Returns:
        List of scenario dicts.
    """
    scenarios = []

    for i in range(count):
        base_time = datetime(2025, 1, 15 + i, 8, 0, 0)
        scenario = generate_scenario(i, base_time=base_time, seed=42 + i)
        scenarios.append(scenario)

        if output_dir:
            out_path = Path(output_dir) / scenario["id"]
            out_path.mkdir(parents=True, exist_ok=True)

            with open(out_path / "ground_truth.json", "w") as f:
                json.dump(scenario, f, indent=2, default=str)

            logger.info(f"Generated scenario: {scenario['id']} → {out_path}")

    logger.info(f"Generated {len(scenarios)} benchmark scenarios")
    return scenarios


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "scenarios"
    generate_all_scenarios(count=10, output_dir=output)
    print(f"Generated 10 scenarios in {output}")
