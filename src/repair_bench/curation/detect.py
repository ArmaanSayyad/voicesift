"""Find timed overlap candidates, not semantic interruption ground truth."""

VERSION = "user-onset-during-agent/v1"


def intervals(turns, role):
    groups = []
    spans = sorted(
        (t["start_s"], t["end_s"], i)
        for i, t in enumerate(turns)
        if t["role"] == role and t.get("start_s") is not None
    )
    for start, end, i in spans:
        if groups and start <= groups[-1]["end"]:
            groups[-1]["end"] = max(end, groups[-1]["end"])
            groups[-1]["indices"].append(i)
        else:
            groups.append({"start": start, "end": end, "indices": [i]})
    return groups


def detect(conversation):
    turns = conversation["turns"]
    found = []
    agents = intervals(turns, "assistant")
    users = intervals(turns, "user")
    for user in users:
        for agent in agents:
            if agent["start"] < user["start"] < agent["end"]:
                end = min(user["end"], agent["end"])
                found.append(
                    {
                        "turn_index": user["indices"][0],
                        "user_turn_indices": user["indices"],
                        "agent_turn_indices": agent["indices"],
                        "user_onset_s": user["start"],
                        "overlap_end_s": end,
                        "overlap_s": end - user["start"],
                        "detector": VERSION,
                        "status": "unreviewed_overlap_candidate",
                    }
                )
                break
    return found, {
        "timed_user_turns": sum(
            t["role"] == "user" and t.get("start_s") is not None for t in turns
        ),
        "untimed_user_turns": sum(
            t["role"] == "user" and t.get("start_s") is None for t in turns
        ),
    }
