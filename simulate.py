import os
import imageio
from stable_baselines3 import PPO
from config import DEFAULT_MODEL_PATH
from sailing_env import MultiAgentSailingZoo

def load_model(model_path: str = DEFAULT_MODEL_PATH) -> PPO | None:
    """Load a trained PPO model; returns *None* if not found."""
    try:
        model = PPO.load(model_path)
        print(f"Loaded model: {model_path}")
        return model
    except FileNotFoundError:
        print("No model found — random actions will be used.")
        return None

def run_episode(
    model: PPO | None,
    field_size: int = 400,
    max_steps: int = 250,
    render: bool = True,
) -> dict:
    """Run a single episode and return frames + metadata.

    Returns:
        dict with keys: frames, steps, winner, infos, env (for post-hoc inspection).
    """
    render_mode = "rgb_array" if render else None
    env = MultiAgentSailingZoo(
        field_size=field_size, max_steps=max_steps, render_mode=render_mode
    )
    observations, infos = env.reset()

    # Seed recording with initial state for 3D
    trajectories = {a: [list(env.trajectories[a][0])] for a in env.possible_agents}
    headings = {a: [env.boat_states[a]['heading']] for a in env.possible_agents}
    speeds = {a: [env.boat_states[a]['speed']] for a in env.possible_agents}
    wind_directions = [float(env.wind_direction)]

    frames = []
    if render:
        frames.append(env.render())

    step = 0
    last_infos = infos

    while env.agents and step < max_steps:
        actions = {}
        for agent_id in env.agents:
            obs = observations[agent_id]
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space(agent_id).sample()
            actions[agent_id] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        if infos:
            last_infos = infos

        # Collect 3D data
        wind_directions.append(float(env.wind_direction))
        for a in env.possible_agents:
            s = env.boat_states[a]
            trajectories[a].append([float(s['x']), float(s['y'])])
            headings[a].append(float(s['heading']))
            speeds[a].append(float(s['speed']))

        if render:
            frames.append(env.render())
        step += 1

    if env.winner and render and frames:
        for _ in range(15):
            frames.append(frames[-1])

    result = {
        "frames": frames,
        "steps": step,
        "winner": env.winner,
        "infos": last_infos,
        "env": env,
        "trajectories": trajectories,
        "headings": headings,
        "speeds": speeds,
        "wind_directions": wind_directions,
        "target": [float(env.target[0]), float(env.target[1])],
        "field_size": field_size,
        "wind_speed": float(env.wind_speed),
    }
    env.close()
    return result

def run_episode_3d(
    model,
    field_size: int = 400,
    max_steps: int = 250,
) -> dict:
    """Run a single episode without rendering and collect 3D trajectory data.

    Returns a dict with:
        trajectories  - {agent: [[x, y], ...]}   (one entry per step including reset)
        headings      - {agent: [heading_rad, ...]}
        speeds        - {agent: [speed, ...]}
        wind_directions - [wind_dir_rad, ...]     (one per step)
        target        - [x, y]
        field_size    - int
        wind_speed    - float (initial)
        winner        - str | None
        steps         - int
        infos         - last info dict
    """
    env = MultiAgentSailingZoo(field_size=field_size, max_steps=max_steps, render_mode=None)
    observations, infos = env.reset()

    # Seed recording with initial state
    trajectories = {a: [list(env.trajectories[a][0])] for a in env.possible_agents}
    headings = {a: [env.boat_states[a]['heading']] for a in env.possible_agents}
    speeds = {a: [env.boat_states[a]['speed']] for a in env.possible_agents}
    wind_directions = [float(env.wind_direction)]

    step = 0
    last_infos = infos

    while env.agents and step < max_steps:
        actions = {}
        for agent_id in env.agents:
            obs = observations[agent_id]
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space(agent_id).sample()
            actions[agent_id] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        if infos:
            last_infos = infos

        wind_directions.append(float(env.wind_direction))

        for a in env.possible_agents:
            s = env.boat_states[a]
            trajectories[a].append([float(s['x']), float(s['y'])])
            headings[a].append(float(s['heading']))
            speeds[a].append(float(s['speed']))

        step += 1

    env.close()

    return {
        "trajectories": trajectories,
        "headings": headings,
        "speeds": speeds,
        "wind_directions": wind_directions,
        "target": [float(env.target[0]), float(env.target[1])],
        "field_size": field_size,
        "wind_speed": float(env.wind_speed),
        "winner": env.winner,
        "steps": step,
        "infos": last_infos,
        "env": env,
    }


def generate_videos(
    num_episodes: int = 10,
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = ".",
    field_size: int = 400,
    max_steps: int = 250,
    fps: int = 15,
) -> list[str]:
    """Run *num_episodes* episodes and save each as an mp4 video.

    Returns:
        List of saved video file paths.
    """
    model = load_model(model_path)
    saved: list[str] = []

    for i in range(num_episodes):
        print(f"\n--- Episode {i + 1}/{num_episodes} ---")
        result = run_episode(model, field_size=field_size, max_steps=max_steps, render=True)

        if result["winner"]:
            print(f"   Target reached by {result['winner']}!")
        else:
            print(f"   No winner (steps={result['steps']})")

        video_path = os.path.join(output_dir, f"multi_sailing_demo_{i}.mp4")
        imageio.mimsave(video_path, result["frames"], fps=fps)
        print(f"   Saved {video_path} ({len(result['frames'])} frames)")
        saved.append(video_path)

    return saved