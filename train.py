"""Train the agent. Headless by default, because drawing caps it at 10 fps.

    python train.py --games 400
    python train.py --games 400 --render      # watch it learn, slowly
"""

import argparse
import csv
import time
from pathlib import Path

from agent import Agent
from game import SnakeGame

ROOT = Path(__file__).parent


def train(games, render, out):
    agent = Agent()
    game = SnakeGame(render=render)

    scores = []
    started = time.perf_counter()

    while agent.n_games < games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if not done:
            continue

        game.reset()
        agent.n_games += 1
        agent.train_long_memory()
        scores.append(score)

        if score > agent.record:
            agent.record = score
            agent.model.save(out)

        if agent.n_games % 25 == 0:
            last = scores[-25:]
            print(
                f"game {agent.n_games:>4}  "
                f"mean(last 25) {sum(last) / len(last):5.2f}  "
                f"record {agent.record:>3}"
            )

    elapsed = time.perf_counter() - started
    print(f"\n{games} games in {elapsed:.1f}s. Record {agent.record}.")
    print(f"Mean over the last 50: {sum(scores[-50:]) / len(scores[-50:]):.2f}")

    with open(ROOT / "scores.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["game", "score"])
        writer.writerows(enumerate(scores, start=1))
    print(f"wrote scores.csv ({len(scores)} rows)")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", default="model.pth")
    args = parser.parse_args()
    train(args.games, args.render, args.out)
