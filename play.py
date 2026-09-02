"""Watch the trained agent, or record it.

    python play.py                       # a window, until you close it
    python play.py --record demo.gif     # no window, writes a GIF
"""

import argparse
import os

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--record", metavar="PATH", help="write a GIF instead of opening a window")
    parser.add_argument("--max-frames", type=int, default=400, help="cap GIF length")
    args = parser.parse_args()

    if args.record:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame

    from game import SnakeGame
    from model import Linear_QNet

    model = Linear_QNet(11, 256, 3).load(args.model)
    game = SnakeGame(render=not args.record, fps=args.fps)

    # Reuse the agent's encoder rather than reimplementing it: training and
    # playing must not drift apart about what the network is being shown.
    from agent import Agent

    encode = Agent.get_state

    frames = []
    scores = []
    for _ in range(args.games):
        game.reset()
        while True:
            state = torch.tensor(encode(game), dtype=torch.float)
            with torch.no_grad():
                move = torch.argmax(model(state)).item()
            action = [0, 0, 0]
            action[move] = 1

            _, done, score = game.play_step(action)
            if args.record:
                frames.append(
                    np.transpose(pygame.surfarray.array3d(game.display), (1, 0, 2)).copy()
                )
            if done:
                scores.append(score)
                break

    print(f"scores: {scores}  mean {sum(scores) / len(scores):.1f}")

    if args.record:
        from PIL import Image

        # A good game runs to several hundred steps; keep the GIF to something a
        # README will actually load by taking an even sample rather than a prefix,
        # so it still shows the snake at full length at the end.
        if len(frames) > args.max_frames:
            step = len(frames) / args.max_frames
            frames = [frames[int(i * step)] for i in range(args.max_frames)]
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            args.record,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / args.fps),
            loop=0,
            optimize=True,
        )
        print(f"wrote {args.record}  ({len(images)} frames)")

    pygame.quit()


if __name__ == "__main__":
    main()
