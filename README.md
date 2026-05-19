# 🐆 Multi-Gait Controller — Teaching a Robot 3 Ways to Move

![Multi-Gait Demo](results/demo.gif)

---

## What is this?

Imagine teaching a dog three different commands:

- 🐢 **"Easy"** → walk slowly and carefully
- 🐆 **"Go!"** → run as fast as possible  
- 🌿 **"Conserve"** → move efficiently, don't waste energy

That's exactly what this project does — except with a robot, and instead
of voice commands, we use numbers.

One single robot brain learns all three behaviors at the same time.

---

## Why is this hard?

Most robot learning projects teach a robot **one thing only** — just walk forward.

The challenge here is teaching the **same robot brain** to behave completely
differently depending on what you ask it to do.

It's like training one person to be both a ballet dancer AND a sprinter.
The body is the same. The movement style is completely different.

---

## How does the robot learn?

### The reward system (points!)

The robot learns through trial and error — exactly like a video game score:

```
✅ Do what I want   →  get points
❌ Do something bad →  lose points
```

But here's the twist — **the points change depending on the mode:**

| Mode | What gets rewarded | What gets penalised |
|------|--------------------|---------------------|
| 🟣 Prowl | Moving at a calm steady pace | Going too fast or jerky |
| 🟡 Sprint | Moving as FAST as possible | Almost nothing — just go! |
| 🟢 Trot | Moving forward smoothly | Using too much energy |

The robot tries millions of times and slowly figures out:
> *"Oh! When I'm in Sprint mode, going faster gets me more points.  
> When I'm in Trot mode, smooth gentle movements score better."*

---

## How does the robot know which mode it's in?

The robot receives a list of numbers every frame describing its body:
- Where are my joints? 
- How fast am I moving?
- Am I tilting sideways?

We add **3 extra numbers at the end** that tell it the current mode:

```
Prowl mode  →  [...body info...,  1, 0, 0]
Sprint mode →  [...body info...,  0, 1, 0]
Trot mode   →  [...body info...,  0, 0, 1]
```

Every episode (every "life") the robot gets randomly assigned a mode.
Over millions of tries, it learns to read those 3 numbers and
move accordingly.

---

## The robot — HalfCheetah

The HalfCheetah is a 2D cat-like robot with:
- 6 joints (2 hips, 2 knees, 2 feet)
- No head, no arms — just a body and legs
- Lives in a physics simulation (MuJoCo)

It can't fall over sideways (it's 2D) so it can focus entirely
on learning forward locomotion.

---

## What actually happens during training?

```
Step 1:  Robot spawns  →  randomly assigned a mode
Step 2:  Robot tries to move  →  gets reward based on that mode
Step 3:  Robot falls or finishes  →  episode ends
Step 4:  Robot adjusts its "brain" slightly based on what worked
Step 5:  Repeat 3,000,000 times
```

After 3 million attempts (~8 minutes on a laptop!) the robot has
figured out how to move differently for each mode.

---

## What the numbers mean

| What we measured | Result | What it means |
|-----------------|--------|---------------|
| Training attempts | 3,000,000 | How many times the robot tried |
| Training time | ~8 minutes | How long it took on a Mac |
| Parallel robots | 4 | 4 robots training at once |
| Behaviors learned | 3 | Prowl, Sprint, Trot |

---

## Why does this matter in real robotics?

Real robots like **Tesla's Optimus** need to move differently
depending on their situation:

| Situation | Best behavior |
|-----------|--------------|
| Walking near a human | 🟣 Slow and careful |
| Moving across an empty warehouse | 🟡 Fast |
| Working a long 8-hour shift | 🟢 Energy efficient |

A robot that can only do one of these isn't very useful in the real world.
This project is a small step toward robots that can adapt how they move
based on context — just like humans do naturally.

---

## Run it yourself

```bash
# 1. Setup Python environment
conda create -n rl_humanoid python=3.10
conda activate rl_humanoid
pip install gymnasium mujoco stable-baselines3 tensorboard imageio

# 2. Train the robot (takes ~8 minutes)
python train_multigait.py

# 3. Watch it learn live in your browser
tensorboard --logdir ./logs/
# Open http://localhost:6006

# 4. Record a video of all 3 gaits
python demo_multigait.py
```

---

## What's inside this repo

```
📁 multi-gait-controller/
├── multi_gait_env.py      ← The custom environment (reward switching logic)
├── train_multigait.py     ← Training script
├── demo_multigait.py      ← Records the demo video
├── results/
│   └── demo.gif           ← The robot doing all 3 gaits
└── videos/
    └── multigait_demo.mp4 ← Full quality video
```

---

## What's next

- [ ] Let user switch modes with keyboard in real time
- [ ] Add domain randomisation (random floor friction, robot weight)
- [ ] Train on a humanoid robot (2 legs instead of 4)
- [ ] Port to NVIDIA Isaac Lab for faster GPU training
- [ ] Compare PPO vs SAC — which learns gaits better?

---

## The big picture

This project sits at the intersection of two big ideas:

**Reinforcement Learning** — learning by trial, error, and rewards  
**Multi-task Learning** — one brain, many behaviors

Both are fundamental to building robots that can handle the
messy, unpredictable real world.

---

*Built from scratch on a MacBook — no GPU required.*  
*Part of an ongoing robotics portfolio. More projects coming.*
