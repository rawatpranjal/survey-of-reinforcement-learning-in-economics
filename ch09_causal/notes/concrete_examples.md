# Concrete Examples for Causal RL Landscape (Plain English)

## Example 1: Sim-to-real robotics (causal representation learning)

**The problem:** You train a robot arm to pick up objects inside a simulator. The simulator has a blue background and simple textures. The robot learns to pick things up perfectly in simulation, but fails completely on a real factory floor with different lighting and backgrounds. Why? Because the neural network learned "blue background = reach left" instead of "object is at position X = reach left." It memorized visual quirks that happened to correlate with success in training but mean nothing in reality.

**Why standard RL fails:** A standard RL agent treats all input pixels equally. If blue pixels always appeared when the correct action was "move left," the agent learns that association. It has no way to distinguish "this pixel pattern causes success" from "this pixel pattern coincidentally appeared during success."

**The causal fix:** Force the agent's state representation to be invariant across multiple visually different training environments (same physics, different textures/lighting). If the representation predicts equally well in all environments, the visual quirks have been stripped out, and only the causally relevant features (object position, joint angles) remain.

## Example 2: Atari Breakout (counterfactual policy optimization)

**The problem:** In Breakout, you move a paddle to bounce a ball into bricks. The reward (breaking a brick) happens many timesteps after the critical action (positioning the paddle). Standard credit assignment struggles: which of the 50 actions between "move paddle" and "brick breaks" actually mattered?

**Why standard approaches fail:** You could try replaying the episode with one action changed, but model-based rollouts drift from reality quickly (small errors compound). Importance sampling is noisy because the ratio of two policies' probabilities explodes over long trajectories.

**The causal fix:** Record one real episode. Reverse-engineer the random numbers (dice rolls) that the environment used. Now replay the episode with one action changed but the same dice rolls. The ball bounces the same way at each step (same randomness), except for the consequences of your changed action. Any difference in outcome is therefore caused by that action alone. This is "counterfactual reasoning" in a literal sense: same world, different choice, observed consequence.

## Example 3: Self-driving sim-to-real (causal transfer)

**The problem:** You have millions of miles of simulated driving data and only thousands of miles of real driving data. You want to use the sim data to train, then deploy on real roads. But the sim's graphics look different from reality, so a policy trained purely in sim fails on real roads.

**Why standard domain adaptation fails:** Standard transfer learning assumes source and target domains are "similar enough" in distribution. But simulation and reality differ in specific, structured ways: physics is nearly identical, visuals are very different. Treating everything as "one big distribution shift" wastes the fact that most of the model transfers perfectly.

**The causal fix:** Draw a diagram of which parts of the world model are the same across sim and reality (physics, steering dynamics) and which differ (visual rendering, weather). Transfer the invariant parts directly. Only retrain the parts that differ using real data. This is much more sample-efficient than retraining everything from scratch.
