I wanted to give an update on the current state. I'll keep it short, so I'll only sketch the RL methods behind these improvements — happy to go deeper when we meet.

Last week I mentioned that the setup which had worked near-perfectly on a single mesh (through memorization) broke completely under multi-mesh training — accuracy stalled below 20%. After a series of experiments that broke the 50% barrier (first Asymmetric Actor-Critic results, described below), the failure mode of RL navigation on unseen meshes has become much clearer.

Successful neurovascular navigation decomposes into two distinct behaviors:

1. **Path-following** — tracking the planned centerline with the right heading and step size. This governs ~90–95% of all steps.
2. **Recovery** — retracting, easing wire slack, and re-approaching at choke points, tight turns, and after wrong-branch entries. A small fraction of steps, but it decides the hard cases.

The core difficulty is that a single policy trained end-to-end cannot retain both. The two behaviors interfere through the *training data distribution*: as path-following improves, the wire gets stuck less often, so recovery states nearly vanish from the replay buffer — and without gradient signal from those states, the recovery skill decays (an interference/forgetting effect specific to off-policy RL, where the policy's own competence starves the data its rarer skill depends on). We can measure this directly: mid-episode recovery events (genuine retract → re-align → advance sequences, tracked at every checkpoint eval) first increase and then fade as path-following sharpens. The reverse direction doesn't occur, since the overwhelming majority of steps demand forward path-following. This asymmetric forgetting is what capped multi-mesh accuracy: there are always episodes where getting stuck is unavoidable, and a policy that has lost recovery cannot finish them.

There are several ways to attack this in the literature:

1. **Larger/sequence models** (our actor–critic networks are currently shallow MLPs): more capacity to hold both behaviors. Cons: more compute, slower iteration.
2. **Model-based RL / world models**: we now have enough logged transitions to learn a dynamics model (state, action → next state) and plan or train against it. Cons: a major departure from our current pipeline.
3. **Privileged-information methods for partially observed control** (the direction I took). The policy only sees fluoroscopy-like observations, but the *simulator* knows the full physical state — contact forces, wire tension/slack — and our *planner* knows the path geometry. These two privileged channels let us explicitly identify the rare "stress" states (contact, buckling, wrong-branch) and treat the two behaviors differently during training, instead of hoping one network balances them on its own. This was also the fastest to implement and test.

**A. Asymmetric Actor-Critic.** The critic is trained on the full privileged state (SOFA contact forces, tension/slack, ground-truth branch) while the actor observes only the deployable, fluoroscopy-like state. The better-informed critic learns more accurate values in exactly the stress states the actor can barely distinguish, and transfers that knowledge through the policy update.
Accuracy ~60%+, best = 63.6%.

**B. Privileged Teacher–Student, built on the planner (currently running).** The stronger version, possible because of our detailed path planning. Two changes over (A):
- The **teacher actor itself** receives the full privileged state (not just the critic), and is trained as a *corrective policy on top of a scripted centerline-following controller* derived from the planner — so the teacher starts at the controller's competence and RL concentrates its learning on the corrections, i.e. precisely the recovery behavior. Planner-derived classifiers additionally identify the stress/choke states so they can be re-weighted in training rather than fading away.
- A **student actor** is then distilled from the teacher in a subsequent run: it observes only deployable inputs, with planner-derived proxies (path geometry, slack estimates) standing in for the privileged channels it cannot see.
Accuracy ~70%+ (first results), best = 81.6% (by region: 100% below the ICA, ~86% mid-ICA, ~50% at the farthest points — the remaining gap is concentrated at the deepest targets).

In short: the combination of the planned path and the simulator's physics state gives us handles to separate — and separately preserve — the two behaviors that one undifferentiated policy cannot hold at once.

I can describe the methods in more detail when we meet. I hadn't started the paper while the RL method was still being finalized; I'm now starting to write it around the planner-based Privileged Teacher–Student method.

Please let me know if you have any questions.
