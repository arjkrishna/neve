 I wanted to give an update on the current state but also try to keep the email short so won’t be able described the “new RL methods” that I needed to implement for these improvements. 

Last week I briefly mentioned that our setup that had started working perfectly in one mesh because of  memorization broke completely even when training with different meshes. Accuracy stalled below 20%.

After running several experiments and breaking past the 50% (last week; first results of Asymmetric Actor Critic briefly described below) accuracy barrier, failure of RL navigation on unseen meshes have become more clear

There are two key behaviors for successful neurovascular navigation that become crucial (when there is no memorization) :

1.	Ability to follow planned path centerlines accurately :  This behavior is needed for most steps (about 90% – 95% of total steps)
2.	Ability to recover : Needed at choke points, when getting stuck during tight turns and when accidently getting into the wrong branch.(makes a small fraction of steps)

The problem with learning the above two behaviors simultaneously (in one RL algo. setup) is that learning of each behavior facilitates the non-learning (forgetting of other). If the centerlines are followed with the correct heading and step displacement the number of incidents when the wire is stuck reduces hence reducing the learning of recovery behavior and vice versa (the opposite case doesn’t happen since the majority of steps demand forward steps following the centerlines).

Hence, during learning; recovery behavior is learned to a certain extent (when the wire genuinely retracts a few steps, correct the heading and move forward; I have metrics for such instances during checkpoints evals) but with more learning the skill is lost as the wire get better in following centerlines . This created a ceiling for the accuracy on different meshes since there are always some cases when the wire will be stuck and the ability to recover would be necessary.

There are several ways (in literature) to tackle this :

1.	Bigger and more complex deep learning models: At the moment our actor/critic networks are shallow and FCNs . Cons: More time consuming / need more compute
2.	Moving on to the model based learning: We have enough data now to build a model of state, action -> state chains and RL acts as a processing step (akin to training large language models based chat-boxes or code editors). Cons: Major change to our current approach
3.	Trying more complex RL harnesses designed to tackle POMDPs. Pros:
a.	Faster to implement and test 
b.	Gives a way to different the above two modes of learning (for the 2 behaviors described above) via inert SOFA and planned path  variables since contact forces gives us an ability to identify “stress” states with regular states.

A.	Assymetric Actor Critic : SOFA have contact forces and tension values that are not exposed to policy but can be exposed to the critic learning better values for the state and hence transferring better state knowledge to the actor which only observes the floroscpy like values as the state 
Accuracy ~ 60%+ 
Best = 63.6%


         Currently Running (Possible because of detailed path planning in our setup )
B.	Student Teacher Actor Critic with Planned Path classifiers: Stronger version of the above where teacher actor sees the full priveldged  information as the critic and then teaches another student actorr in the subsequent run. Student critic also gets planned path proxies for the priveldged information to assist it’s learning from the teacher
Accuracy ~ 70%+ (First results)
Best = 81.6% (Areas of interest : Mid-ICA ~ 86% and 100% below; Furthest points on the vessel ~ 50%)

	
In short,  the combination of planned path and SOFA simulation physics state gives us an ability to separate the learning of 2 key behaviors

I can describe them in more detail when we meet .  I didn’t begin writing the paper since I was still finalizing the RL methods. I will start writing the planned path version (Student Teacher Actor Critic RL method ) now.
Please let me know in case of questions.
