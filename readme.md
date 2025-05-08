Project Log

13/04/2025 - Day 1

    Happy Songkran!!! Sa wat dii pee mai!
    Completed:
        Finished Harvard course
        Got linkedin page up and running, including all 4 course certificates
        Started Ant Sim
        Got basic running working, including making all pygame-related features optional. Got a grass texture and a grid background, all responsive to different screen (square) sizes and grid sizes. Got colonies up and running, starting with 2 hard-coded in. Also sorted ants. They all appear, and I got the turn taking sorted out as well. Got data storage in array of map sorted. Encountered issue with map being updated during iteration, but solved it with mild copilot and chatGPT aid. All entities now take a turn, and can be added and removed. Colony can do nothing, add worker larva or soldier larva. Larvae take a set time to grow then are removed and an ant of their type added on top of the colony. Ants can move or do nothing. Ants hunger degrades over time, including damaging health at low levels, and got ant death sorted. Food also degrades over time, might have to add special logic to stop degrading when on colony space. Food cannot be picked up yet and spawns in randomly. Got basic stuff set up tho!

    Remember:
        MUST ADD TO GITHUB TOMORROW. Before making changes, make initial commit of today.

    Happy Songkran!

14/04/2025 - Day 2

    Italian night! Hell yeah!
    Created github repo and pushed the project. Finished food related work, with ants able to pick up, drop and eat, no decay on colony spots. Ants and the colony die if they don't eat, and creating larvae costs food. Ants and colony can both release pheromones and each have a designated 'range' of pheromones. Added attacking of ants and colony, and generally moved as much as possible to the consts file. Started work on state generation for ants. Found helpful video for AI: https://www.youtube.com/watch?v=L8ypSXwyBds

    Remember:
        Sort states and rest of AI tomorrow!

15/04/2025 - Day 3

    Andaman night! Hell yeah!
    Pushed yesterday to github. Finished all AI implementation, ie finished agent including rewards, states, actions etc, built the models and a trainer using Q-learning algorithm and fixed some other things that I had missed, such as edge boundary complications. Basically, all AI functionality implemented, all now needed is to fine-tune the values for the training process.

    Issues:
        Upon initial test, my fears were confirmed. The colony quickly learns just not to produce ants because they start so stupid. Tomorrow try making a gym for the ants, or simply initially setting the colony to be random and not die to encourage it to not just stop ant production. Save ant models when they are a bit smarter then introduce colony learning as well. Another thought-of issue is that the same relative reward is given for most actions - possibly change reward function to be more accurate, eg rewarding ant hp and other directly-tied things that will actually change each turn for each ant.

    Cya!

16/04/2025 - Day 4

    Initially very stupid. Tried training only ants and maintaining the colony as random first. Will probably follow this approach, and introduce colony AI once the ants are smarter. Made some fixes to the reward function, including adding caste and food held. Bug where I had not added eating to the colony's action list, so fixed that. During training of ants, massively raised colony hp to try to allow ants to discover more. Fiddled with numbers in general to attempt to make progress in training. Ants still very stupid but begin to exhibit some slight indicators of actual behaviour. I'll have to see if anything gets better by tomorrow. Made ant more complicate (20 layers) network as discovered that size is normal for RL.

    Ideas: For visual representation, maybe draw bar chart of the portion of actions that are taken by ants/colonies? ie What percentage of actions are move, what are pheromones (or each pheromone), what percentage are attack ants, what percentage are eat foods etc. Also, introduce some kind of overall, long-term reward for win/loss if possible throughout long memory. Also, change epsilon so ant remains very random by introducing a minimum epsilon temporarily throughout training and raising initial epsilon. Introduce delayed memory/learning so actions in sequence can be learned off of.

17/04/2025 - Day 5

    Training underway. Ant still very stupid. Played around with some values such as introducing a minimum epsilon. Changed the environment to have only one ant per team to encourage it to learn from its own behaviours. Realised how inefficient the setup I had was and thinned down the state a little. Rebuild the state to be global states plus a 3x3 grid with channels for each locational state. Changed the ant model to accept this and put the grid through a CNN before joining it with the fc1'ed global variables into a combined network. In the process of adding an RNN, complications with hidden state and how the order of actions is interfered with by having multiple ants taking actions.

    Overall, biggest challenge in training is trying to speed the convergence and make it more likely the ant will discover successful rewards and policies while also not biasing the experiment at all. I would like not to have to introduce rewards for actions such as picking up food, as that would be against the spirit of the simulation. Research other strategies to help, such as evolutionary simulations with fitness tests or competitive AI. Refer to chatGPT prompts from today lmao.

    TODO:
        Code currently VERY broken, so don't commit to github. Learn how to make github branches. Change code so hidden state order matches order of ants in some id list. Possibly append the ant object to a list every time it takes an action? Then search the hiddenstate arrays by the same index. Test this, then continue attempting to speed convergence. Research nuanced strategies to do this. Possibly fine-tune AI, but I believe the main issue is to do with the AI not experiencing the multi-step high rewards. Also, keep it at one ant per team so as not to falsely attribute value. Very interesting sim, but I have a headache so I'm going to gala night now. Cya!

18/04/2025 - Day 6

    Overview of Strategy:
        Curriculum learning in the following order:
            3x3 grid, one colony, one ant. Teach it to pick up and drop off food, and to not step off the map boundaries.

            4x4 grid, one colony, one ant, reduceed food spawning. Teach the ant to search for food and get it used to having to pathfind to food, as last approach would likely have taught it to wait for food to be in neighbouring squares.

            5x5 grid with 2 colonies next to each other. Teach ants to be aggressive in finding food. Make food sparse to encourage aggression.

            Slowly widen grid and adjust epsilon and food values etc to suit.

            Make all of these as separate gyms that get simulated.

    Worked on issues that implementing RNN had created. After much, MUCH effort, finally succeeded. However, from what I can see, program runs much slower, even with only one ant per colony. Might have to update program in future to make it more efficient. As far as I can see, no significant performance difference, however I believe theoretically that the RNN and CNN have helped, something that hopefully will become apparent when I implement the curriculum learning gyms. Didn't upload to github as I'm not completely confident in the changes despite theoretical success.

    TODO:
        Implement gyms and curriculum learning. I think it's the best shot at getting a semi-trained ant without biasing the simulation. Also, remember the guys I posted on MEE6 and possibly email or get in contact? Primarily, work on gyms and debug anything that comes up due to RNN implementation. Also, though it doesn't matter now, when I start training a colony model as well, I will need to refactor the colony model and class to also have RNN and CNN implementation with it, depending on the reward function. However, colony might be simpler as its tasks are not multi-step and are less complicated, so complicating the model may not be necessary. Best approach is probably to refactor the agent etc to regain functionality, then only implement RNN and CNN if the model is very stupid. Also regularisation (I forget if it's l1 or l2) for the ant to emphasize the important state values eg food value.Anyway, I've got Mediterranean night now so cya!

08/05/2025 - Day 7

    Have worked on this for several days, but only just finishing stage of development. Now got simulation working with RNN and CNN and performance only drops when there are many ants, other than this the sim runs very smoothly. If I ever fully run it, possibly implement ability to not train, as this would likely greatly increase performance. I had many issues with the structure, turns out I was handling things very strangely. Slightly reduced number of layers as well as I found out that the number I was using was likely unneccessary. Glad to have it all working, however need to verify that some parts of code eg reinforcement learning algorithm have been written correctly and do not contain logic errors hindering ants development, as they are still very stupid. However, that is to be expected.

    TODO:
        Next steps will likely be, after verifying an abscence of logic errors, implementing curriculum learning. I will do this before reintroducing the colony as the colony is not initially needed. I will then have to adapt some of my code and uncomment certain parts to reintroduce colony AI, however it should be manageable as most functions work for both within the agent. First curriculum gym will be 3x3 square, only 1 colony, maximum 1 ant with max 1 food at a time. This will teach ant basics of food fetching process before teaching navigation.
