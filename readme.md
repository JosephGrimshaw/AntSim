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
