
TOTAL_ACTOR_LEARNERS = 24

"""
    SINGLE TASK A3C
"""
pong = ["Pong-v0" for _ in range(TOTAL_ACTOR_LEARNERS)]
breakout = ["Breakout-v0" for _ in range(TOTAL_ACTOR_LEARNERS)]

space_invaders = ["SpaceInvadersDeterministic-v4" for _ in range(TOTAL_ACTOR_LEARNERS)]
carnival = ["CarnivalDeterministic-v4" for _ in range(TOTAL_ACTOR_LEARNERS)]
phoenix = ["PhoenixDeterministic-v4" for _ in range(TOTAL_ACTOR_LEARNERS)]
assault = ["AssaultDeterministic-v4" for _ in range(TOTAL_ACTOR_LEARNERS)]
demon = ["DemonAttackDeterministic-v4" for _ in range(TOTAL_ACTOR_LEARNERS)]

"""
    Multi-Task A3C (UGP)
    
    From several games
        DemonAttackDeterministic
        CarnivalDeterministic
        PhoenixDeterministic
        AssaultDeterministic        
        
    UGP will be tested on the target game:

        SpaceInvadersDeterministic

"""

pong_breakout = \
    ["Breakout-v0" for _ in range(8)] + \
    ["PongDeterministic-v0" for _ in range(8)]

bottom_up_shooters = \
    ["Carnival-v0" for _ in range(4)] + \
    ["Phoenix-v0" for _ in range(4)] + \
    ["Assault-v0" for _ in range(4)] + \
    ["DemonAttack-v0" for _ in range(4)]

multi_task = \
    ["SpaceInvadersDeterministic-v4" for _ in range(int(TOTAL_ACTOR_LEARNERS/2))] + \
    ["DemonAttackDeterministic-v4" for _ in range(int(TOTAL_ACTOR_LEARNERS/2))]
