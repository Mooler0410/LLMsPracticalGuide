# Lesson -- Social Game: One night ultimate werewolf (Chinese/中文)

In this lesson, we ask the model to play a social game called "One night ultimate werewolf". The game is played by a group of people. Each person is assigned a role. The game requires players to play strategically and to lie to each other. The model is expected to play the game with other players.

It requries the following skills:
- Understanding the rules and game flow of the game.
- Understanding the roles of the game.
- Knowing how to play strategically.
- Knowing how to lie when necessary (e.g., when you are a werewolf).
- Knowing how to tell the truth when necessary (e.g., when you are a villager).
- Knowing how to argue with other players.

Comparing the the ChatGPT and GPT4 examples, we can see that 
- ChatGPT can not understand the game rules and flow, while GPT4 can. ChatGPT follows the werewolf game rule not the one night ultimate werewolf game rule.
- ChatGPT can not understand the roles of the game, while GPT4 can. ChatGPT hullucinates a non-existing player called "Tom".
- Both ChatGPT and GPT4 know how to play strategically, i.e., they know how to lie when necessary and how to tell the truth when necessary. However, ChatGPT's strategy is much more naive than GPT4's. & ChatGPT's argument is less convincing than GPT4's.
- ChatGPT is also constrained by the limited context length.
- However, it's hard to differentiate different players just in the style they talk. All bots chat in similar tones. It's hard to tell who is who, without the name as the uniqure identifier. It's will be gap to fill in the future.


But the finding is not conclusive, since the game is implemented in Chinese.

Codebase: https://github.com/HMJiangGatech/GPT4-werewolf

# Examples 

## ChatGPT Example 1

```
玩家 Eve 加入游戏.
你是第 4 位玩家.
你的身份是 酒鬼.
场上有 5 位玩家.
玩家 0: Alice
玩家 1: Bob
玩家 2: Charlie
玩家 3: David
玩家 4: Eve
轮到你行动了.
你是酒鬼，你可以选择交换自己的身份牌和中央区域的一张牌.
请输入你要交换的牌的编号（0, 1, 2）: 0 
你交换了自己的身份牌和中央区域的第 1 张牌.
玩家 上帝 说: 天亮了，所有人睁眼！请大家开始讨论。讨论时间为3轮。
玩家 Alice 说: 天亮了，我们有三分钟的时间来讨论一下。我昨晚没有收到什么有价值的信息。但是Charlie是爪牙，这对我们接下来的推理可能有重要影响。所以我希望领先的几位可以分享一下你们的信息和想法。
玩家 Bob 说: 好的，我昨晚是强盗，但我没有进行交换身份牌的操作。我的想法是，我们现在应该把重心放在尽快找到狼人身上。我个人觉得，Tom昨天的发言比较可疑，他好像一直在试图把话题引开，并保持低调。我建议我们重点关注一下Tom。
玩家 Charlie 说: 我昨晚是没有什么特殊的操作。Bob你觉得Tom可疑是因为他试图引开话题？我有同感，除此之外，我觉得David的表现也很可疑，他昨晚没有发言，而且看起来有点紧张。我建议我们同时关注Tom和David。
玩家 David 说: 我昨晚变成了失眠者，我在讨论中跟着大家的思路走，我觉得目前的可疑人物是Tom和Alice。Tom昨晚没说什么，而且他的表现也让我感到可疑。而Alice则一直在发言，但是她的发言似乎是在转移话题。
玩家 上帝 说: 请Eve发言。记住你晚上的信息是"酒鬼请睁眼，你交换了自己的身份牌和中央区域的第 1 张牌 （左）. "。根据别人的信息，如果你觉得你是狼人，不要轻易说自己是狼人，请想办法归票到别人身上。如果你觉得你不是狼人，请想办法归票到狼人身上。
```

Totally wrong understanding of the rules of game. But it's fun to see how AI plays the game. 
ChatGPT can not complete the game, due to the limited context length.

## ChatGPT Example 2


## GPT-4 Example 1

```
玩家 Eve 加入游戏.
你是第 4 位玩家.
你的身份是 预言家.
场上有 5 位玩家.
玩家 0: Alice
玩家 1: Bob
玩家 2: Charlie
玩家 3: David
玩家 4: Eve
轮到你行动了.
你是预言家，你可以查看一位玩家或者两张中央区域的牌.
请输入你的选择（0:玩家/1:中央区域）: 1
请输入你要查看的第一张牌的编号: 0
你查看了中央区域的第 1 张牌 （左）. 
第 1 张牌是 强盗.
请输入你要查看的第二张牌的编号: 1
你查看了中央区域的第 2 张牌 （中）. 
第 2 张牌是 酒鬼.
玩家 上帝 说: 天亮了，所有人睁眼！请大家开始讨论。讨论时间为3轮。
玩家 Alice 说: 大家好，我在昨晚觉得有些奇怪，我觉得我是预言家，我查看了Bob的身份，他似乎是一个捣蛋鬼。那么我们可以从其他信息中寻找狼人的线索，请大家分享一下各自的信息。
玩家 Bob 说: Alice，既然你是预言家，我就告诉你一个情况。我一开始的确是捣蛋鬼，但后来我发现自己可能身份发生了改变。因为昨晚我交换了Carol和David的身份牌，而且我觉得David有可能是狼人。大家可以先考虑下这个线索。
玩家 Charlie 说: 各位，我想告诉大家一个情况，我原本是失眠者，但昨晚我重新察看了自己的身份牌发现自己变成了酒鬼。根据Bob的说法，他交换了Carol和David的身份牌，而David可能是狼人。我觉得这个线索很有可能是真的。大家可以考虑一下这个信息，找出狼人。
玩家 David 说: 各位，我觉得这里有些许矛盾。首先，我昨晚是狼人，我的同伴是Bob。但是根据Alice的说法，她是预言家，她查看了Bob的身份牌，说他是捣蛋鬼。如果Bob真的是捣蛋鬼，那么他应该不会是我的同伴狼人。而且，根据Charlie的说法，他原本是失眠者，但现在变成了酒鬼。这种情况下，我很有可能已经不再是狼人了。所以我建议我们应该更加仔细地分析各自的身份，找出潜在的狼人。
玩家 上帝 说: 请Eve发言。记住你晚上的信息是"预言家请睁眼，你查看了中央区域的第 1 张牌是 强盗, 第 2 张牌是 酒鬼."。根据别人的信息，如果你觉得你是狼人，不要轻易说自己是狼人，请想办法归票到别人身上。如果你觉得你不是狼人，请想办法归票到狼人身上。
请输入你的发言: 我才是真的预言家，昨天晚上我看见了中央区域的左边的牌是强盗， 中间的牌是酒鬼。所以Alice在说谎话。Charlie也在说谎，应为酒鬼在下面不可能被换上来最后换给失眠者。所以Charlie不是失眠者。我倾向于相信David和Bob。所以Charlie很有可能就是狼人
玩家 Alice 说: Eve, 我明白你的怀疑，但是我真的看了Bob的身份证，并发现他是捣蛋鬼。我没必要撒谎。而关于Charlie的话，我觉得有可能他变成了酒鬼，但也有可能他变成了其他角色。所以我们不能排除其他可能性。虽然David说他原本是狼人，但现在可能已经不是，所以我们这边应该继续关注其他玩家的身份。请大家继续分享信息，我们要确保找到狼人。
玩家 Bob 说: Alice和Eve，我明白你们的观点。实际上，我昨晚确实是捣蛋鬼，但现在我也不能确定自己是不是狼人，因为我的身份可能已经改变。关于David的说法，如果他现在已经不是狼人，那么我们应该去寻找谁替换了他的身份。我觉得我们应该把注意力集中在Charlie身上，因为根据Eve的说法，他可能跟酒鬼有关，而我觉得酒鬼也是一个潜在的狼人身份。所以我们应该尽快找出狼人，确保我们的胜利。
玩家 Charlie 说: 各位，实际上我现在也不能确定自己是否是狼人，因为我身份发生了变化。根据Eve的说法，我可能跟酒鬼有些关联，但我不能肯定是不是变成了酒鬼。从Bob的说法来看，他有可能是狼人，因为他当时是捣蛋鬼，但现在不能确定他的身份了。

在这种情况下，我觉得我们应该更多地关注那些没有提到自己身份的玩家。我们不能排除可能有人在隐瞒自己的真实身份。所以我建议大家多分享一些信息，试图找出狼人。同时，我们也应该关注David的身份，因为他说自己曾经是狼人，所以有可能现在狼人就在他这边。大家继续讨论，我们要确保找到狼人。
玩家 David 说: 各位，我理解大家对我身份的担忧，但现在我也不能肯定自己是否还是狼人。从目前的情况来看，Bob和我都有可能是狼人，但我们也不能排除其他玩家的可能性。我觉得我们应该更关注那些没有透露自己身份的玩家，因为有可能他们是在隐瞒自己的真实身份。此外，我认为Eve可能真的是预言家，如果她说的是真的，那么中央区域的强盗和酒鬼牌就已经被确认了。这对我们分析身份也有帮助。

请大家继续讨论，尽可能地分享更多信息。我们要找出狼人，确保我们的胜利。
玩家 上帝 说: 请Eve发言。记住你晚上的信息是"预言家请睁眼，你查看了中央区域的第 1 张牌是 强盗, 第 2 张牌是 酒鬼."。根据别人的信息，如果你觉得你是狼人，不要轻易说自己是狼人，请想办法归票到别人身上。如果你觉得你不是狼人，请想办法归票到狼人身上。
请输入你的发言: 至少没有人质疑Bob的说法和David的说法，我还是认为Charlie是狼人。除非后面有人改变了说法。如果你没有讲出事实，请务必在下一轮也就是最后一轮说清事实
玩家 Alice 说: Eve，我理解你的观点，但我们不能仅仅凭借目前的信息就认定Charlie一定是狼人。我们还需要更多确凿的证据。同时，根据David和Bob的说法，我们还不能确定他们现在是否是狼人，因此我们不能排除其他可能性。

请大家最后一轮继续分享信息，我们要确保找出狼人。不论如何，我们要在这轮结束之前找到最可疑的对象进行投票。大家加油！
玩家 Bob 说: 各位，我们已经进入最后一轮讨论了。从目前的信息来看，虽然David、Charlie和我都有可能是狼人，但我们仍然需要更多的证据来确认。我觉得我们不能轻易下结论，否则可能会错过真实的狼人。

我们要充分地利用最后一轮讨论，继续分享信息，寻找潜在的狼人。请大家慎重思考，提出自己的看法。我们要确保找到真正的狼人，切勿忽略任何线索。大家加油！
玩家 Charlie 说: 各位，这已经是我们最后一轮讨论了。我知道有些玩家怀疑我是狼人，但实际上我现在也不能确定自己的身份。从目前的信息来看，David、Bob和我都有可能成为狼人，我们需要更多的证据来找出真正的狼人。

在最后一轮讨论中，请大家认真思考，分享自己的观点和信息。我们不能因为怀疑而轻易地定性某人是狼人。为了村庄的胜利，请大家继续努力，争取找出潜在的狼人。希望大家能够慎重投票，确保我们可以找到真正的狼人！大家加油！
玩家 David 说: 大家好，这是我们的最后一轮讨论。从目前的信息来看，我、Bob和Charlie都有可能是狼人。尽管我开始是狼人，但现在我可能已经不是了。我觉得我们应该在最后一轮讨论中充分分享信息，从而确定谁才是狼人。如果你们相信Eve是真正的预言家，那么我们可以根据她所说的中央区域的身份牌来推断其他玩家的身份。

请大家在最后一轮讨论中认真思考，提出自己的看法，我们要确保找出真正的狼人。我们不能因为怀疑而下定论，否则可能会误判。希望大家能够慎重投票，确保我们可以找到真正的狼人，为村庄的胜利努力！大家加油！
玩家 上帝 说: 请Eve发言。记住你晚上的信息是"预言家请睁眼，你查看了中央区域的第 1 张牌是 强盗, 第 2 张牌是 酒鬼."。根据别人的信息，如果你觉得你是狼人，不要轻易说自己是狼人，请想办法归票到别人身上。如果你觉得你不是狼人，请想办法归票到狼人身上。
请输入你的发言: 我还是会投Charlie，既然没有人归票，我就来归这个票，希望好人都跟我投Charlie。如果票太分散狼人可能会因此逃脱
玩家 上帝 说: Alice 投票结果：Charlie
玩家 上帝 说: Bob 投票结果：Charlie
玩家 上帝 说: Charlie 投票结果：David
玩家 上帝 说: David 投票结果：Charlie
玩家 上帝 说: 请投票。请输出你要投票的玩家名字（Alice,Bob,Charlie,David,Eve）或者输出"弃票"。
请输入你的投票目标: Charlie
玩家 上帝 说: Eve 投票结果：Charlie
玩家 上帝 说: 投票结果为{'Alice': 0, 'Bob': 0, 'Charlie': 4, 'David': 1, 'Eve': 0}
玩家 上帝 说: Charlie被投票出局。他的身份是失眠者。
玩家 上帝 说: 狼人阵营胜利。
```