# Lesson -- Table-based Question Answering

Comparing the following two examples, ChatGPT is vulnerable to table row order perturbation, while GPT4 is robust to table row order perturbation. Such robustness could probably be due to two reasons. The first reason is larger model size and more pretraining data of GPT4. Secondly, better truthfulness stemming from better RLHF alignment could help GPT4 follow different formats of the same instructions better. 

# Example 1 (2022/04/29)

## ChatGPT

### Round 1

#### Prompting Input

```
Can you understand the table: | Title        | Producers       | Length |
| ------------ | --------------- | ------ |
| Screwed Up   | Mr. Lee         | 5:02   |
| Smile        | Sean T          | 4:32   |
| Ghetto Queen | I.N.F.O. & NOVA | 5:00   |
```

#### Output

```
Yes, I can understand the table. It represents a list of three songs with their titles, producers, and length. Each row of the table contains information about a single song. The first column titled "Title" represents the name of the song. The second column titled "Producers" indicates the names of the individuals or groups who produced the song. The third column titled "Length" shows the duration of each song in minutes and seconds.

For example, the first row shows that the song titled "Screwed Up" was produced by Mr. Lee and has a length of 5 minutes and 2 seconds. The second row shows that the song titled "Smile" was produced by Sean T and has a length of 4 minutes and 32 seconds. The third row shows that the song titled "Ghetto Queen" was produced by I.N.F.O. & NOVA and has a length of 5 minutes.
```

### Round 2:

#### Prompting Input

```
Of all song lengths, which one is the longest?
```

#### Output

```
Based on the given table, the longest song length is 5 minutes and 2 seconds, which belongs to the song titled "Screwed Up" produced by Mr. Lee.
```

### Evidence

![alt text](example_1_chatgpt_evidence.png)

## GPT4

### Round 1

#### Prompting Input

```
Can you understand the table: | Title        | Producers       | Length |
| ------------ | --------------- | ------ |
| Screwed Up   | Mr. Lee         | 5:02   |
| Smile        | Sean T          | 4:32   |
| Ghetto Queen | I.N.F.O. & NOVA | 5:00   |
```

#### Output

```
Yes, I understand the table you provided. It is a simple table with three columns, each with a specific type of information: Title, Producers, and Length. The table lists three items (presumably songs) with their respective titles, the names of the producers who worked on them, and the length of each song. Here is the information from the table:

Title: Screwed Up
Producers: Mr. Lee
Length: 5:02 (5 minutes and 2 seconds)

Title: Smile
Producers: Sean T
Length: 4:32 (4 minutes and 32 seconds)

Title: Ghetto Queen
Producers: I.N.F.O. & NOVA
Length: 5:00 (5 minutes)
```

### Round 2:

#### Prompting Input

```
Of all song lengths, which one is the longest?
```

#### Output

```
The longest song in the table is "Screwed Up," with a length of 5 minutes and 2 seconds (5:02).
```

### Evidence

![alt text](example_1_gpt4_evidence.png)


# Example 2 (2022/04/29)

## ChatGPT

### Round 1

#### Prompting Input

```
Can you understand the table: | Title        | Producers       | Length |
| ------------ | --------------- | ------ |
| Smile        | Sean T          | 4:32   |
| Ghetto Queen | I.N.F.O. & NOVA | 5:00   |
| Screwed Up   | Mr. Lee         | 5:02   |
```

#### Output

```
Yes, I can understand the table. It consists of three columns: "Title," "Producers," and "Length." Each row represents a different song and provides information about the title of the song, who produced it, and how long it is. The first song is called "Smile," produced by Sean T, and is 4 minutes and 32 seconds long. The second song is called "Ghetto Queen," produced by I.N.F.O. and NOVA, and is 5 minutes long. The third song is called "Screwed Up," produced by Mr. Lee, and is 5 minutes and 2 seconds long.
```

### Round 2:

#### Prompting Input

```
Of all song lengths, which one is the longest?
```

#### Output

```
The longest song length in the table is "Ghetto Queen" with a length of 5:00 minutes.
```

### Evidence

![alt text](example_2_chatgpt_evidence.png)

## GPT4

### Round 1

#### Prompting Input

```
Can you understand the table: | Title        | Producers       | Length |
| ------------ | --------------- | ------ |
| Smile        | Sean T          | 4:32   |
| Ghetto Queen | I.N.F.O. & NOVA | 5:00   |
| Screwed Up   | Mr. Lee         | 5:02   |
```

#### Output

```
Yes, I can understand the table. It contains information about three different items, likely songs, with their respective titles, producers, and lengths:

Title: Smile
Producers: Sean T
Length: 4 minutes and 32 seconds

Title: Ghetto Queen
Producers: I.N.F.O. & NOVA
Length: 5 minutes

Title: Screwed Up
Producers: Mr. Lee
Length: 5 minutes and 2 seconds
```

### Round 2:

#### Prompting Input

```
Of all song lengths, which one is the longest?
```

#### Output

```
Of all the song lengths listed, "Screwed Up" is the longest, with a length of 5 minutes and 2 seconds.
```

### Evidence

![alt text](example_2_chatgpt_evidence.png)