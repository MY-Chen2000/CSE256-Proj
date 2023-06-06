## model

using OpenAI's Chat Completions API

- model: "gpt-3.5-turbo"
- temperature: 0.8
- prompt: "You are a student and only need to answer A, B, C or D without explanation"
- question format
    - [question stem]. (A) [choice] (B) [choice] (C) [choice] (D) [choice]
    - [fact]. [question stem] (A) [choice] (B) [choice] (C) [choice] (D) [choice]

## result

- without fact accuracy: 0.740
- with fact accuracy: 0.868

## analysis

ChatGPT model itself has strong common knowledge in its memory, leading to a good result to answer these questions. The gap between "open book" (with facts) and "closed book" results show that more information are still useful to Large Language Models. Also, the result of this model is worse than the SOTA (PaLM 540B) by around 6%. This is foreseeable since ChatGPT 3.5 has fewer parameters (154B).

Below is some false example of experiments:

- without fact

    There is most likely going to be fog around
    (A) a marsh
    (B) a tundra
    (C) the plains 
    (D)a desert

    predction: C

    Correct Answer: A
    
- with fact
    
    `the sun is located directly overhead at noon.` The middle of the day usually involves the bright star nearest to the earth to be straight overhead why?
    (A) moons gravity
    (B) human planet rotation
    (C) global warming
    (D) moon rotation

    predction: D

    Correct Answer: B
