// Ask Neuro, part B: XOR, passing blame back, memorising, words as numbers, the next word, attention.
export default {
  xor: [
    { say: "One neuron draws one straight line: “on” on one side, “off” on the other. Is that enough for everything? Let's test it." },
    { say: "Here are four dots. Blue dots are where the switches are <b>the same</b> (both off, or both on). Orange dots are where they are <b>different</b>. This puzzle is called <b>XOR</b>. Turn and move the line to put blue on one side and orange on the other.", viz: "xor", try: true, done: "Two lines did it! Did you notice? With one line, you could never get all four right. The best was 3 out of 4." },
    { ask: "Why can't one straight line split them?", options: ["The blue dots are in opposite corners, and so are the orange ones", "The line is too short", "There are too many dots"], answer: 0, yes: "Yes. Opposite corners can't be cut apart with one straight cut. Try it on paper!" },
    { say: "In 1969, a famous book pointed this out. It helped convince people that neurons were a dead end, and money for this research dried up for years. This cold time is called an <b>AI winter</b>." },
    { ask: "How did we get two lines?", options: ["Use two neurons side by side, then a third neuron to combine them", "Make one neuron bigger", "Use a curved screen"], answer: 0, yes: "Yes! Neurons feeding into neurons. Neurons in the middle are called a <b>hidden layer</b>." },
    { ask: "So what does stacking neurons in layers give us?", options: ["Bends and shapes, not just straight lines", "Nothing new", "Faster straight lines"], answer: 0, yes: "Right. Each layer bends the space a bit more. With enough neurons, a network can draw almost any shape. That's a <b>neural network</b>." },
    { recap: ["One neuron can only draw a straight line.", "Some puzzles, like <b>XOR</b>, need more than one line.", "Neurons feeding into neurons form <b>layers</b>.", "Layers can draw curved, complex shapes."], words: [["XOR", "“on when the two inputs are different”"], ["hidden layer", "neurons between the input and the output"], ["neural network", "many neurons connected in layers"]] },
  ],

  blame: [
    { say: "We have layers now. But here's a hard problem. The network gave a wrong answer. Which of its thousands of weights should change?" },
    { ask: "A team of three cooks makes a bad soup. How should they fix it next time?", options: ["Everyone changes everything at random", "Each cook changes their part by how much it made the soup bad", "Fire the last cook"], answer: 1, yes: "Yes. Blame each part by how much it pushed the result. Fix big culprits a lot, small ones a little." },
    { say: "A network does the same thing, but backwards. Start at the answer, then pass the blame back, layer by layer.", viz: "blame", try: true, done: "That's <b>backpropagation</b>, or “backprop”: passing blame backwards so every weight knows its share." },
    { ask: "After backprop, what does each weight do?", options: ["Takes a small step to lower its blame: gradient descent again", "Stays the same", "Doubles"], answer: 0, yes: "Exactly. Backprop finds the slope for every weight. Gradient descent takes the step." },
    { ask: "A network has 1 billion weights. Does backprop need to try each one separately?", options: ["Yes, a billion separate tries", "No: one backward pass gives the blame for all of them at once"], answer: 1, yes: "Yes! That's why it works at all. One forward pass to guess, one backward pass to blame, then a step. Repeat millions of times." },
    { say: "Every modern model, from your phone's camera to the biggest chatbot, is trained with this loop: <b>guess, measure the loss, pass the blame back, step</b>." },
    { recap: ["When the answer is wrong, each weight gets blame by how much it pushed the answer.", "<b>Backprop</b> passes blame from the answer backwards through the layers.", "Then gradient descent nudges every weight a little.", "The training loop: guess → loss → backprop → step."], words: [["backpropagation", "working out every weight's share of the blame, from the answer backwards"], ["training loop", "guess, measure, blame, step, repeat"]] },
  ],

  memorise: [
    { ask: "A student learns all the answers of last year's test by heart. Then the real test has new questions. What happens?", options: ["They ace it", "They fail: they memorised, but didn't understand"], answer: 1, yes: "Right. Doing well on practice doesn't mean you learned." },
    { say: "Models do this too. Here a curve learns from 9 dots. Make it bendier and watch the practice mistakes. Then show some <b>new</b> dots it has never seen.", viz: "wiggle", try: true, done: "Look at the bars. More bending made practice mistakes tiny, but mistakes on the new dots grew. The curve chased the noise." },
    { say: "This is called <b>overfitting</b>: learning the examples, including their random noise, instead of the real pattern." },
    { ask: "So how do we check a model honestly?", options: ["Test it on examples it never saw while training", "Test it on its training examples", "Ask the model if it's good"], answer: 0, yes: "Yes. We keep some examples aside, locked away. This is the <b>test set</b>." },
    { ask: "You keep changing your model until the test score is great. Is the test still honest?", options: ["Yes, it's still unseen", "No: you've slowly tuned the model to the test"], answer: 1, yes: "Exactly. Use a separate <b>validation set</b> for tuning, and open the test set only once at the end." },
    { ask: "What usually helps most against overfitting?", options: ["More training examples", "A bendier model", "Training longer"], answer: 0, yes: "More data, by far. Noise averages out when there's more of it." },
    { recap: ["Doing well on training data isn't enough.", "<b>Overfitting</b> means memorising noise instead of the pattern.", "Judge a model on data it never saw: the <b>test set</b>.", "More data is the best cure."], words: [["overfitting", "memorising the examples instead of learning the pattern"], ["test set", "examples kept aside to check the model honestly"], ["validation set", "examples used to tune choices, so the test stays clean"]] },
  ],

  words: [
    { say: "Now the exciting part: language. But a computer needs numbers. So how do we turn words into numbers?" },
    { ask: "First, we cut text into pieces. Why not just give every word in English its own number?", options: ["There are too many words, and new ones appear every day", "Words are too short", "Computers hate words"], answer: 0, yes: "Right. Names, typos, new slang… So we cut text into common chunks instead." },
    { say: "These chunks are called <b>tokens</b>. Common words are one token. Rare words get cut into smaller pieces. Type anything below and watch.", viz: "tokens", try: true, done: "Every token gets an ID number. That's all the model ever sees: a list of numbers." },
    { ask: "But an ID is just a label. Is token 412 “more” than token 17?", options: ["Yes, it's bigger", "No, it's just a name tag, like a jersey number"], answer: 1, yes: "Right. So we need a better way to give a word meaning." },
    { say: "The trick: give every token a <b>place on a map</b>. Words with similar meanings sit close together. Tap two words to see their distance.", viz: "wordmap", try: true, done: "Close words mean similar things. And even directions mean something: man → woman is the same arrow as king → queen." },
    { ask: "Nobody drew this map by hand. Where does it come from?", options: ["The model learns it during training, from how words are used", "A dictionary", "Random"], answer: 0, yes: "Yes. Words used in similar sentences end up close. These map positions are called <b>embeddings</b>." },
    { recap: ["Text is cut into <b>tokens</b>: common chunks of letters.", "Each token gets an ID number.", "Each ID gets a learned place on a map: an <b>embedding</b>.", "Similar words sit close; directions carry meaning."], words: [["token", "a chunk of text the model reads as one piece"], ["embedding", "a token's learned place on the meaning map"]] },
  ],

  nextword: [
    { say: "Here's the big secret of chatbots. They are trained on one simple game, over and over: <b>guess the next word</b>." },
    { ask: "“The cat sat on the ___”. What comes next?", options: ["mat", "moon", "seventeen"], answer: 0, yes: "Easy for you. To guess well, you used grammar, meaning, and what cats do. A model has to learn all of that too." },
    { say: "A model doesn't pick just one word. It gives every possible word a chance. Then we pick one. Press <b>Pick a word</b> a few times. Then move the temperature.", viz: "nextword", try: true, done: "Low temperature: it almost always picks the top word. High temperature: it takes more risks." },
    { ask: "You want a creative poem. What temperature?", options: ["Higher: more surprising words", "Lower: always the safest word"], answer: 0, yes: "Yes. For maths or code, lower is better. For stories, a bit higher." },
    { ask: "To write a whole answer, what does the model do?", options: ["Guess one token, add it, then guess the next, again and again", "Write the whole answer in one go", "Look it up on the internet"], answer: 0, yes: "Exactly. One token at a time. Each new word becomes part of the input for the next one." },
    { guess: "Big models were trained on roughly how many tokens? (In trillions.)", range: [0, 40, 1, 5], answer: 15, tol: 10, unit: "trillion", yes: "Modern models read around 10 to 30 trillion tokens. A person who reads an hour every day reads a few hundred million words in a whole life." },
    { ask: "Why does such a simple game make models so smart?", options: ["To guess the next word in any text, you must understand a lot about the world", "It doesn't; it's a trick", "Because of the temperature"], answer: 0, yes: "Yes. Guessing the end of a mystery novel means knowing who did it. Good guessing needs understanding." },
    { recap: ["Chatbots are trained to <b>guess the next token</b>.", "They give every word a chance, then one is picked.", "<b>Temperature</b> controls how risky the picks are.", "Answers are written one token at a time."], words: [["next-token prediction", "guessing what comes next, the game LLMs train on"], ["temperature", "a dial: low = safe picks, high = surprising picks"]] },
  ],

  attention: [
    { ask: "“The animal didn't cross the street because <b>it</b> was too tired.” What is “it”?", options: ["the animal", "the street"], answer: 0, yes: "Easy for you! Streets don't get tired. But how would a model know?" },
    { say: "Each word needs to <b>look back</b> at the other words and decide which ones matter. Tap “it” and see.", viz: "attention", try: true, done: "“It” looks mostly at “animal”. This looking around is called <b>attention</b>." },
    { ask: "How does a word decide where to look?", options: ["It asks a question, and every word shows a label; good matches get more attention", "It always looks at the word before", "It looks at random"], answer: 0, yes: "Yes. Each word makes a <b>query</b> (what am I looking for?). Each word shows a <b>key</b> (what do I have?). Good matches win." },
    { ask: "When “it” finds “animal”, what does it take from it?", options: ["Some of animal's meaning, called its value", "Nothing, it just points", "Its letters"], answer: 0, yes: "Right. Matching words share their <b>value</b>. So “it” now carries the meaning “animal”." },
    { ask: "A model has many attention <b>heads</b>. Why more than one?", options: ["Each head can look for something different: who did it, when, which word came just before", "For backup", "To be faster"], answer: 0, yes: "Yes. One head tracks grammar, another tracks names, another tracks the previous word, and so on." },
    { recap: ["Words need to look at other words to understand.", "<b>Attention</b> lets every word pick which words matter.", "Query = what I'm looking for. Key = what I have. Value = what I share.", "Many <b>heads</b> look for different things at once."], words: [["attention", "each word looking back at the words that matter to it"], ["query, key, value", "the question, the label, and the meaning that attention uses"], ["head", "one attention pattern; models have many"]] },
  ],
};
