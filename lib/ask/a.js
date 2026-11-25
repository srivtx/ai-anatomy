// Ask Neuro, part A: what learning is, numbers, a line, being wrong, going downhill, one neuron.
// House rules for this file: short sentences, everyday words, and explain every new word the first time.
export default {
  learn: [
    { say: "Hi! I'm Neuro. Let's start with a big question, but an easy one." },
    { ask: "You want a computer to sort email into <b>spam</b> and <b>not spam</b>. What is the old way to do it?", options: ["Write rules by hand, like “if it says FREE MONEY, it's spam”", "Ask the email nicely", "Delete all email"], answer: 0, yes: "Right. People wrote rules for years.", no: "Think of how a person would tell a computer exactly what to do." },
    { ask: "Spammers change their words every day: “FR3E M0NEY”, “fr.ee mon.ey”. What happens to your rules?", options: ["They keep working forever", "They break, and you write new ones again and again", "The computer fixes them by itself"], answer: 1, yes: "Yes. Rules break when the world changes. That's the problem." },
    { say: "Here's the new idea. Don't write rules. <b>Show examples.</b> Lots of emails, each with the right answer next to it. Try it: label these six." , viz: "sortMail", try: true, done: "That's it. A pile of examples with answers is called <b>training data</b>." },
    { say: "Now the computer looks at many examples and finds the pattern by itself. Finding the pattern from examples is what we call <b>learning</b>. The thing that learns is called a <b>model</b>." },
    { ask: "So what does a model need most to learn well?", options: ["A very fast screen", "Lots of good examples with the right answers", "Very long rules"], answer: 1, yes: "Yes. Good examples are the fuel. This stays true all the way up to the biggest chatbots." },
    { ask: "One more. Which of these is a <b>learning</b> problem?", options: ["Add two numbers", "Spot a cat in a photo", "Sort names A to Z"], answer: 1, yes: "Right. Nobody can write the rules for “cat”, but we have millions of cat photos. That's where learning wins.", no: "Which one is easy to show with examples, but hard to write rules for?" },
    { recap: ["Rules break when the world changes.", "Learning means finding a pattern from <b>examples</b>.", "Examples with answers are called <b>training data</b>.", "The thing that learns is a <b>model</b>."], words: [["training data", "examples with the right answers"], ["model", "the thing that learns from examples"]] },
  ],

  numbers: [
    { say: "Computers only understand one thing: <b>numbers</b>. So before we can learn anything, we must turn the world into numbers." },
    { ask: "How could a computer store a black-and-white picture?", options: ["As one big number", "As a grid of small squares, each with a number for how dark it is", "It can't"], answer: 1, yes: "Yes! Each small square is a <b>pixel</b>. Its number says how dark it is." },
    { say: "Here is a tiny picture, 8 by 8 pixels. Tap squares to paint them, and press the button to see the numbers underneath.", viz: "pixels", try: true, done: "See? The picture and the numbers are the same thing. 0 means white, 9 means black." },
    { guess: "A phone photo is about 4,000 × 3,000 pixels. Each pixel has 3 numbers (red, green, blue). Roughly how many numbers is one photo, in millions?", range: [1, 100, 1, 10], answer: 36, tol: 6, unit: "million", yes: "4,000 × 3,000 × 3 = 36 million numbers. One photo!" },
    { ask: "Sound is also numbers. What do you think the numbers are?", options: ["How loud the air is pushing, measured many times a second", "The name of the song", "The colour of the sound"], answer: 0, yes: "Yes. A microphone measures the push of the air about 44,000 times a second." },
    { say: "A list of numbers that describes one thing has a name: a <b>vector</b>. A picture is a vector. A sound is a vector. Soon, a word will be a vector too." },
    { ask: "Two pictures of the same cat. Will their lists of numbers be exactly the same?", options: ["Yes, always", "No, but they should be similar"], answer: 1, yes: "Right. Light and angle change every number a little. A good model must see past those small changes." },
    { recap: ["Computers only work with numbers.", "A picture is a grid of <b>pixels</b>; each pixel is a number.", "A list of numbers that describes something is a <b>vector</b>."], words: [["pixel", "one small square of a picture, stored as a number"], ["vector", "a list of numbers that describes one thing"]] },
  ],

  line: [
    { say: "Let's build our first real model. Here's the job: guess the <b>price of a house</b> from its <b>size</b>." },
    { ask: "Bigger houses usually cost…", options: ["more", "less", "the same"], answer: 0, yes: "Right. So size and price go up together." },
    { say: "When two things go up together, a straight line is a good first guess. A line has just two numbers: how <b>steep</b> it is, and where it <b>starts</b>. Move them until the line runs through the dots.", viz: { type: "line", goal: 160, errors: true }, try: true, done: "Nice fit! Your model is just two numbers. Those numbers are called <b>parameters</b>." },
    { ask: "A new house is 100 m². How does your model guess its price?", options: ["Find 100 on the line and read the price", "Pick the price of the closest house", "Take the average of all prices"], answer: 0, yes: "Yes. That's using the model: put in a size, read out a price. This is called a <b>prediction</b>." },
    { ask: "You chose the two numbers by hand. What would be better?", options: ["Let the computer find the best two numbers by itself", "Guess randomly", "Use more pencils"], answer: 0, yes: "That's the whole game of machine learning. Next we'll teach the computer to find them." },
    { ask: "Our model only looks at size. What else could help guess the price?", options: ["Number of rooms", "How far from the city centre", "Both"], answer: 2, yes: "Both! Each extra fact is called a <b>feature</b>. Real models use many features at once." },
    { recap: ["A model can be as simple as a line.", "Its numbers are called <b>parameters</b>.", "Using the model to guess is a <b>prediction</b>.", "Each fact we feed in is a <b>feature</b>."], words: [["parameter", "a number inside the model that can be changed"], ["prediction", "the model's guess"], ["feature", "one fact about an example, like size or rooms"]] },
  ],

  wrong: [
    { say: "To let the computer fix the line, it first needs to know <b>how wrong</b> the line is. We need one number for “how bad”." },
    { say: "Look at the red lines. Each one is a <b>miss</b>: the gap between the real price and the line. Make the total miss as small as you can.", viz: { type: "line", errors: true, goal: 120 }, try: true, done: "The smaller the total miss, the better the line." },
    { ask: "One miss is +20 and another is −20. If we just add them, we get 0. Is that a problem?", options: ["No, 0 is perfect", "Yes: the misses cancel out and hide the mistakes"], answer: 1, yes: "Right. So we make every miss positive before adding." },
    { ask: "Which trick makes every miss positive?", options: ["Square it: 20 × 20 = 400 and −20 × −20 = 400", "Divide by 2", "Ignore the minus sign only on Tuesdays"], answer: 0, yes: "Squaring works, and it also punishes big misses much more than small ones." },
    { guess: "A miss of 2 squared is 4. A miss of 10 squared is…?", range: [0, 200, 5, 50], answer: 100, tol: 0, yes: "10 × 10 = 100. Five times the miss, twenty-five times the punishment." },
    { say: "Now with squared misses. Notice how one far-away dot makes the total jump.", viz: { type: "line", errors: true, squares: true, goal: 25 }, try: true, done: "This total of squared misses has a name: the <b>loss</b>. Learning means making the loss small." },
    { ask: "Loss goes from 900 to 200 after a change. Was the change good?", options: ["Yes, the loss went down", "No, the loss went down"], answer: 0, yes: "Yes. Lower loss means a better model, at least on these examples." },
    { recap: ["A <b>miss</b> is the gap between the guess and the truth.", "We square misses so they can't cancel out.", "The total is the <b>loss</b>: one number for how bad the model is.", "Learning = making the loss smaller."], words: [["loss", "one number that says how wrong the model is"]] },
  ],

  downhill: [
    { say: "We have a loss. Now: how does the computer make it smaller, all by itself?" },
    { say: "Picture the loss as a valley. Where you stand is your current line. Height is how bad it is. The lowest point is the best line." },
    { ask: "You're on a hill in thick fog. You can't see the bottom. How do you get down?", options: ["Feel which way the ground slopes, and step that way", "Jump as high as you can", "Stand still and wait"], answer: 0, yes: "Exactly. Feel the slope, take a small step downhill, repeat. That's the whole method." },
    { say: "Try it. Take steps and watch the ball. The arrow shows which way is downhill.", viz: "valley", try: true, done: "You reached the bottom! This is called <b>gradient descent</b>. Gradient means slope. Descent means going down." },
    { ask: "Now set the step size to its biggest value and try again. What happens?", options: ["It gets there faster, always", "It jumps over the bottom, and can even fly away", "Nothing changes"], answer: 1, yes: "Yes. Too big a step overshoots. The step size is called the <b>learning rate</b>, and picking it well matters a lot." },
    { ask: "And if the steps are tiny?", options: ["It's safe but very slow", "It fails right away"], answer: 0, yes: "Right. Too small is slow, too big is wild. The best is in between." },
    { say: "Real models have millions or billions of numbers, not two. The valley has billions of directions. But the rule is still the same: feel the slope, step downhill, repeat." },
    { recap: ["Imagine the loss as a valley.", "<b>Gradient descent</b>: feel the slope, step downhill, repeat.", "The step size is the <b>learning rate</b>: too big overshoots, too small is slow.", "The same rule trains the biggest AI models."], words: [["gradient", "the slope: which way is uphill, and how steep"], ["gradient descent", "stepping downhill again and again to lower the loss"], ["learning rate", "how big each step is"]] },
  ],

  neuron: [
    { say: "A line can only do simple things. To do more, we copy a tiny idea from the brain: the <b>neuron</b>." },
    { say: "A neuron does three things. <b>1.</b> Multiply each input by a number called a <b>weight</b>. <b>2.</b> Add them up, plus one extra number called the <b>bias</b>. <b>3.</b> Squash the result to between 0 and 1." },
    { ask: "If a weight is big, what does that input do?", options: ["It matters a lot", "It matters less", "It's ignored"], answer: 0, yes: "Yes. Weights say how much each input matters. Negative weights push the answer down." },
    { say: "Here's a neuron with two switches. Tap the switches to turn them on and off. Your job: make the light come on <b>only when both switches are on</b>. Watch the table.", viz: { type: "neuron", target: "and" }, try: true, done: "You built an AND gate out of a neuron! Big weights plus a negative bias did it." },
    { ask: "Why did the bias need to be negative?", options: ["So one switch alone isn't enough to turn on the light", "Biases are always negative"], answer: 0, yes: "Exactly. The bias is like a bar the inputs must get over." },
    { ask: "Who will choose the weights in a real model?", options: ["Gradient descent, from examples", "A person, by hand", "Nobody, they stay random"], answer: 0, yes: "Yes. You just did by hand what training does by itself, millions of times." },
    { recap: ["A <b>neuron</b> multiplies inputs by <b>weights</b>, adds a <b>bias</b>, then squashes.", "Weights say how much each input matters.", "The bias is a bar to get over.", "Training finds the weights for us."], words: [["neuron", "a tiny unit: multiply, add, squash"], ["weight", "how much one input matters"], ["bias", "an extra number that shifts the answer up or down"]] },
  ],
};
