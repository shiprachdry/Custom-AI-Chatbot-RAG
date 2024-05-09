# AI-Chatbot-RAG


We have all experimented with LLM models like ChatGPT, and one of the reasons it is so popular is
because of its user friendly interaction in human lang uage . A chatbot interface hides all the underlying
layers and functions, such as how each word is converted into vectors of numbers of 200+ dimensions
known as text embeddings (LLM models like SpaCy and OpenAI ChatGPT have different dimension
lengths)lengths). These numbers represent weights that are updated through numerous iterations in the multi layer
neural network by adjusting gradients to minimize the los s function or to maximize the likelihood that word
“w” appears in the context of word “c” or the context “c” given the input word, as it is trained on a large
corpus of words.One common question is, “Can I use ChatGPT on my own data? Can it answer by refe rencing the
local document instead of relying on its training data?”

Our business case aims to build a ChatGPT like chatbot trained on local context that not only uses the
relevant context from vast text data from pdfs but also presents it in natural language leveraging LLM.

****
Through this prototype, we would like to use AI chatbot as our key lever for adoption of mental health
resources at work place The AI chatbot uses local context & health resources from PDFs/documents,
asks relevant questions, and res ponds with empathy and compassion acting as a personal
companion
