### Adjusting the plan

- i want a tool that summerizes text, it will be used to summerize the conversation, to pass it as very brief context to the next request.
- know where and how i'll be storing the Questions&Answers - the common ones, that would eventually be returned to the user if he makes general/simple requests like "Hi !" or "How are you ?", i'll most likely use HuggingFace to store the data.. we'll see that later in detail
- I will most likely use Groq - llama as a chatbot api, it is so efficient
- And ofc the front-end, i should have something similar to those typical LLM websites. 

Algorithmically i will be:
- balancing the requests (many users might be requesting a response from the API)
- balance the context provision, i mean i should know when to pass it, not for every single request