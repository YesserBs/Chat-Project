### Adjusting the plan

- i want a tool that summerizes text, it will be used to summerize the conversation, to pass it as very brief context to the next request.
`llama-3.1-8b-instant` is a good candidate
- And ofc the front-end, i should have something similar to those typical LLM websites.

Algorithmically i will be:
- balancing the requests (many users might be requesting a response from the API)
- understand how many requests i can make per minute and how i can maximize & optimize it
- balance the context provision, i mean i should know when to pass it, not for every single request