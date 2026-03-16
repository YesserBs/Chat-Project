from groq import Groq
from dotenv import load_dotenv
import os

# You need a .env file containing your API key like this:
# GROQ_API_KEY=your_api_key_here

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

client = Groq(api_key=api_key)

text = input("Paste the text you want to summarize:\n")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes text clearly and concisely."
        },
        {
            "role": "user",
            "content": f"Summarize the following text in about one sentence:\n\n{text}"
        }
    ]
)

print("\nSummary:\n")
print(response.choices[0].message.content)

'''
i tested with this prompt:
LIVE: Results of the 2026 municipal elections – incumbent left-wing mayors rule out any alliance with LFI in Marseille and BordeauxAlliances, mergers, withdrawals, or maintaining candidacies… The day after a vote marked by a local breakthrough of La France Insoumise (LFI) and very strong scores for the National Rally (RN) in Marseille, political realignments on both the left and the right have begun—or, conversely, appear difficult.In Avignon, the LFI list merges with the left-wing list led by the Socialist PartyThe La France Insoumise (LFI) list led by Mathilde Louvain in Avignon will merge with the list led by Socialist Party (PS) candidate David Fournier, which finished respectively fourth and third in the city, according to both parties contacted by Agence France-Presse (AFP) on Monday.“There will be only one left-wing list” in the second round, confirmed a close associate of socialist candidate David Fournier to AFP. This is considered a necessary condition to keep Avignon under left-wing leadership.The surprise candidate of this election, television journalist Olivier Galzi, is currently in the lead with more than 27% of the vote, followed by RN candidate Anne-Sophie Rigault, with over 25%.
'''